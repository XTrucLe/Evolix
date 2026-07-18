import json
import os
from typing import Type

import torch
from huggingface_hub.utils import disable_progress_bars, logging
from safetensors.torch import load_file, save_file

from evolix.config.config import Config
from evolix.utils.serialization import get_raw_model, to_cpu

logging.set_verbosity_error()
disable_progress_bars()


class CheckpointManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.repo_id = self.cfg.hf_repo_id
        self.token = self.cfg.hf_token
        if not self.token:
            raise ValueError("Hugging Face token is not set. Please set the 'hf_token' in the configuration.")

        self.local_temp_dir = os.path.join(self.cfg.checkpoint_dir, "hf_staging")
        os.makedirs(self.local_temp_dir, exist_ok=True)

        from huggingface_hub import HfApi

        self.api = HfApi()

    def save(self, model, optimizer, scaler, step, loss, ddp_world_size=1):
        raw_model = get_raw_model(model)

        raw_state = raw_model.state_dict()
        clean_state = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in raw_state.items()}
        cpu_clean_state = to_cpu(clean_state)
        cpu_clean_state = {k: v.contiguous() for k, v in cpu_clean_state.items()}

        config_data = {
            "architectures": ["EvolixForCausalLM"],
            "model_type": "evolix",
            "vocab_size": self.cfg.vocab_size,
            "hidden_size": self.cfg.dim,
            "intermediate_size": self.cfg.ffn_dim,
            "num_hidden_layers": self.cfg.layers,
            "num_attention_heads": self.cfg.heads,
            "max_position_embeddings": self.cfg.block_size,
            "kv_lora_rank": self.cfg.kv_lora_rank,
            "rope_dim": self.cfg.rope_dim,
            "rope_theta": self.cfg.rope_theta,
            "torch_dtype": self.cfg.dtype,
            "tie_word_embeddings": False,
        }
        config_path = os.path.join(self.local_temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        safetensors_path = os.path.join(self.local_temp_dir, "model.safetensors")
        save_file(
            cpu_clean_state,
            safetensors_path,
        )

        current_global_batch_size = self.cfg.batch_size * self.cfg.grad_accum * ddp_world_size
        samples_trained = step * current_global_batch_size

        training_state_path = os.path.join(self.local_temp_dir, "training_state.pt")
        torch.save(
            {
                "step": step,
                "batch_size": self.cfg.batch_size,
                "grad_accum": self.cfg.grad_accum,
                "samples_trained": samples_trained,
                "loss": float(loss),
                "optimizer": to_cpu(optimizer.state_dict()),
                "scaler": to_cpu(scaler.state_dict()) if scaler.is_enabled() else None,
            },
            training_state_path,
        )

        try:
            self.api.upload_folder(
                folder_path=self.local_temp_dir,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Checkpoint at step {step}",
            )
            print("✅ Checkpoint uploaded to Hugging Face Hub successfully.")
        except Exception:
            print("❌ Failed to upload checkpoint to Hugging Face Hub.")

    def load(self, model, optimizer=None, scaler=None, dataset=None, ddp_world_size=1) -> int:
        from huggingface_hub import hf_hub_download

        try:
            safetensors_path = hf_hub_download(repo_id=self.repo_id, filename="model.safetensors", token=self.token)
            training_state_path = hf_hub_download(repo_id=self.repo_id, filename="training_state.pt", token=self.token)
        except Exception:
            print("❌ No checkpoint found on Hugging Face Hub. Starting from scratch.")
            return 0

        state = load_file(safetensors_path, device="cpu")

        raw_model = get_raw_model(model)
        raw_model.load_state_dict(state)

        ckpt = torch.load(training_state_path, map_location="cpu", weights_only=True)
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])

        if ckpt.get("scaler") and len(ckpt["scaler"]) > 0:
            if hasattr(scaler, "is_enabled") and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])

        samples_trained = ckpt.get(
            "samples_trained",
            ckpt["step"] * self.cfg.batch_size * self.cfg.grad_accum * ddp_world_size,
        )

        if dataset and hasattr(dataset, "set_resume_state"):
            dataset.set_resume_state(samples_trained)

        rescaled_step = samples_trained // (self.cfg.batch_size * self.cfg.grad_accum * ddp_world_size)

        print("✅ Checkpoint loaded successfully.")
        return rescaled_step
