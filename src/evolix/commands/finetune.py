import os, dataclasses

from evolix.config.finetune_config import FinetuneConfig
from evolix.data.tokenizer import Tokenizer
from evolix.data.sft_dataset import SFTDataManager
from evolix.engine.finetune import FinetuneTrainer
from evolix.utils.checkpoint import CheckpointManager


def finetune():
    try:
        cfg = FinetuneConfig()
        cfg.hf_token = os.environ.get("HF_TOKEN", None) if not cfg.hf_token else cfg.hf_token
        if not cfg.hf_token:
            raise ValueError("Hugging Face token is not set. Please set the 'hf_token' in the configuration.")

        tokenizer = Tokenizer(f"{cfg.spm_prefix}.model")
        data_manager = SFTDataManager(cfg, tokenizer)

        checkpoint_manager = CheckpointManager(cfg)

        base_checkpoint_manager = None
        if cfg.base_checkpoint_repo:
            base_cfg = dataclasses.replace(cfg, hf_repo_id=cfg.base_checkpoint_repo)
            base_checkpoint_manager = CheckpointManager(base_cfg, require_write=False)

        trainer = FinetuneTrainer(cfg, data_manager, checkpoint_manager, base_checkpoint_manager)
        trainer.run()
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    finetune()
