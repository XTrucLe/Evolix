from typing import Type
import torch
import bitsandbytes as bnb

from evolix.config.config import Config
from evolix.models.architecture import Evolix
from evolix.utils.system import SM
from evolix.engine.distributed import setup_distributed


class EngineBase:
    def __init__(self, cfg: Type[Config]):
        self.cfg = cfg

    def build_model(self) -> Evolix:
        return Evolix(cfg=self.cfg)

    def build_optimizer(self, model):
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_no_decay = p.ndim < 2 or "bias" in name.lower() or "norm" in name.lower()
            (no_decay if is_no_decay else decay).append(p)

        params = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        use_8bit = self.cfg.optimizer == "8bit" or (self.cfg.optimizer == "auto" and SM < 90)

        if use_8bit:
            return bnb.optim.AdamW8bit(params, lr=self.cfg.lr, betas=self.cfg.betas)

        return torch.optim.AdamW(params, lr=self.cfg.lr, betas=self.cfg.betas, fused=True)

    def configure_optimization_backends(self, device):
        if device.type != "cuda":
            return

        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            if SM >= 90:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_cudnn_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            else:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_cudnn_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
        else:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    def setup(self):
        device, ddp, world_size, master_process, local_rank = setup_distributed(self.cfg.seed)
        self.configure_optimization_backends(device)

        return device, ddp, world_size, master_process, local_rank

    def maybe_compile(self, model: Evolix):
        if self.cfg.compile and hasattr(torch, "compile"):
            torch._inductor.config.coordinate_descent_tuning = True
            model = torch.compile(model, mode="max-autotune")
        return model
