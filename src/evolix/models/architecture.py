import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from evolix.config.config import Config
from evolix.models.layers import Block


class Evolix(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.grad_checkpoint = cfg.grad_checkpoint

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln_f = nn.RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self._init_weights_all(cfg.layers)

    def _init_weights_all(self, layers):
        self.apply(self._init_module)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        offset: int = 0,
    ):
        B, T = x.shape
        h = self.drop(self.embedding(x))

        is_inference = y is None
        new_kv_caches = [] if is_inference else None

        for i, block in enumerate(self.blocks):
            if self.grad_checkpoint and self.training:
                h, _ = checkpoint(block, h, offset, None, use_reentrant=False)
            else:
                past_cache = kv_caches[i] if kv_caches is not None else None
                h, new_cache = block(h, offset=offset, kv_cache=past_cache)
                if new_kv_caches is not None:
                    new_kv_caches.append(new_cache)

        logits = self.lm_head(self.ln_f(h))

        if is_inference:
            return logits, new_kv_caches

        assert y is not None and y.shape == (B, T)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    def num_params(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"{total / 1e9:.2f}B" if total >= 1e9 else f"{total / 1e6:.2f}M"
