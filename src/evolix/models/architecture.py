import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from evolix.config.config import Config
from evolix.models.layers import Block


class Evolix(nn.Module):
    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__()
        self.use_gc = cfg.grad_checkpoint

        self.dim = cfg.dim
        self.embedding = nn.Embedding(cfg.vocab_size, self.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln_f = nn.RMSNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, cfg.vocab_size, bias=False)
        self._init_weights_all(cfg.layers)
        self.lm_head.weight = self.embedding.weight

    def _init_weights_all(self, layers: int):
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

        use_cache = kv_caches is not None or (y is None and not self.training)
        new_kv_caches = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            if self.use_gc and self.training:
                h = checkpoint(block, h, offset, None, use_reentrant=False)[0]
            else:
                past_cache = kv_caches[i] if (kv_caches is not None) else None

                h, new_cache = block(h, offset=offset, kv_cache=past_cache)

                if use_cache:
                    new_kv_caches.append(new_cache)

        logits = self.lm_head(self.ln_f(h))

        if y is None:
            return logits, new_kv_caches

        assert y.shape == (B, T)

        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    def num_params(self) -> str:
        fmt = lambda n: f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.2f}M"

        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad) - self.embedding.weight.numel()

        return f"{fmt(train)} / {fmt(total)}"
