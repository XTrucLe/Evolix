import math

from evolix.config.config import Config


def get_lr(step: int, cfg: Config):
    w, t = cfg.warmup_steps, max(cfg.total_steps, cfg.warmup_steps + 1)
    base, min_lr = cfg.lr, cfg.min_lr

    if step < w:
        return base * (step + 1) / w if w else base

    p = min(max((step - w) / (t - w), 0.0), 1.0)
    return max(base * 0.5 * (1 + math.cos(math.pi * p)), min_lr)
