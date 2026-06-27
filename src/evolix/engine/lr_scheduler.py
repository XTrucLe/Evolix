import math


def get_lr(step: int, cfg):
    warmup = cfg.warmup_steps
    total = cfg.total_steps
    base_lr = cfg.lr
    min_lr = cfg.min_lr

    if step < warmup:
        return base_lr * (step + 1) / warmup

    progress = (step - warmup) / (total - warmup)
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return max(base_lr * cosine, min_lr)
