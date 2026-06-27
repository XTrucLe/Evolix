import torch


def get_raw_model(model):
    while hasattr(model, "module"):
        model = model.module
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)

    return obj
