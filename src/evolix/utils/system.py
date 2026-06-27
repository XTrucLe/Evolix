import torch


def gpu_info():
    if not torch.cuda.is_available():
        return {"arch": None, "bf16": False, "sm": None}

    major, minor = torch.cuda.get_device_capability(0)
    sm = major * 10 + minor
    bf16 = torch.cuda.is_bf16_supported()

    if major == 7:
        arch = "turing/volta"
    elif major == 8:
        arch = "ampere"
    elif major == 9:
        arch = "hopper"
    elif major >= 12:
        arch = "blackwell"
    else:
        arch = "unknown"

    return {"arch": arch, "bf16": bf16, "sm": sm}


def best_sdpa_backend(sm: int):
    from torch.nn.attention import SDPBackend

    if torch.backends.cuda.flash_sdp_enabled():
        return SDPBackend.FLASH_ATTENTION
    if sm >= 90 and torch.backends.cuda.cudnn_sdp_enabled():
        return SDPBackend.CUDNN_ATTENTION
    if torch.backends.cuda.mem_efficient_sdp_enabled():
        return SDPBackend.EFFICIENT_ATTENTION
    return SDPBackend.MATH


ARCH, BF16, SM = gpu_info().values()
SDPA_BACKEND = best_sdpa_backend(SM)
