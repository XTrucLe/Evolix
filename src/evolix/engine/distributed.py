import os
import torch
import torch.distributed as dist


def setup_distributed(seed: int):
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        world_size = int(os.environ["WORLD_SIZE"])
        master_process = int(os.environ["RANK"]) == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        master_process = True
        local_rank = None

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    return device, ddp, world_size, master_process, local_rank


def wrap_ddp(model, ddp: bool, local_rank: int):
    if not ddp:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(model, device_ids=[local_rank])


def cleanup_distributed(ddp: bool):
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
