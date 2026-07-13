from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class Config:
    # --- MODEL ARCHITECTURE ---
    vocab_size: int = 65536
    spm_prefix: str = "evolix"
    block_size: int = 8192
    layers: int = 24
    heads: int = 20
    dim: int = 2560
    ffn_dim: int = 8192
    kv_lora_rank: int = 512
    rope_dim: int = 64
    rope_theta: float = 25600.0
    dropout: float = 0.0
    bias: bool = False
    grad_checkpoint: bool = False
    compile: bool = True
    # --- DATA & TRAINING SYSTEM ---
    data_split: str = "train"
    batch_size: int = 4
    grad_accum: int = 16
    dtype: str = "bfloat16"
    num_workers: int = 8
    prefetch_factor: int = 4
    chunk_size: int = 64
    shuffle_buffer: int = 512
    # --- CHECKPOINT & LOG ---
    checkpoint_dir: str = "evolix/checkpoints"
    resume: bool = True
    short_run: bool = False
    short_run_steps: int = 50
    save_every: int = 200
    log_every: int = 10
    seed: int = 55
    # --- OPTIMIZER & SCHEDULER ---
    total_steps: int = 100_000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    optimizer: str = "8bit"  # "adamw", "8bit", "auto"
    # --- HUGGINGFACE HUB ---
    hf_dataset_repo: str = "trucle5503/dataset_pretrain"
    hf_repo_id: str = "trucle5503/Evolix"
    hf_token: str = ""
