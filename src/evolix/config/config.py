from dataclasses import dataclass


@dataclass(slots=True)
class Config:
    # --- MODEL ARCHITECTURE ---
    vocab_size: int = 65536
    spm_prefix: str = "evolix"

    block_size: int = 8192
    layers: int = 32
    dim: int = 2304
    heads: int = 18
    ffn_dim: int = 6144

    # --- ATTENTION & POSITION ENCODING ---
    kv_lora_rank: int = 512
    rope_dim: int = 64
    rope_theta: float = 128000.0
    epsilon: float = 1e-6

    # --- REGULARIZATION & RUNTIME ---
    dropout: float = 0.0
    bias: bool = False
    grad_checkpoint: bool = False
    compile: bool = True
    dtype: str = "bfloat16"

    # --- DATA PIPELINE ---
    data_split: str = "train"
    batch_size: int = 4
    grad_accum: int = 32

    num_workers: int = 2
    prefetch_factor: int = 4

    chunk_size: int = 64
    shuffle_buffer: int = 512

    # --- OPTIMIZER & SCHEDULER ---
    optimizer: str = "8bit"  # "adamw", "8bit", "auto"

    lr: float = 3e-4
    min_lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    total_steps: int = 120_000
    warmup_steps: int = 1200

    # --- CHECKPOINT & LOGGING ---
    checkpoint_dir: str = "evolix/checkpoints"
    resume: bool = True

    save_every: int = 200
    log_every: int = 10

    short_run: bool = False
    short_run_steps: int = 50

    seed: int = 55

    # --- HUGGINGFACE HUB ---
    hf_dataset_repo: str = "trucle5503/dataset_pretrain"
    hf_repo_id: str = "trucle5503/Evolix"
    hf_token: str = ""
