from dataclasses import dataclass
from typing import Tuple

from evolix.config.config import Config

@dataclass(slots=True)
class FinetuneConfig(Config):
    sft_data_path: str = "data/sft_train.jsonl"
    max_seq_len: int = 4096
 
    base_checkpoint_repo: str = ""
    total_steps: int = 1000
    warmup_steps: int = 20
    lr: float = 2e-5
    min_lr: float = 2e-6
    save_every: int = 200