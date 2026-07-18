import json
from typing import Type
import torch
from torch.utils.data import Dataset, DataLoader

from evolix.data.tokenizer import Tokenizer
from evolix.config.finetune_config import FinetuneConfig

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt_ids = self.tokenizer.encode(ex["prompt"])
        completion_ids = self.tokenizer.encode(ex["completion"], add_eos=True)

        ids = (prompt_ids + completion_ids)[: self.max_seq_len + 1]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + completion_ids)[: self.max_seq_len + 1]

        pad_len = self.max_seq_len + 1 - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        ids_t = torch.tensor(ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return ids_t[:-1], labels_t[1:]


class SFTDataManager:
    def __init__(self, cfg: Type[FinetuneConfig], tokenizer: Tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def build_loader(self) -> DataLoader:
        ds = SFTDataset(self.cfg.sft_data_path, self.tokenizer, self.cfg.max_seq_len)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
