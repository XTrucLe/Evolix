from typing import List
from tokenizers import Tokenizer as BPE_Tokenizer


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = BPE_Tokenizer.from_file(model_path)
        self.pad_id = self.tokenizer.token_to_id("<pad>") if self.tokenizer.token_to_id("<pad>") is not None else 0
        self.eos_id = self.tokenizer.token_to_id("</s>") if self.tokenizer.token_to_id("</s>") is not None else 3

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i != self.pad_id]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
