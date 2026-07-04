from typing import List
import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else self.sp.vocab_size() - 1

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i != self.pad_id]
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()