from typing import Type
import torch

from evolix.config.config import Config
from evolix.engine.base import EngineBase
from evolix.data.tokenizer import Tokenizer
from evolix.utils.sampling import sample_next


class Generator(EngineBase):
    def __init__(self, cfg: Type[Config], checkpoint_manager, tokenizer: Tokenizer):
        super().__init__(cfg)
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer = tokenizer
        self._model = None
        self._device = None

    def load(self) -> "Generator":
        device, *_ = self.setup()
        model_dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float32

        model = self.build_model().to(device, dtype=model_dtype)
        self.checkpoint_manager.load(model=model, optimizer=None, scaler=None)
        model.eval()

        self._model, self._device = model, device
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> str:
        assert self._model is not None, "Call .load() before .generate()."
        device = self._device
        model_dtype = next(self._model.parameters()).dtype

        ids = self.tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        kv_caches = [block.attn.init_cache(1, device, model_dtype) for block in self._model.blocks]

        logits, kv_caches = self._model(x, kv_caches=kv_caches, offset=0)
        next_id = sample_next(logits[:, -1], temperature, top_k, top_p)

        generated = [next_id.item()]
        offset = x.shape[1]
        cur = next_id

        for _ in range(max_new_tokens - 1):
            if generated[-1] == self.tokenizer.eos_id:
                break
            logits, kv_caches = self._model(cur, kv_caches=kv_caches, offset=offset)
            next_id = sample_next(logits[:, -1], temperature, top_k, top_p)
            generated.append(next_id.item())
            cur = next_id
            offset += 1

        return self.tokenizer.decode(generated)
