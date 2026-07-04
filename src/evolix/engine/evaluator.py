from typing import Type, Optional
import torch
import torch.distributed as dist

from evolix.config.config import Config
from evolix.engine.base import EngineBase
from evolix.engine.distributed import cleanup_distributed
from evolix.utils.metrics import RunningAverage, perplexity


class Evaluator(EngineBase):
    def __init__(self, cfg: Type[Config], data_manager, checkpoint_manager):
        super().__init__(cfg)
        self.data_manager = data_manager
        self.checkpoint_manager = checkpoint_manager

    @torch.no_grad()
    def run(self, max_batches: Optional[int] = None) -> dict:
        device, ddp, _, master_process, _ = self.setup()
        model_dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
        amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(self.config.dtype, torch.float32)

        model = self.build_model().to(device, dtype=model_dtype)
        self.checkpoint_manager.load(model=model, optimizer=None, scaler=None)
        model.eval()

        loader = self.data_manager.build_loader()
        loss_avg = RunningAverage()

        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = model(x, y)
            loss_avg.update(loss.item())

        if ddp:
            t = torch.tensor([loss_avg.avg], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg_loss = t.item()
        else:
            avg_loss = loss_avg.avg

        result = {"loss": avg_loss, "perplexity": perplexity(avg_loss), "batches": loss_avg.count}

        if master_process:
            print(f"[eval] batches={result['batches']} loss={result['loss']:.5f} ppl={result['perplexity']:.3f}")

        cleanup_distributed(ddp)
        return result