from typing import Type
import torch
import torch.distributed as dist

from evolix.config import Config
from evolix.engine.base import EngineBase
from evolix.utils.logger import task_queue
from evolix.engine.distributed import wrap_ddp, cleanup_distributed
from evolix.engine.lr_scheduler import get_lr


class Trainer(EngineBase):
    def __init__(self, config: Type[Config], data_manager, checkpoint_manager):
        super().__init__(config)
        self.data_manager = data_manager
        self.checkpoint_manager = checkpoint_manager

    def _load_init_weights(self, model, optimizer, scaler, dataset, ddp_world_size) -> int:
        return self.checkpoint_manager.load(model, optimizer, scaler, dataset, ddp_world_size) + 1 if self.cfg.resume else 0

    def _log_config(self, master_process, model, device, scaler, tokens_per_step, ddp, world_size, step, start_step):
        if not master_process:
            return
        print("─" * 80)
        print("🚀 MODEL CONFIG :")
        print(f"   • Architecture: Dim = {self.cfg.dim} | Layers = {self.cfg.layers} | Heads = {self.cfg.heads}")
        print(f"   • Context Len : {self.cfg.block_size} tokens")
        print(f"   • Total Params: {model.num_params() if hasattr(model, 'num_params') else 'N/A'}")
        print(f"   • Gradient Checkpt: {'Enabled' if self.cfg.grad_checkpoint else 'Disabled'}")
        print("\n⚙️ TRAINING CONFIG:")
        print(f"   • Device/GPU  : {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
        print(f"   • Precision   : {self.cfg.dtype.upper()} {'(Gradient Scaler Enabled)' if scaler.is_enabled() else ''}")
        print(f"   • Total Batch : {tokens_per_step:,} tokens/step (Accum: {self.cfg.grad_accum})")
        print(f"   • DDP Mode    : {'Active' if ddp else 'Disabled'} ({world_size} GPU{'' if world_size == 1 else 's'})")
        steps_info = f"{step:,} → {start_step + self.cfg.short_run_steps:,} [SHORT RUN]" if self.cfg.short_run else f"{step:,} → {self.cfg.total_steps:,}"
        print(f"   • Total Steps : {steps_info}")
        print("─" * 80 + "\n")

    def run(self):
        device, ddp, world_size, master_process, local_rank = self.setup()
        is_cuda = device.type == "cuda"

        amp_dtype = torch.float16 if self.cfg.dtype == "float16" else torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float32
        scaler = torch.amp.GradScaler(enabled=is_cuda and self.cfg.dtype == "float16")
        model_type = torch.float32 if self.cfg.dtype == "float32" else amp_dtype

        model = self.build_model().to(device, dtype=model_type)
        optimizer = self.build_optimizer(model)
        loader = self.data_manager.build_loader()
        start_step = step = self._load_init_weights(model, optimizer, scaler, loader.dataset, world_size)
        if master_process:
            print("Compiling model via Inductor..." if self.cfg.compile else "Compile disabled.")
        model = self.maybe_compile(model)
        model = wrap_ddp(model, ddp, local_rank)
        raw_model = model.module if ddp else model

        data_iter = iter(loader)
        model.train()

        t0, t1 = None, None
        if master_process:
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        tokens_per_step = self.cfg.batch_size * self.cfg.block_size * self.cfg.grad_accum
        accum_loss_tensor = torch.zeros(1, device=device)
        loss_tensor = torch.zeros(1, device=device) if ddp else None
        accum_loss_val = 0.0

        self._log_config(master_process, raw_model, device, scaler, tokens_per_step, ddp, world_size, start_step, start_step)

        while step < self.cfg.total_steps:
            torch.compiler.cudagraph_mark_step_begin()
            lr = get_lr(step, self.cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            do_log = step % self.cfg.log_every == 0
            do_save = step > 0 and step % self.cfg.save_every == 0
            optimizer.zero_grad(set_to_none=True)
            accum_loss_tensor.zero_()

            if is_cuda and do_log and master_process:
                t0.record()

            for micro_step in range(self.cfg.grad_accum):
                if ddp:
                    model.require_backward_grad_sync = micro_step == self.cfg.grad_accum - 1

                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss = model(x, y)
                scaled_loss = loss / self.cfg.grad_accum

                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accum_loss_tensor.add_(scaled_loss.detach())

            if ddp:
                loss_tensor.fill_(accum_loss_tensor)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                accum_loss_val = loss_tensor.item()
            else:
                accum_loss_val = accum_loss_tensor.item()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()

            if do_log and master_process:
                grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                if is_cuda:
                    t1.record()
                    torch.cuda.synchronize()
                    ms = t0.elapsed_time(t1)
                    msg = f"step {step:7d} | loss {accum_loss_val:8.5f} | gnorm {grad_norm_val:6.2f} | lr {lr:9.2e} | {tokens_per_step / ms:8.2f}k tok/s | {ms:7.0f}ms"
                else:
                    msg = f"step {step:7d} | loss {accum_loss_val:8.5f} | gnorm {grad_norm_val:6.2f} | lr {lr:9.2e}"
                task_queue.put({"type": "log", "data": msg})

            if self.cfg.short_run and step >= start_step + self.cfg.short_run_steps:
                if master_process:
                    print(f"Short run complete: ran {self.cfg.short_run_steps} steps starting from step {start_step}.")
                break

            if do_save and master_process:
                self.checkpoint_manager.save(raw_model, optimizer, scaler, step, accum_loss_val, world_size)

            step += 1

        if master_process:
            self.checkpoint_manager.save(raw_model, optimizer, scaler, step, accum_loss_val, world_size)
            task_queue.join()
            print("Training complete")

        cleanup_distributed(ddp)
