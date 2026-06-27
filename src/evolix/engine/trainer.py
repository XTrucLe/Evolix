from math import dist
from typing import Type

import torch
import bitsandbytes as bnb

from evolix.config import Config
from evolix.models.architecture import Evolix
from evolix.utils.system import SM, SDPA_BACKEND
from evolix.utils.logger import task_queue

from evolix.engine.distributed import setup_distributed, wrap_ddp, cleanup_distributed
from evolix.engine.lr_scheduler import get_lr


class Trainer:
    def __init__(self, cfg: Type[Config], data_manager, checkpoint_manager):
        self.cfg = cfg
        self.data_manager = data_manager
        self.checkpoint_manager = checkpoint_manager

    def _build_model(self) -> Evolix:
        return Evolix(
            vocab_size=self.cfg.vocab_size,
            dim=self.cfg.dim,
            lora_rank=self.cfg.lora_rank,
            layers=self.cfg.layers,
            heads=self.cfg.heads,
            block_size=self.cfg.block_size,
            dropout=self.cfg.dropout,
            grad_checkpoint=self.cfg.grad_checkpoint,
            rope_dim=self.cfg.rope_dim,
        )

    def _build_optimizer(self, model: Evolix) -> bnb.optim.AdamW8bit:
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_no_decay = p.ndim < 2 or "bias" in name or "norm" in name
            (no_decay if is_no_decay else decay).append(p)
        return bnb.optim.AdamW8bit(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
        )

    def _configure_optimization_backends(self, device):
        if device.type != "cuda":
            return

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            if SM >= 90:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_cudnn_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cudnn.enabled = True
            else:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
        else:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cudnn.enabled = True

    def _print_config_info(self, master_process, model, device, scaler, tokens_per_step, ddp, world_size, step, start_step):
        """In thông tin cấu hình hệ thống một cách trực quan."""
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
        device, ddp, world_size, master_process, local_rank = setup_distributed(self.cfg.seed)
        is_cuda = device.type == "cuda"

        self._configure_optimization_backends(device)

        amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(self.cfg.dtype, torch.float32)
        scaler = torch.amp.GradScaler(enabled=(is_cuda and self.cfg.dtype == "float16"))

        model = self._build_model().to(device, dtype=amp_dtype)
        optimizer = self._build_optimizer(model)

        loader = self.data_manager.build_loader()
        start_step = step = self.checkpoint_manager.load(model=model, optimizer=optimizer, scaler=scaler, dataset=loader.dataset, ddp_world_size=world_size) + 1 if self.cfg.resume else 0
        raw_model = model

        if self.cfg.compile and hasattr(torch, "compile"):
            if master_process:
                print("Compiling model via Inductor...")
            torch._inductor.config.triton.cudagraphs = False
            torch._inductor.config.coordinate_descent_tuning = True
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=True)

        model = wrap_ddp(model, ddp, local_rank)
        if ddp:
            raw_model = model.module

        data_iter = iter(loader)
        model.train()

        t0, t1 = None, None
        if is_cuda:
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        tokens_per_step = self.cfg.batch_size * self.cfg.block_size * self.cfg.grad_accum * world_size
        accum_loss_tensor = torch.zeros(1, device=device)
        loss_tensor = torch.zeros(1, device=device) if ddp else None

        self._print_config_info(master_process, raw_model, device, scaler, tokens_per_step, ddp, world_size, step, start_step)

        while step < self.cfg.total_steps:
            lr = get_lr(step, self.cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            do_log = step % self.cfg.log_every == 0
            do_save = step > 0 and step % self.cfg.save_every == 0

            optimizer.zero_grad(set_to_none=True)
            accum_loss_tensor.zero_()

            if is_cuda and do_log and master_process:
                t0.record()

            for _ in range(self.cfg.grad_accum):
                if ddp:
                    model.require_backward_grad_sync = _ == self.cfg.grad_accum - 1

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()

            # --- Ghi Log ---
            if do_log and master_process:
                if is_cuda:
                    t1.record()
                    torch.cuda.synchronize()
                    ms = t0.elapsed_time(t1)
                    msg = f"step {step:7d} | loss {accum_loss_val:8.5f} | lr {lr:9.2e} | {tokens_per_step / ms:8.2f}k tok/s | {ms:7.0f}ms"
                else:
                    msg = f"step {step:7d} | loss {accum_loss_val:8.5f} | lr {lr:9.2e}"
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
