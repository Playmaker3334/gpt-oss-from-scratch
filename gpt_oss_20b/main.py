# main.py
"""
Main training script for GPT-OSS - Multi-GPU Version
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import argparse
from typing import Optional, Dict
import json
import time
import psutil
from collections import deque

from config.model_config import GPTOSSConfigMini
from src.model.gpt_oss import GPTOSSForCausalLM
from src.tokenizer.harmony_tokenizer import HarmonyTokenizer
from src.utils.tensor_utils import get_device, count_parameters
from src.utils.math_utils import warmup_cosine_schedule, compute_grad_norm


OUTPUT_DIR = "./output"
TOKENIZER_SAVE_DIR = "./output/tokenizer"

BATCH_SIZE = 1
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
MAX_STEPS = 2000
GRAD_ACCUM_STEPS = 8
MAX_GRAD_NORM = 1.0
EVAL_STEPS = 200
SAVE_STEPS = 500

USE_WANDB = False
MIXED_PRECISION = False
GRADIENT_CHECKPOINTING = True

DEFAULT_CONFIG = GPTOSSConfigMini()
PAD_ID = DEFAULT_CONFIG.pad_token_id  # usado en collate_fn


class MetricsTracker:
    """Tracks training metrics with rolling averages"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.perplexities = deque(maxlen=window_size)
        self.accuracies = deque(maxlen=window_size)
        self.grad_norms = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)

        self.global_step = 0
        self.tokens_seen = 0
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.batch_size = 1
        self.seq_len = 2048

    def update(self, metrics: Dict):
        if 'loss' in metrics:
            self.losses.append(metrics['loss'])
            self.perplexities.append(torch.exp(torch.tensor(metrics['loss'])).item())
        if 'accuracy' in metrics:
            self.accuracies.append(metrics['accuracy'])
        if 'grad_norm' in metrics:
            self.grad_norms.append(metrics['grad_norm'])
        if 'lr' in metrics:
            self.learning_rates.append(metrics['lr'])

    def report(self) -> Dict:
        def avg(x):
            return sum(x) / len(x) if len(x) > 0 else 0.0

        elapsed = time.time() - self.start_time
        steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0.0
        tokens_per_step = self.batch_size * self.seq_len
        tps = steps_per_sec * tokens_per_step

        return {
            "loss": avg(self.losses),
            "ppl": avg(self.perplexities),
            "acc": avg(self.accuracies),
            "grad_norm": avg(self.grad_norms),
            "lr": avg(self.learning_rates),
            "steps_per_sec": steps_per_sec,
            "tokens_per_sec": tps,
            "global_step": self.global_step,
            "uptime_s": int(elapsed),
            "ram_gb": psutil.virtual_memory().used / (1024**3),
        }


class TextFileDataset(Dataset):
    """Dataset simple que lee un archivo de texto y devuelve ids tokenizados por línea."""
    def __init__(self, file_path: str, tokenizer: HarmonyTokenizer, seq_len: int = 2048):
        self.file_path = file_path
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [x.strip() for x in f.readlines() if x.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx: int):
        text = self.lines[idx]
        ids = self.tokenizer.encode(text)
        # For simplicity, pad/truncate to seq_len
        if len(ids) < self.seq_len:
            ids = ids + [PAD_ID] * (self.seq_len - len(ids))
        else:
            ids = ids[:self.seq_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    attn_mask = (input_ids != PAD_ID).long()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn_mask
    }


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
    state = {
        "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }
    torch.save(state, ckpt_path)
    print(f"[INFO] Checkpoint saved at: {ckpt_path}")


def load_checkpoint_if_available(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], output_dir: str):
    if not os.path.isdir(output_dir):
        return 0
    files = [f for f in os.listdir(output_dir) if f.startswith("checkpoint_step_") and f.endswith(".pt")]
    if not files:
        return 0
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest = files[-1]
    path = os.path.join(output_dir, latest)
    print(f"[INFO] Loading checkpoint from {path}")
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model"], strict=False)
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    return int(data.get("step", 0))


def setup_tokenizer(tokenizer_dir: str) -> HarmonyTokenizer:
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer = HarmonyTokenizer()
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer


def create_dataloaders(train_path: str,
                       eval_path: Optional[str],
                       tokenizer: HarmonyTokenizer,
                       batch_size: int,
                       seq_len: int) -> Dict[str, Optional[DataLoader]]:

    train_ds = TextFileDataset(train_path, tokenizer=tokenizer, seq_len=seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    if eval_path and os.path.isfile(eval_path):
        eval_ds = TextFileDataset(eval_path, tokenizer=tokenizer, seq_len=seq_len)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        eval_loader = None

    return {"train": train_loader, "eval": eval_loader}


def configure_model(default_config: GPTOSSConfigMini,
                    gradient_checkpointing: bool = True) -> GPTOSSForCausalLM:
    model = GPTOSSForCausalLM(
        vocab_size=default_config.vocab_size,
        hidden_size=default_config.hidden_size,
        num_layers=default_config.num_layers,
        num_attention_heads=default_config.num_attention_heads,
        num_key_value_heads=default_config.num_key_value_heads,
        intermediate_size=default_config.intermediate_size,
        num_experts=default_config.num_experts,
        num_experts_per_token=default_config.num_experts_per_token,
        max_position_embeddings=default_config.max_position_embeddings,
        rope_theta=default_config.rope_theta,
        use_attention_sinks=default_config.use_attention_sinks,
        attention_sink_size=default_config.attention_sink_size,
        use_sparse_attention=default_config.use_sparse_attention,
        sparse_window_size=default_config.sparse_window_size,
        rms_norm_eps=default_config.rms_norm_eps,
        aux_loss_coef=default_config.aux_loss_coef,
        router_jitter_noise=default_config.router_jitter_noise,
        pad_token_id=default_config.pad_token_id,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


class Trainer:
    def __init__(self,
                 model: GPTOSSForCausalLM,
                 tokenizer: HarmonyTokenizer,
                 device: torch.device,
                 train_loader: DataLoader,
                 eval_loader: Optional[DataLoader],
                 batch_size: int,
                 learning_rate: float,
                 warmup_steps: int,
                 max_steps: int,
                 grad_accum_steps: int,
                 max_grad_norm: float,
                 mixed_precision: bool = False,
                 use_wandb: bool = False):

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.use_wandb = use_wandb

        self.is_multi_gpu = torch.cuda.device_count() > 1 and isinstance(self.model, torch.nn.Module)
        if self.is_multi_gpu:
            print(f"[INFO] Using DataParallel over {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        self.global_step = 0
        self.step_in_epoch = 0
        self.metrics = MetricsTracker(window_size=100)
        self.metrics.batch_size = batch_size
        self.metrics.seq_len = DEFAULT_CONFIG.max_position_embeddings

        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None

        if use_wandb:
            try:
                import wandb as _wandb
                self._wandb = _wandb
                self._wandb.init(project="gpt-oss-training")
            except Exception as e:
                print(f"[WARN] W&B not available: {e}")
                self._wandb = None
        else:
            self._wandb = None

        # LR scheduler
        self.lr_scheduler = warmup_cosine_schedule(learning_rate, warmup_steps, max_steps)

        # Build eval loader if not provided
        if self.eval_loader is None:
            self.eval_loader = None

        # Optimizer
        model_params = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        self.optimizer = AdamW(
            model_params, 
            lr=learning_rate, 
            betas=(0.9, 0.98), 
            weight_decay=0.01, 
            eps=1e-6
        )

        # AMP scaler (solo si se pide por flag)
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None

        if self.use_wandb:
            try:
                import wandb as _wandb
                self._wandb = _wandb
                self._wandb.init(
                    project="gpt-oss-training",
                    config={
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "max_steps": max_steps,
                        "model_params": count_parameters(model),
                        "multi_gpu": self.is_multi_gpu,
                    }
                )
            except Exception as e:
                print(f"[WARN] W&B init failed: {e}")
                self._wandb = None

    def train_step(self, batch) -> Dict:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=self.mixed_precision):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / self.grad_accum_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm = 0.0
        if ((self.global_step + 1) % self.grad_accum_steps) == 0:
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                ).item()

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            # LR scheduler update
            lr = next(self.lr_scheduler)
        else:
            lr = self.optimizer.param_groups[0]["lr"]

        self.global_step += 1
        self.metrics.global_step = self.global_step
        self.metrics.update({
            "loss": loss.item() * self.grad_accum_steps,
            "grad_norm": grad_norm,
            "lr": lr
        })

        return {
            "loss": loss.item() * self.grad_accum_steps,
            "grad_norm": grad_norm,
            "lr": lr
        }

    @torch.no_grad()
    def evaluate(self) -> Dict:
        if self.eval_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Simple token-level accuracy (cuando label != PAD)
            preds = torch.argmax(outputs.logits, dim=-1)
            mask = labels != PAD_ID
            correct = (preds[mask] == labels[mask]).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

        avg_loss = total_loss / max(len(self.eval_loader), 1)
        accuracy = total_correct / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_accuracy": accuracy}

    def maybe_log(self):
        if self.global_step % 10 == 0:
            rep = self.metrics.report()
            print(
                f"[step {rep['global_step']}] loss={rep['loss']:.4f} "
                f"ppl={rep['ppl']:.2f} acc={rep['acc']:.3f} "
                f"grad={rep['grad_norm']:.2f} lr={rep['lr']:.6f} "
                f"steps/s={rep['steps_per_sec']:.2f} tok/s={int(rep['tokens_per_sec'])} "
                f"uptime_s={rep['uptime_s']} ram_gb={rep['ram_gb']:.2f}"
            )
            if self._wandb is not None:
                try:
                    self._wandb.log(rep)
                except Exception as e:
                    print(f"[WARN] W&B log failed: {e}")

    def maybe_eval_and_save(self, output_dir: str, eval_steps: int, save_steps: int):
        if self.global_step % eval_steps == 0 and self.eval_loader is not None:
            eval_metrics = self.evaluate()
            msg = f"Eval @step {self.global_step}: " + \
                  ", ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
            print(msg)
            if self._wandb is not None:
                try:
                    self._wandb.log({"eval": eval_metrics, "step": self.global_step})
                except Exception as e:
                    print(f"[WARN] W&B eval log failed: {e}")

        if self.global_step % save_steps == 0:
            save_checkpoint(self.model, self.optimizer, self.global_step, output_dir)

    def train(self, output_dir: str, eval_steps: int, save_steps: int):
        for epoch in range(10**9):  # endless epochs until max_steps
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                self.maybe_log()
                self.maybe_eval_and_save(output_dir, eval_steps, save_steps)

                if self.global_step >= self.max_steps:
                    print("[INFO] Training complete.")
                    save_checkpoint(self.model, self.optimizer, self.global_step, output_dir)
                    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to eval text file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--tokenizer_dir", type=str, default=TOKENIZER_SAVE_DIR)
    parser.add_argument("--mixed_precision", action="store_true", default=MIXED_PRECISION)
    parser.add_argument("--use_wandb", action="store_true", default=USE_WANDB)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=GRADIENT_CHECKPOINTING)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_CONFIG.max_position_embeddings)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Tokenizer
    if os.path.isdir(args.tokenizer_dir) and os.path.isfile(os.path.join(args.tokenizer_dir, "tokenizer.json")):
        tokenizer = HarmonyTokenizer.from_pretrained(args.tokenizer_dir)
    else:
        tokenizer = setup_tokenizer(args.tokenizer_dir)

    # Model
    model = configure_model(DEFAULT_CONFIG, gradient_checkpointing=args.gradient_checkpointing)
    print(model)
    print(f"[INFO] Model parameters: {count_parameters(model):,}")

    # Data
    loaders = create_dataloaders(
        train_path=args.train_data,
        eval_path=args.eval_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        use_wandb=args.use_wandb
    )

    # Optionally load checkpoint
    start_step = load_checkpoint_if_available(trainer.model, trainer.optimizer, args.output_dir)
    if start_step > 0:
        trainer.global_step = start_step
        print(f"[INFO] Resuming from step {start_step}")

    # Train
    trainer.train(output_dir=args.output_dir, eval_steps=args.eval_steps, save_steps=args.save_steps)

    # Save final tokenizer (again)
    tokenizer.save_pretrained(args.tokenizer_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
