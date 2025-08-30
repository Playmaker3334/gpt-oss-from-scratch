import os
import math
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
NUM_EPOCHS = 10
GRAD_ACCUM_STEPS = 8
MAX_GRAD_NORM = 1.0
LOG_EVERY = 100
SAVE_EVERY_EPOCHS = 1
GENERATION_PROMPT = "The meaning of life is"

USE_WANDB = False
MIXED_PRECISION = False
GRADIENT_CHECKPOINTING = True
USE_EPOCHS = False

DEFAULT_CONFIG = GPTOSSConfigMini()
PAD_ID = 100277  


class MetricsTracker:
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
        self.current_epoch = 0

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
        if 'epoch' in metrics:
            self.current_epoch = metrics['epoch']

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
            "epoch": self.current_epoch,
            "uptime_s": int(elapsed),
            "ram_gb": psutil.virtual_memory().used / (1024**3),
        }


class TextFileDataset(Dataset):
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


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, epoch: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    state = {
        "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch
    }
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint_if_available(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], output_dir: str):
    if not os.path.isdir(output_dir):
        return 0, 0
    files = [f for f in os.listdir(output_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not files:
        return 0, 0
    files.sort(key=lambda x: int(x.split("_")[3].split(".")[0]))
    latest = files[-1]
    path = os.path.join(output_dir, latest)
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model"], strict=False)
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    return int(data.get("step", 0)), int(data.get("epoch", 0))


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
    num_workers = min(4, os.cpu_count() or 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    if eval_path and os.path.isfile(eval_path):
        eval_ds = TextFileDataset(eval_path, tokenizer=tokenizer, seq_len=seq_len)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False
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
        gradient_checkpointing=gradient_checkpointing,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=torch.float32
    )
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
                 num_epochs: int,
                 grad_accum_steps: int,
                 max_grad_norm: float,
                 use_epochs: bool = False,
                 log_every: int = 100,
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
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_epochs = use_epochs
        self.log_every = log_every
        self.mixed_precision = mixed_precision
        self.use_wandb = use_wandb

        self.is_multi_gpu = torch.cuda.device_count() > 1 and isinstance(self.model, torch.nn.Module)
        if self.is_multi_gpu:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        self.global_step = 0
        self.current_epoch = 0
        self.metrics = MetricsTracker(window_size=100)
        self.metrics.batch_size = batch_size
        self.metrics.seq_len = DEFAULT_CONFIG.max_position_embeddings

        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision and torch.cuda.is_available() else None

        model_params = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        self.optimizer = AdamW(
            model_params, 
            lr=learning_rate, 
            betas=(0.9, 0.98), 
            weight_decay=0.01, 
            eps=1e-6
        )

        if use_wandb:
            try:
                import wandb as _wandb
                self._wandb = _wandb
                self._wandb.init(project="gpt-oss-training")
            except Exception as e:
                self._wandb = None
        else:
            self._wandb = None

        self.opt_step = 0
        micro_steps_per_epoch = len(self.train_loader)
        updates_per_epoch = math.ceil(micro_steps_per_epoch / self.grad_accum_steps)
        if self.use_epochs:
            self.total_opt_steps = self.num_epochs * updates_per_epoch
        else:
            self.total_opt_steps = math.ceil(self.max_steps / self.grad_accum_steps)
        self.warmup_updates = max(1, math.ceil(self.warmup_steps / self.grad_accum_steps))

    def train_step(self, batch) -> Dict:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=not self.is_multi_gpu
            )
            
            if self.is_multi_gpu:
                loss = outputs[0].mean() / self.grad_accum_steps
            else:
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

            self.opt_step += 1
            lr = warmup_cosine_schedule(
                self.opt_step,
                self.warmup_updates,
                self.total_opt_steps,
                0.0, self.learning_rate
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = self.optimizer.param_groups[0]["lr"]

        self.global_step += 1
        self.metrics.global_step = self.global_step
        self.metrics.update({
            "loss": loss.item() * self.grad_accum_steps,
            "grad_norm": grad_norm,
            "lr": lr,
            "epoch": self.current_epoch
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
                labels=labels,
                return_dict=not self.is_multi_gpu
            )
            
            if self.is_multi_gpu:
                loss = outputs[0].mean()
                logits = outputs[1]
            else:
                loss = outputs.loss
                logits = outputs.logits
                
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != PAD_ID
            correct = (preds[mask] == labels[mask]).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

        avg_loss = total_loss / max(len(self.eval_loader), 1)
        accuracy = total_correct / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_accuracy": accuracy}

    @torch.no_grad()
    def generate_sample(self, prompt: str, max_length: int = 50) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated = self.model.module.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        ) if self.is_multi_gpu else self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        
        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    def train_epochs(self, output_dir: str, save_every_epochs: int, generation_prompt: str):
        print(f"\nStarting training for {self.num_epochs} epochs\n")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            print(f"===== Epoch {epoch + 1}/{self.num_epochs} =====")
            
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                epoch_steps += 1
                
                if self.global_step % self.log_every == 0 and self.global_step > 0:
                    rep = self.metrics.report()
                    print(f"[Epoch {epoch+1} Step {self.global_step}] "
                          f"loss={rep['loss']:.4f} ppl={rep['ppl']:.2f} "
                          f"lr={rep['lr']:.6f} tok/s={int(rep['tokens_per_sec'])}")
            
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"\nEpoch {epoch + 1} completed. Avg loss: {avg_epoch_loss:.4f}")
            
            if self.eval_loader is not None:
                eval_metrics = self.evaluate()
                print(f"Evaluation: loss={eval_metrics['eval_loss']:.4f} "
                      f"accuracy={eval_metrics['eval_accuracy']:.4f}")
            
            generated_text = self.generate_sample(generation_prompt)
            print(f"\nGeneration sample: {generated_text}\n")
            
            if (epoch + 1) % save_every_epochs == 0:
                ckpt_path = save_checkpoint(
                    self.model, self.optimizer, 
                    self.global_step, epoch + 1, output_dir
                )
                print(f"Checkpoint saved: {ckpt_path}")
            
            print("-" * 50 + "\n")
        
        print("Training completed!")
        final_ckpt = save_checkpoint(
            self.model, self.optimizer, 
            self.global_step, self.num_epochs, output_dir
        )
        print(f"Final checkpoint saved: {final_ckpt}")

    def train_steps(self, output_dir: str, save_steps: int):
        print(f"\nStarting training for {self.max_steps} steps\n")
        
        epoch_iter = 0
        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                
                if self.global_step % self.log_every == 0 and self.global_step > 0:
                    rep = self.metrics.report()
                    print(f"[Step {self.global_step}/{self.max_steps}] "
                          f"loss={rep['loss']:.4f} ppl={rep['ppl']:.2f} "
                          f"lr={rep['lr']:.6f} tok/s={int(rep['tokens_per_sec'])}")
                
                if self.global_step % save_steps == 0 and self.global_step > 0:
                    ckpt_path = save_checkpoint(
                        self.model, self.optimizer, 
                        self.global_step, epoch_iter, output_dir
                    )
                    print(f"Checkpoint saved: {ckpt_path}")
                
                if self.global_step >= self.max_steps:
                    break
            
            epoch_iter += 1
        
        print("\nTraining completed!")
        final_ckpt = save_checkpoint(
            self.model, self.optimizer, 
            self.global_step, epoch_iter, output_dir
        )
        print(f"Final checkpoint saved: {final_ckpt}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to eval text file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--use_epochs", action="store_true", default=USE_EPOCHS,
                       help="Train by epochs instead of steps")
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--log_every", type=int, default=LOG_EVERY)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_every_epochs", type=int, default=SAVE_EVERY_EPOCHS)
    parser.add_argument("--generation_prompt", type=str, default=GENERATION_PROMPT,
                       help="Prompt for generation at the end of each epoch")
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
    print(f"Device: {device}")

    if os.path.isdir(args.tokenizer_dir) and os.path.isfile(os.path.join(args.tokenizer_dir, "tokenizer.json")):
        tokenizer = HarmonyTokenizer.from_pretrained(args.tokenizer_dir)
    else:
        tokenizer = setup_tokenizer(args.tokenizer_dir)

    model = configure_model(DEFAULT_CONFIG, gradient_checkpointing=args.gradient_checkpointing)
    print(f"Model parameters: {count_parameters(model)['trainable']:,}")

    loaders = create_dataloaders(
        train_path=args.train_data,
        eval_path=args.eval_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]

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
        num_epochs=args.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        use_epochs=args.use_epochs,
        log_every=args.log_every,
        mixed_precision=args.mixed_precision,
        use_wandb=args.use_wandb
    )

    start_step, start_epoch = load_checkpoint_if_available(trainer.model, trainer.optimizer, args.output_dir)
    if start_step > 0:
        trainer.global_step = start_step
        trainer.current_epoch = start_epoch
        trainer.opt_step = start_step // trainer.grad_accum_steps
        print(f"Resuming from epoch {start_epoch}, step {start_step}")

    if args.use_epochs:
        trainer.train_epochs(
            output_dir=args.output_dir,
            save_every_epochs=args.save_every_epochs,
            generation_prompt=args.generation_prompt
        )
    else:
        trainer.train_steps(
            output_dir=args.output_dir,
            save_steps=args.save_steps
        )

    tokenizer.save_pretrained(args.tokenizer_dir)
    print("Done!")


if __name__ == "__main__":
    main()