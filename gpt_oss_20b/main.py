"""
Main training script for GPT-OSS
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import argparse
from tqdm import tqdm
from typing import Optional, Dict, List
import json
import time
import psutil
from collections import deque

from config.model_config import GPTOSSConfig
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
MIXED_PRECISION = True
GRADIENT_CHECKPOINTING = True

DEFAULT_CONFIG = GPTOSSConfig()


class MetricsTracker:
    """Tracks training metrics with rolling averages"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.perplexities = deque(maxlen=window_size)
        self.accuracies = deque(maxlen=window_size)
        self.grad_norms = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.batch_size = 1
        self.seq_len = 2048
        
    def update(self, metrics: Dict):
        """Update metrics with new values"""
        if 'loss' in metrics:
            self.losses.append(metrics['loss'])
            self.perplexities.append(torch.exp(torch.tensor(metrics['loss'])).item())
        
        if 'accuracy' in metrics:
            self.accuracies.append(metrics['accuracy'])
            
        if 'grad_norm' in metrics:
            self.grad_norms.append(metrics['grad_norm'])
            
        if 'lr' in metrics:
            self.learning_rates.append(metrics['lr'])
            
        # Time per step
        current_time = time.time()
        step_time = current_time - self.step_start_time
        self.step_times.append(step_time)
        self.step_start_time = current_time
        
    def get_averages(self) -> Dict:
        """Get rolling averages"""
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        tokens_per_sec = (self.batch_size * self.seq_len) / avg_step_time if avg_step_time > 0 else 0
        
        return {
            'avg_loss': sum(self.losses) / len(self.losses) if self.losses else 0,
            'avg_perplexity': sum(self.perplexities) / len(self.perplexities) if self.perplexities else 0,
            'avg_accuracy': sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0,
            'avg_grad_norm': sum(self.grad_norms) / len(self.grad_norms) if self.grad_norms else 0,
            'avg_step_time': avg_step_time,
            'tokens_per_sec': tokens_per_sec,
            'elapsed_time': time.time() - self.start_time
        }


def compute_model_metrics(outputs, labels, vocab_size: int) -> Dict:
    """Compute additional model metrics"""
    metrics = {}
    
    # Perplexity
    if hasattr(outputs, 'loss') and outputs.loss is not None:
        metrics['perplexity'] = torch.exp(outputs.loss).item()
    
    # Next token accuracy
    if hasattr(outputs, 'logits'):
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()
        
        # Top-1 accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        mask = (targets != -100).float()
        accuracy = (correct * mask).sum() / mask.sum() if mask.sum() > 0 else 0
        metrics['accuracy'] = accuracy.item()
        
        # Top-5 accuracy
        _, top5_preds = torch.topk(logits, min(5, vocab_size), dim=-1)
        top5_correct = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float()
        top5_accuracy = (top5_correct * mask).sum() / mask.sum() if mask.sum() > 0 else 0
        metrics['top5_accuracy'] = top5_accuracy.item()
        
        # Prediction entropy
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        metrics['prediction_entropy'] = entropy.mean().item()
    
    # MoE metrics if available
    if hasattr(outputs, 'router_losses') and outputs.router_losses is not None:
        lb_loss, z_loss = outputs.router_losses
        metrics['moe_load_balance_loss'] = lb_loss.item()
        metrics['moe_z_loss'] = z_loss.item()
    
    return metrics


def get_system_metrics() -> Dict:
    """Get system resource metrics"""
    metrics = {}
    
    # CPU and memory
    metrics['cpu_percent'] = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    metrics['memory_percent'] = memory.percent
    metrics['memory_used_gb'] = memory.used / (1024**3)
    
    # GPU metrics if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
        metrics['gpu_memory_gb'] = gpu_memory
        metrics['gpu_memory_cached_gb'] = gpu_memory_cached
        
    return metrics


def format_display_metrics(step, metrics, averages, max_steps):
    """Format metrics for console display"""
    progress = step / max_steps * 100
    
    # Main metrics line
    main_line = f"Step {step:>6}/{max_steps} ({progress:5.1f}%) | "
    main_line += f"Loss: {metrics.get('loss', 0):.4f} | "
    main_line += f"PPL: {metrics.get('perplexity', 0):.2f} | "
    main_line += f"Acc: {metrics.get('accuracy', 0)*100:.1f}% | "
    main_line += f"LR: {metrics.get('lr', 0):.2e} | "
    main_line += f"Tok/s: {averages.get('tokens_per_sec', 0):.0f}"
    
    if 'gpu_memory_gb' in metrics:
        main_line += f" | GPU: {metrics['gpu_memory_gb']:.1f}GB"
    
    # ETA calculation
    elapsed = averages.get('elapsed_time', 0)
    eta = (elapsed / step * (max_steps - step)) if step > 0 else 0
    main_line += f" | ETA: {eta/60:.0f}m"
    
    return main_line


def get_or_create_tokenizer(save_dir: str) -> HarmonyTokenizer:
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, "harmony_tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = HarmonyTokenizer.from_pretrained(save_dir)
    else:
        tokenizer = HarmonyTokenizer()
        tokenizer.save_pretrained(save_dir)
    return tokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: HarmonyTokenizer,
        max_length: int = 2048,
        stride: int = 1024
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        if not os.path.exists(data_path) or not os.path.isfile(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            self.texts = f.read().split("\n\n")

        self.examples = self._create_examples()

    def _create_examples(self):
        examples = []
        for text in self.texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors=None)
            for i in range(0, max(0, len(tokens) - self.max_length + 1), self.stride):
                examples.append(tokens[i:i + self.max_length])
            if 0 < len(tokens) < self.max_length:
                pad_id = self.tokenizer.pad_token_id
                padded = tokens + [pad_id] * (self.max_length - len(tokens))
                examples.append(padded[:self.max_length])
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        labels = ids.clone()
        return {"input_ids": ids, "labels": labels}


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    attention_mask = (input_ids != 200002).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class Trainer:
    def __init__(
        self,
        model: GPTOSSForCausalLM,
        tokenizer: HarmonyTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        batch_size: int = 1,
        learning_rate: float = 3e-4,
        warmup_steps: int = 500,
        max_steps: int = 2000,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: int = 1000,
        output_dir: str = "./output",
        use_wandb: bool = False,
        mixed_precision: bool = True,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_accum_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self._wandb = None

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        self.metrics_tracker.batch_size = batch_size
        self.metrics_tracker.seq_len = 2048

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        else:
            self.eval_loader = None

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

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
                        "model_params": count_parameters(model)
                    }
                )
            except:
                print("Failed to initialize wandb. Continuing without logging.")
                self.use_wandb = False

        os.makedirs(output_dir, exist_ok=True)

        self.global_step = 0
        self.best_eval_loss = float('inf')

    def train(self):
        self.model.train()
        
        print("Training started")
        print("=" * 80)

        for _, batch in enumerate(self._infinite_loader(self.train_loader)):
            if self.global_step >= self.max_steps:
                break

            # Learning rate schedule
            lr = warmup_cosine_schedule(
                self.global_step,
                self.warmup_steps,
                self.max_steps,
                self.learning_rate
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_router_losses=True,
                    use_cache=False
                )
                loss = outputs.loss / self.grad_accum_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss = loss.item() * self.grad_accum_steps

            # Optimization step
            if (self.global_step + 1) % self.grad_accum_steps == 0:
                grad_norm = compute_grad_norm(self.model)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = 0.0

            # Compute metrics
            base_metrics = {
                'loss': accumulated_loss,
                'lr': lr,
                'grad_norm': grad_norm
            }
            
            # Model metrics
            model_metrics = compute_model_metrics(outputs, labels, self.tokenizer.vocab_size)
            
            # System metrics (only every few steps to avoid overhead)
            system_metrics = {}
            if self.global_step % 10 == 0:
                system_metrics = get_system_metrics()
            
            # Combine metrics
            all_metrics = {**base_metrics, **model_metrics, **system_metrics}
            
            # Update tracker
            self.metrics_tracker.update(all_metrics)
            averages = self.metrics_tracker.get_averages()

            # Display metrics
            if self.global_step % 5 == 0:  # Display every 5 steps
                display_line = format_display_metrics(
                    self.global_step, all_metrics, averages, self.max_steps
                )
                print(display_line)
                
                # Detailed metrics every 50 steps
                if self.global_step % 50 == 0 and self.global_step > 0:
                    print(f"    Details: Top5_Acc: {all_metrics.get('top5_accuracy', 0)*100:.1f}% | "
                          f"Entropy: {all_metrics.get('prediction_entropy', 0):.2f} | "
                          f"StepTime: {averages.get('avg_step_time', 0)*1000:.0f}ms | "
                          f"CPU: {system_metrics.get('cpu_percent', 0):.0f}% | "
                          f"RAM: {system_metrics.get('memory_percent', 0):.0f}%")

            # Wandb logging
            if self.use_wandb and self._wandb is not None:
                try:
                    log_dict = {
                        "train/loss": all_metrics.get('loss', 0),
                        "train/perplexity": all_metrics.get('perplexity', 0),
                        "train/accuracy": all_metrics.get('accuracy', 0),
                        "train/top5_accuracy": all_metrics.get('top5_accuracy', 0),
                        "train/prediction_entropy": all_metrics.get('prediction_entropy', 0),
                        "optim/learning_rate": all_metrics.get('lr', 0),
                        "optim/grad_norm": all_metrics.get('grad_norm', 0),
                        "speed/tokens_per_sec": averages.get('tokens_per_sec', 0),
                        "speed/step_time_ms": averages.get('avg_step_time', 0) * 1000,
                        "system/cpu_percent": system_metrics.get('cpu_percent', 0),
                        "system/memory_percent": system_metrics.get('memory_percent', 0),
                        "system/gpu_memory_gb": system_metrics.get('gpu_memory_gb', 0),
                        "step": self.global_step
                    }
                    if 'moe_load_balance_loss' in all_metrics:
                        log_dict["moe/load_balance_loss"] = all_metrics['moe_load_balance_loss']
                        log_dict["moe/z_loss"] = all_metrics['moe_z_loss']
                    
                    self._wandb.log(log_dict)
                except:
                    pass

            # Evaluation
            if self.eval_loader and (self.global_step % self.eval_steps == 0) and self.global_step > 0:
                eval_loss = self.evaluate()
                print(f"    Evaluation: Loss: {eval_loss:.4f} | PPL: {torch.exp(torch.tensor(eval_loss)):.2f}")
                if self.use_wandb and self._wandb is not None:
                    try:
                        self._wandb.log({"eval/loss": eval_loss, "eval/perplexity": torch.exp(torch.tensor(eval_loss)).item()}, step=self.global_step)
                    except:
                        pass
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best")
                    print(f"    New best model saved (eval_loss: {eval_loss:.4f})")

            # Save checkpoints
            if self.global_step % self.save_steps == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step}")
                print(f"    Checkpoint saved: step_{self.global_step}")

            self.global_step += 1

        print("=" * 80)
        print("Training completed")
        self.save_checkpoint("final")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_router_losses=False,
                    use_cache=False
                )
                loss = outputs.loss
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, name: str):
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)

        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

        state = {
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "max_grad_norm": self.max_grad_norm
        }
        with open(os.path.join(save_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _infinite_loader(self, loader):
        while True:
            for batch in loader:
                yield batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--use_wandb", action="store_true" if USE_WANDB else "store_false", default=USE_WANDB)
    parser.add_argument("--mixed_precision", action="store_true" if MIXED_PRECISION else "store_false", default=MIXED_PRECISION)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print("Device:", device)
    print("Mixed precision:", args.mixed_precision and torch.cuda.is_available())
    print("W&B:", args.use_wandb)
    print("Data path:", args.data_path)
    print("=" * 60)

    tokenizer = get_or_create_tokenizer(TOKENIZER_SAVE_DIR)

    dataset = TextDataset(args.data_path, tokenizer, max_length=args.seq_len, stride=args.stride)
    n = len(dataset)
    split = int(0.98 * n) if n > 1 else n
    if split == n:
        train_dataset, eval_dataset = dataset, None
    else:
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [split, n - split])

    model = GPTOSSForCausalLM(
        vocab_size=DEFAULT_CONFIG.vocab_size,
        hidden_size=DEFAULT_CONFIG.hidden_size,
        num_layers=DEFAULT_CONFIG.num_layers,
        num_attention_heads=DEFAULT_CONFIG.num_attention_heads,
        num_key_value_heads=DEFAULT_CONFIG.num_key_value_heads,
        intermediate_size=DEFAULT_CONFIG.intermediate_size,
        num_experts=DEFAULT_CONFIG.num_experts,
        num_experts_per_token=DEFAULT_CONFIG.num_experts_per_token,
        max_position_embeddings=DEFAULT_CONFIG.max_position_embeddings,
        rope_theta=DEFAULT_CONFIG.rope_theta,
        use_attention_sinks=DEFAULT_CONFIG.use_attention_sinks,
        attention_sink_size=DEFAULT_CONFIG.attention_sink_size,
        use_sparse_attention=DEFAULT_CONFIG.use_sparse_attention,
        sparse_window_size=DEFAULT_CONFIG.sparse_window_size,
        rms_norm_eps=DEFAULT_CONFIG.rms_norm_eps,
        aux_loss_coef=DEFAULT_CONFIG.aux_loss_coef,
        router_jitter_noise=DEFAULT_CONFIG.router_jitter_noise,
        pad_token_id=DEFAULT_CONFIG.pad_token_id,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        tie_word_embeddings=True,
        device=device,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    param_info = count_parameters(model)
    print("Trainable parameters:", f"{param_info['trainable']:,} ({param_info['trainable_billions']:.2f}B)")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        mixed_precision=args.mixed_precision
    )

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60 + "\n")

    try:
        t0 = time.time()
        trainer.train()
        t1 = time.time()
        print(f"\nTraining completed in {(t1 - t0)/60:.2f} minutes")
        trainer.save_checkpoint("final")
        print(f"Models saved to: {args.output_dir}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        trainer.save_checkpoint("interrupted")
        print(f"Checkpoint saved to: {args.output_dir}/interrupted")
    except Exception as e:
        print(f"\nError during training: {e}")
        trainer.save_checkpoint("error")
        print(f"Emergency checkpoint saved to: {args.output_dir}/error")
        raise


if __name__ == "__main__":
    main()