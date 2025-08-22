"""
Main training script for GPT-OSS
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import argparse
from tqdm import tqdm
from typing import Optional
import json
import time

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
            raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

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
                print("No se pudo inicializar wandb. Continuando sin logging.")
                self.use_wandb = False

        os.makedirs(output_dir, exist_ok=True)

        self.global_step = 0
        self.best_eval_loss = float('inf')

    def train(self):
        self.model.train()
        progress = tqdm(total=self.max_steps, desc="Training")

        for _, batch in enumerate(self._infinite_loader(self.train_loader)):
            if self.global_step >= self.max_steps:
                break

            lr = warmup_cosine_schedule(
                self.global_step,
                self.warmup_steps,
                self.max_steps,
                self.learning_rate
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

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

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss = loss.item() * self.grad_accum_steps

            if (self.global_step + 1) % self.grad_accum_steps == 0:
                grad_norm = compute_grad_norm(self.model.parameters())
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.use_wandb and self._wandb is not None:
                try:
                    self._wandb.log({
                        "loss": accumulated_loss,
                        "learning_rate": lr,
                        "step": self.global_step
                    })
                except:
                    pass

            progress.update(1)
            progress.set_postfix({"loss": f"{accumulated_loss:.4f}", "lr": f"{lr:.2e}"})

            if self.eval_loader and (self.global_step % self.eval_steps == 0) and self.global_step > 0:
                eval_loss = self.evaluate()
                if self.use_wandb and self._wandb is not None:
                    try:
                        self._wandb.log({"eval_loss": eval_loss}, step=self.global_step)
                    except:
                        pass
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best")

            if self.global_step % self.save_steps == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step}")

            self.global_step += 1

        progress.close()
        self.save_checkpoint("last")

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
    print("Dispositivo:", device)
    print("Precision mixta:", args.mixed_precision and torch.cuda.is_available())
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
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    print("Parámetros entrenables:", count_parameters(model))

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
    print("Iniciando entrenamiento")
    print("=" * 60 + "\n")

    try:
        t0 = time.time()
        trainer.train()
        t1 = time.time()
        print("\nEntrenamiento finalizado.")
        print(f"Tiempo total: {(t1 - t0)/60:.2f} min")
        trainer.save_checkpoint("final")
        print(f"Modelos guardados en: {args.output_dir}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
        trainer.save_checkpoint("interrupted")
        print(f"Checkpoint guardado en: {args.output_dir}/interrupted")
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        trainer.save_checkpoint("error")
        print(f"Checkpoint de emergencia guardado en: {args.output_dir}/error")
        raise


if __name__ == "__main__":
    main()
