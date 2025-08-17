"""
Main training script for GPT-OSS 20B
Unsupervised foundational model training
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any
import json
import time

from config.model_config import GPTOSSConfig
from src.model.gpt_oss import GPTOSSForCausalLM
from src.tokenizer.harmony_tokenizer import HarmonyTokenizer
from src.utils.tensor_utils import get_device, count_parameters
from src.utils.math_utils import warmup_cosine_schedule, compute_grad_norm


# ==========================================
# CONFIGURACIÓN PREDEFINIDA - MODIFICA AQUÍ
# ==========================================

# Rutas de datos
DATA_PATH = "./data"  # Carpeta con archivos .txt de entrenamiento
EVAL_DATA_PATH = None  # Opcional: "./data/eval" para evaluación

# Configuración del modelo
MODEL_VARIANT = "20B"  # "20B" o "120B"
MAX_LENGTH = 512  # Longitud máxima de secuencia

# Configuración de entrenamiento
BATCH_SIZE = 1  # Tamaño de batch
LEARNING_RATE = 3e-4  # Tasa de aprendizaje
WARMUP_STEPS = 100  # Pasos de calentamiento
MAX_STEPS = 1000  # Número máximo de pasos
GRADIENT_ACCUMULATION_STEPS = 16  # Acumulación de gradientes
EVAL_STEPS = 100  # Frecuencia de evaluación
SAVE_STEPS = 500  # Frecuencia de guardado
OUTPUT_DIR = "./output"  # Directorio de salida

# Otras configuraciones
USE_WANDB = False  # Activar Weights & Biases
MIXED_PRECISION = True  # Usar precisión mixta
GRADIENT_CHECKPOINTING = True  # Usar gradient checkpointing

# ==========================================


class TextDataset(Dataset):
    """
    Simple text dataset for unsupervised training
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: HarmonyTokenizer,
        max_length: int = 2048,
        stride: int = 1024
    ):
        """
        Args:
            data_path: Path to text file or directory
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for sliding window
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Load text data
        self.texts = self._load_texts(data_path)
        
        # Tokenize all texts
        self.examples = self._create_examples()
        
        if len(self.examples) == 0:
            raise ValueError(f"No se encontraron ejemplos de entrenamiento en {data_path}")
        
    def _load_texts(self, data_path: str) -> list:
        """Load text files"""
        texts = []
        
        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        elif os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(data_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            texts.append(content)
                            print(f"Cargado: {filename} ({len(content)} caracteres)")
                        
        if not texts:
            raise ValueError(f"No se encontraron archivos .txt en {data_path}")
            
        return texts
    
    def _create_examples(self) -> list:
        """Create training examples with sliding window"""
        examples = []
        
        for text in self.texts:
            # Tokenize entire text
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                return_tensors=None
            )
            
            # Create sliding window examples
            for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                example = tokens[i:i + self.max_length]
                examples.append(example)
                
            # Si el texto es más corto que max_length, agregarlo con padding
            if len(tokens) < self.max_length and len(tokens) > 10:
                padded = tokens + [self.tokenizer.special_tokens["<|pad|>"]] * (self.max_length - len(tokens))
                examples.append(padded)
                
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, labels are same as inputs
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }


class Trainer:
    """
    Trainer class for GPT-OSS
    """
    
    def __init__(
        self,
        model: GPTOSSForCausalLM,
        tokenizer: HarmonyTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        eval_steps: int = 500,
        save_steps: int = 1000,
        output_dir: str = "./output",
        use_wandb: bool = False,
        mixed_precision: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: GPT-OSS model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            max_steps: Maximum training steps
            gradient_accumulation_steps: Gradient accumulation
            max_grad_norm: Maximum gradient norm
            eval_steps: Evaluation frequency
            save_steps: Save frequency
            output_dir: Output directory
            use_wandb: Whether to use Weights & Biases
            mixed_precision: Whether to use mixed precision
            device: Device for training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.mixed_precision = mixed_precision
        
        # Device
        self.device = device or get_device()
        self.model.to(self.device)
        
        print(f"Usando dispositivo: {self.device}")
        
        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # 0 para evitar problemas en Windows
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
            
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None
        
        # Initialize wandb
        if use_wandb:
            try:
                wandb.init(
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
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
    def train(self):
        """Main training loop"""
        self.model.train()
        
        progress_bar = tqdm(total=self.max_steps, desc="Training")
        accumulated_loss = 0.0
        
        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                # Add MoE losses if present
                if outputs.router_losses is not None:
                    lb_loss, z_loss = outputs.router_losses
                    loss = loss + lb_loss + z_loss
                    
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                accumulated_loss += loss.item()
                
                # Backward pass
                if self.mixed_precision and self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Update weights
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.mixed_precision and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.mixed_precision and self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    # Update learning rate
                    lr = warmup_cosine_schedule(
                        self.global_step,
                        self.warmup_steps,
                        self.max_steps,
                        min_lr=self.learning_rate * 0.1,
                        max_lr=self.learning_rate
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                        
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Print loss every 10 steps
                    if self.global_step % 10 == 0:
                        print(f"Step {self.global_step}: Loss = {accumulated_loss:.4f}, LR = {lr:.6f}")
                    
                    # Logging
                    if self.use_wandb:
                        try:
                            wandb.log({
                                "loss": accumulated_loss,
                                "learning_rate": lr,
                                "grad_norm": grad_norm.item()
                            }, step=self.global_step)
                        except:
                            pass
                        
                    accumulated_loss = 0.0
                    
                # Evaluation
                if self.eval_loader and self.global_step % self.eval_steps == 0:
                    eval_loss = self.evaluate()
                    print(f"Evaluation at step {self.global_step}: Loss = {eval_loss:.4f}")
                    
                    if self.use_wandb:
                        try:
                            wandb.log({"eval_loss": eval_loss}, step=self.global_step)
                        except:
                            pass
                        
                    # Save best model
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint("best")
                        
                    self.model.train()
                    
                # Save checkpoint
                if self.global_step % self.save_steps == 0 and self.global_step > 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
                    
                # Update progress
                progress_bar.update(1)
                self.global_step += 1
                
                if self.global_step >= self.max_steps:
                    break
                    
        progress_bar.close()
        
        # Save final model
        self.save_checkpoint("final")
        
    def evaluate(self) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                    
                total_loss += outputs.loss.item()
                total_steps += 1
                
        avg_loss = total_loss / max(1, total_steps)
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "model.pt")
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))
        
        print(f"Checkpoint saved to {checkpoint_dir}")


def main():
    """
    Función principal
    """
    
    print("=" * 60)
    print("GPT-OSS Training Script")
    print("=" * 60)
    
    # Verificar que existe la carpeta de datos
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: No se encontró la carpeta de datos: {DATA_PATH}")
        print(f"Por favor crea la carpeta '{DATA_PATH}' y agrega archivos .txt")
        return
    
    # Verificar que hay archivos .txt
    txt_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    if not txt_files:
        print(f"\nERROR: No se encontraron archivos .txt en {DATA_PATH}")
        print(f"Por favor agrega archivos de texto con extensión .txt")
        return
    
    print(f"\nEncontrados {len(txt_files)} archivos .txt en {DATA_PATH}")
    
    # Mostrar configuración
    print("\nConfiguración:")
    print(f"  - Modelo: {MODEL_VARIANT}")
    print(f"  - Datos: {DATA_PATH}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max steps: {MAX_STEPS}")
    print(f"  - Max length: {MAX_LENGTH}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Mixed precision: {MIXED_PRECISION}")
    print(f"  - Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print("=" * 60)
    
    # Initialize configuration
    config = GPTOSSConfig(model_variant=MODEL_VARIANT)
    
    # Initialize tokenizer
    print("\nInicializando tokenizer...")
    tokenizer = HarmonyTokenizer()
    
    # Initialize model
    print("Inicializando modelo...")
    model = GPTOSSForCausalLM(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.intermediate_size,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_token,
        gradient_checkpointing=GRADIENT_CHECKPOINTING
    )
    
    # Print model info
    params = count_parameters(model)
    print(f"\nModelo inicializado:")
    print(f"  - Parámetros totales: {params['total_billions']:.2f}B")
    print(f"  - Parámetros activos por token: {config.active_params_per_token / 1e9:.2f}B")
    
    # Create datasets
    print(f"\nCargando dataset desde {DATA_PATH}...")
    try:
        train_dataset = TextDataset(
            DATA_PATH,
            tokenizer,
            max_length=MAX_LENGTH
        )
        print(f"Dataset de entrenamiento: {len(train_dataset)} ejemplos")
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return
    
    eval_dataset = None
    if EVAL_DATA_PATH and os.path.exists(EVAL_DATA_PATH):
        print(f"Cargando dataset de evaluación desde {EVAL_DATA_PATH}...")
        try:
            eval_dataset = TextDataset(
                EVAL_DATA_PATH,
                tokenizer,
                max_length=MAX_LENGTH
            )
            print(f"Dataset de evaluación: {len(eval_dataset)} ejemplos")
        except Exception as e:
            print(f"Advertencia: No se pudo cargar dataset de evaluación: {e}")
        
    # Initialize trainer
    print("\nInicializando entrenador...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        output_dir=OUTPUT_DIR,
        use_wandb=USE_WANDB,
        mixed_precision=MIXED_PRECISION
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Iniciando entrenamiento...")
    print("=" * 60 + "\n")
    
    try:
        trainer.train()
        print("\n" + "=" * 60)
        print("¡Entrenamiento completado exitosamente!")
        print(f"Modelos guardados en: {OUTPUT_DIR}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
        trainer.save_checkpoint("interrupted")
        print(f"Checkpoint guardado en: {OUTPUT_DIR}/interrupted")
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        trainer.save_checkpoint("error")
        print(f"Checkpoint de emergencia guardado en: {OUTPUT_DIR}/error")
        raise


if __name__ == "__main__":
    main()