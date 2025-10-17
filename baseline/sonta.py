import argparse
import os
import math
import sys
from typing import Tuple, List, Dict, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, f1_score

import util
# 注意：不再使用 from data_sl import data_load

# 导入新的数据加载函数
from dan import data_load_d

from models.patchtst import PatchTST
from models.TSSequencerPlus import TSSequencer
from models.TSPerceiver import TSPerceiver
from models.XCM import XCMPlus
from models.Rocket import MiniRocketPlus

import time 

DEFAULT_SEED = 128
DEFAULT_PATCH_SIZE = 16
DEFAULT_NUM_CLASSES = 5  # 修改为 5 类（0~4）
CHECKPOINT_DIR = "checkpoint/baseline"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training Script with Hyperparameters")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seeds")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="Number of fault classes (0~4 => 5)")
    
    parser.add_argument("--warmup", type=int, default=20, help="Warmup epochs for early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience for each cold")
    parser.add_argument("--cold", type=int, default=3, help="Times for reset the patience")
    
    parser.add_argument("--hidden_size", type=int, default=64, help="Model dimension")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for train data")
    parser.add_argument("--backbone", type=str, default="patchtst", 
                        choices=["patchtst", "tssequencer", "tsp", "xcm", "rocket"],
                        help="Model backbone")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    
    parser.add_argument("--train_ratio", type=float, default=0.24, help="Ratio for train/test split (not subset!)")
    
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--init_lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--max_norm", type=float, default=5.0, help="Maximum norm for gradient clipping")
    
    return parser.parse_args()


class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, 
                 total_steps: int, init_lr: float, min_lr: float):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            lr = self.init_lr * (step + 1) / self.warmup_steps
        else:
            decay_steps = self.total_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.init_lr - self.min_lr) * cosine_decay
            
        return [lr for _ in self.base_lrs]


class Baseline:
    def __init__(self, backbone: str = 'patchtst',
                 num_classes: int = 5,
                 train_ratio: float = 0.8,
                 hidden_size: int = 128,
                 seq_len: int = 512,
                 patch_size: int = 16,
                 num_layers: int = 3,
                 batch_size: int = 256,
                 num_heads: int = 8,
                 hist_save_path: str = None) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert seq_len % patch_size == 0, "Error: seq_len must be an integer multiple of patch_size"
        
        # Supported backbone types
        self.supported_backbones = {
            'patchtst': PatchTST,
            'tssequencer': TSSequencer,
            'tsp': TSPerceiver,
            'xcm': XCMPlus,
            'rocket': MiniRocketPlus
        }
        assert backbone in self.supported_backbones, f"Unsupported backbone: {backbone}"
        self.backbone = backbone
        
        # Model configurations —— c_in 改为 10
        model_config = {
            'patchtst': {
                'c_in': 10,  # ✅ 10 个变量
                'c_out': num_classes,
                'seq_len': seq_len,
                'patch_len': patch_size,
                'd_model': hidden_size,
                'n_heads': num_heads,
                'n_layers': num_layers
            },
            'tssequencer': {
                'c_in': 10,  # ✅
                'c_out': num_classes,
                'seq_len': seq_len,
                'd_model': hidden_size,
                'depth': num_layers,
                'token_size': patch_size,
                'dropout': 0.1
            },
            'tsp': {
                'c_in': 10,  # ✅
                'c_out': num_classes,
                'seq_len': seq_len,
                'n_latents': 128,
                'd_latent': hidden_size,
                'n_layers': num_layers
            },
            'xcm': {
                'c_in': 10,  # ✅
                'c_out': num_classes,
                'seq_len': seq_len,
                'nf': hidden_size
            },
            'rocket': {
                'c_in': 10,  # ✅
                'c_out': num_classes,
                'seq_len': seq_len
            }
        }
        
        model_class = self.supported_backbones[backbone]
        self.model = model_class(**model_config[backbone]).to(self.device)
        print(f'Total trainable parameters of {backbone}: {int(util.get_parameters(self.model) // 1e3):d}K')

        self._init_data(batch_size, train_ratio)
        
        if hist_save_path:
            self._save_label_histogram(hist_save_path)

    def _init_data(self, batch_size: int, train_ratio: float) -> None:
        # 使用新的数据加载函数
        (X_train, y_train), (X_test, y_test) = data_load_d(train_ratio=train_ratio)
        
        # 转为 PyTorch 张量
        self.data_train = torch.tensor(X_train, dtype=torch.float32)
        self.label_train = torch.tensor(y_train, dtype=torch.long)
        self.data_test = torch.tensor(X_test, dtype=torch.float32)
        self.label_test = torch.tensor(y_test, dtype=torch.long)

        self._create_data_loaders(self.data_train, self.label_train, batch_size)

    def _create_data_loaders(self, 
                            data: torch.Tensor, 
                            labels: torch.Tensor, 
                            batch_size: int) -> None:
        dataset = torch.utils.data.TensorDataset(data, labels)
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        self.tra_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    def _save_label_histogram(self, save_path: str) -> None:
        train_labels = self.label_train.cpu().numpy()
        unique, counts = np.unique(train_labels, return_counts=True)

        top3_indices = np.argsort(counts)[-3:]
        colors = ['#1f77b4'] * len(unique)
        for idx in top3_indices:
            colors[idx] = 'orange'

        plt.figure(figsize=(12, 6))
        bars = plt.bar(unique, counts, color=colors, align='center', alpha=0.8, edgecolor='black', linewidth=0.3)
        plt.xlabel('Class Label')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training set label frequency histogram saved to {save_path}")
        plt.close()

    def Train(self, num_epochs: int, warmup: int, patience: int, cold: int, max_norm: float, init_lr: float, 
              min_lr: float) -> None:
        start_train = time.time()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=init_lr)
        
        scheduler = self._init_scheduler(optimizer, num_epochs, warmup, init_lr, min_lr)
        early_stop = util.EarlyStopping(
            warmup, patience, cold, init_lr, min_lr,
            path=f'baseline/checkpoint/baseline/{self.backbone}.pth'
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(criterion, optimizer, scheduler, max_norm)
            val_loss, accuracy, f1_score = self._validate(criterion)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'baseline/checkpoint/baseline/{self.backbone}_best.pth')
                print(f"New best model saved with val_loss: {val_loss:.4f}")

            self._print_epoch_info(epoch, num_epochs, avg_loss, val_loss, accuracy, f1_score)

            early_stop(val_loss, self.model, optimizer)
            if early_stop.early_stop:
                break
                
        elapsed = time.time() - start_train
        print(f"[Training] Total elapsed time: {elapsed:.2f} s")

    def _init_scheduler(self, 
                       optimizer: optim.Optimizer, 
                       num_epochs: int, 
                       warmup: int, 
                       init_lr: float, 
                       min_lr: float) -> WarmupCosineSchedule:
        total_steps = num_epochs * len(self.tra_loader)
        warmup_steps = warmup * len(self.tra_loader)
        return WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            init_lr=init_lr,
            min_lr=min_lr
        )

    def _train_epoch(self, 
                   criterion: nn.Module, 
                   optimizer: optim.Optimizer, 
                   scheduler: WarmupCosineSchedule, 
                   max_norm: float) -> float:
        self.model.train()
        running_loss = 0.0
        
        for step, (inputs, labels) in enumerate(self.tra_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step}, LR: {current_lr:.2e}")

            running_loss += loss.item()
        
        return running_loss / len(self.tra_loader)

    def _validate(self, criterion: nn.Module) -> Tuple[float, float, float]:
        self.model.eval()
        val_running_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item() * labels.size(0)
                total_samples += labels.size(0)
                all_outputs.append(outputs)
                all_labels.append(labels)
        
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        
        val_loss = val_running_loss / total_samples
        accuracy = util.accuracy(all_outputs, all_labels)
        f1_score = util.f1_score(all_outputs, all_labels)
        
        return val_loss, accuracy, f1_score

    def _print_epoch_info(self, 
                        epoch: int, 
                        num_epochs: int, 
                        avg_loss: float, 
                        val_loss: float, 
                        accuracy: float, 
                        f1_score: float) -> None:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}],'
            f' Train Loss: {avg_loss:.4f},'
            f' Val Loss: {val_loss:.4f},'
            f' Val Acc: {accuracy:.4f},'
            f' Val F1-score: {f1_score:.4f}'
        )

    def Test(self) -> None:
        self._load_best_model()
        
        test_dataset = torch.utils.data.TensorDataset(self.data_test, self.label_test)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        all_outputs, all_labels, all_preds = self._run_test(test_loader)
        
        accuracy = util.accuracy(all_outputs, all_labels)
        f1_score = util.f1_score(all_outputs, all_labels)
        
        print("\nClassification Report:")
        print(classification_report(all_labels.cpu(), all_preds.cpu()))
        print(f'\nTest Acc: {accuracy:.4f}, F1-score: {f1_score:.4f}')

    def _load_best_model(self) -> None:
        checkpoint_path = f'baseline/checkpoint/baseline/{self.backbone}_best.pth'
        if os.path.exists(checkpoint_path):
            print(f"Loading best weights from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print(f"Best checkpoint {checkpoint_path} not found, running test without loading weights.")
        self.model.eval()

    def _run_test(self, test_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_outputs = []
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                all_outputs.append(outputs)
                all_labels.append(labels)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds)

        return torch.cat(all_outputs), torch.cat(all_labels), torch.cat(all_preds)


def main(args):
    util.seed_torch(args.seed)

    os.makedirs("baseline/checkpoint/baseline", exist_ok=True)
    hist_save_path = f"baseline/checkpoint/baseline/{args.backbone}_train_label_histogram.png"

    baseline = Baseline(
        backbone=args.backbone,
        num_classes=args.num_classes,
        train_ratio=args.train_ratio,
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        hist_save_path=hist_save_path
    )

    baseline.Train(
        num_epochs=args.epochs,
        warmup=args.warmup,
        patience=args.patience,
        cold=args.cold,
        max_norm=args.max_norm,
        init_lr=args.init_lr,
        min_lr=args.min_lr
    )

    baseline.Test()


if __name__ == '__main__':
    args = get_args()
    main(args)