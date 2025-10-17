import os
import random

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PATHOGENS'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def Spectral_Radius(eig):
    return ["{:.3f}".format(to_numpy(torch.max(torch.abs(torch.real(e)))).item()) for e in eig] if isinstance(eig, list) \
        else ["{:.3f}".format(to_numpy(max(torch.abs(torch.real(eig)))))]


def get_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def Cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, np.conjugate(vec2))
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def accuracy(preds, labels):
    _, preds = torch.max(preds, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct / labels.size(0)


def f1_score(preds, labels, average='macro'):
    num_classes = preds.size(1)
    preds = torch.argmax(preds, dim=1)

    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    for i in range(num_classes):
        TP[i] = torch.sum((preds == i) & (labels == i)).item()
        FP[i] = torch.sum((preds == i) & (labels != i)).item()
        FN[i] = torch.sum((preds != i) & (labels == i)).item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if average == 'macro':
        return f1.mean().item()
    elif average == 'micro':
        TP_total = TP.sum()
        FP_total = FP.sum()
        FN_total = FN.sum()
        precision_micro = TP_total / (TP_total + FP_total + 1e-8)
        recall_micro = TP_total / (TP_total + FN_total + 1e-8)
        return (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)).item()


class EarlyStopping:
    def __init__(self, warmup=20,
                 patience=10,
                 cold=3,
                 init_lr=1e-3,
                 min_lr=1e-6,
                 path='./baseline/checkpoint/baseline/cnn.pth'):
        self.warmup = warmup
        self.patience = patience
        self.cold = cold
        self.counter_p = 0
        self.counter_c = 0
        self.early_stop = False
        self.val_loss_min = np.inf
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.path = path
        self.epoch = 0

    def __call__(self, val_loss, model, optimizer):
        if self.warmup == 0:
            optimizer.param_groups[0]['lr'] = self.init_lr
        if self.warmup > self.epoch:
            # Warm-up phase: linearly increase learning rate
            self.epoch += 1
            optimizer.param_groups[0]['lr'] = self.min_lr + (self.init_lr - self.min_lr) * (self.epoch / self.warmup)
        else:
            if val_loss < self.val_loss_min:
                # New lower val_loss, reset patience and save checkpoint
                self.val_loss_min = val_loss
                self.save_checkpoint(model)
                self.counter_p = 0
            else:
                # If val_loss does not improve, increase patience counter
                self.counter_p += 1
                if self.counter_p > self.patience:
                    # If patience limit reached, increase cold counter
                    self.counter_c += 1
                    if self.counter_c > self.cold:
                        # If cold limit reached, trigger early stopping
                        self.early_stop = True
                    else:
                        # Adjust learning rate with a lower bound of min_lr
                        optimizer.param_groups[0]['lr'] = (
                            optimizer.param_groups[0]['lr'] / 5
                            if optimizer.param_groups[0]['lr'] / 5 > self.min_lr
                            else self.min_lr
                        )
                        self.counter_p = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
