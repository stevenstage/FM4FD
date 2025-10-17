import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
import random

class RandomMasking(nn.Module):
    """Random masking module with span masking strategy"""
    def __init__(self, mask_ratio=0.6, span_length=10):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.span_length = span_length

    def forward(self, inp):
        B, L, _ = inp.size()
        num_masked = int(self.mask_ratio * L)
        mask = torch.ones(B, L, device=inp.device)
        
        # Generate random start indices for masked spans
        starts = torch.randint(0, L - self.span_length, (num_masked,))
        for start in starts:
            end = min(start + self.span_length, L)
            mask[:, start:end] = 0
            
        return mask

class Patching(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, inp):
        B, L, C = inp.shape
        assert L % self.patch_size == 0, "Sequence length must be divisible by patch size"
        return inp.reshape(B, L // self.patch_size, C * self.patch_size)

class SelfGating(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))

class TemporalAPE(nn.Module):
    """Temporal Absolute Position Encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=512, scale=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term * (d_model/max_len))
        pe[:, 1::2] = torch.cos(position * div_term * (d_model/max_len))
        self.register_buffer('pe', scale * pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe)

class Transformer(nn.Module):
    def __init__(self, d_model=256, seq_len=512, num_heads=8, num_layers=6, channel=4,
                 patch_size=16, device='cuda', mask_ratio=0.4):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mask_ratio = mask_ratio

        # Model components
        self.position_enc = TemporalAPE(d_model, max_len=seq_len//patch_size + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.patching = Patching(patch_size)
        self.input_proj = nn.Linear(channel * patch_size, d_model)
        self.self_gating = SelfGating(d_model)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4*d_model
            ),
            num_layers=num_layers
        )
        
        self.masker = RandomMasking(mask_ratio)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, L, C)
        Returns:
            tuple: (original_features, masked_features, mask)
        """
        B, L, C = x.shape
        
        # Process original input
        x_patched = self.patching(x)
        x_proj = self.self_gating(self.input_proj(x_patched))
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x_proj], dim=1)
        x_encoded = self.position_enc(x_with_cls)
        orig_output = self.encoder(x_encoded.transpose(0, 1)).transpose(0, 1)

        if self.mask_ratio > 0:
            # Process masked input
            masked = self.patching(x.clone())
            mask = self.masker(masked)
            masked = self.self_gating(self.input_proj(masked))
            masked[mask == 0] = self.mask_token
            masked = torch.cat([cls_tokens, masked], dim=1)
            masked_output = self.encoder(
                self.position_enc(masked).transpose(0, 1)
            ).transpose(0, 1)
            return orig_output, masked_output, mask
        
        return orig_output, orig_output, torch.zeros(B, orig_output.size(1), device=self.device)