"""Positional encoding utilities for aligning audio and symbolic streams."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class SharedSinusoidalPositionalEncoding(nn.Module):
    """A sinusoidal positional encoding shared across modalities.

    Using a shared module ensures that both audio and symbolic sequences are
    placed within the same positional reference frame before cross attention.
    """

    def __init__(self, dim: int, max_length: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_length = max_length

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if length > self.max_length:
            raise ValueError(
                f"Requested positional encoding of length {length}, but max_length={self.max_length}."
            )
        encoding = self.pe[:length]
        if device is not None:
            encoding = encoding.to(device)
        return encoding
