"""Global conditioning utilities for multimodal music modeling."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class GlobalConditioning(nn.Module):
    """Computes pooled global conditioning vectors from modality latents.

    The module supports optional text prompts, categorical metadata (e.g. composer
    or instrumentation), and style embeddings inferred from the audio/MIDI streams.
    """

    def __init__(
        self,
        latent_dim: int,
        text_encoder: Optional[nn.Module] = None,
        metadata_embedding_sizes: Optional[Dict[str, int]] = None,
        conditioning_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.text_encoder = text_encoder
        self.conditioning_dim = conditioning_dim

        self.pooler = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

        self.metadata_embeddings = nn.ModuleDict()
        if metadata_embedding_sizes:
            for key, size in metadata_embedding_sizes.items():
                self.metadata_embeddings[key] = nn.Embedding(size, conditioning_dim)

        self.fusion = nn.Sequential(
            nn.LayerNorm(conditioning_dim * (1 + len(self.metadata_embeddings) + 1)),
            nn.Linear(conditioning_dim * (1 + len(self.metadata_embeddings) + 1), conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

    def forward(
        self,
        audio_latents: torch.Tensor,
        midi_latents: torch.Tensor,
        text_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return a single conditioning vector."""

        pooled_audio = self.pooler(audio_latents.mean(dim=1))
        pooled_midi = self.pooler(midi_latents.mean(dim=1))
        pooled_style = (pooled_audio + pooled_midi) / 2.0

        conditioning_inputs = [pooled_style]

        if text_prompt is not None and self.text_encoder is not None:
            text_tokens = self.text_encoder(text_prompt)
            if text_tokens.dim() == 1:
                text_tokens = text_tokens.unsqueeze(0).expand_as(pooled_style)
            if text_tokens.size(-1) != pooled_style.size(-1):
                raise ValueError(
                    "Text encoder output dimension must match conditioning_dim"
                )
            conditioning_inputs.append(text_tokens)
        else:
            conditioning_inputs.append(torch.zeros_like(pooled_style))

        if metadata:
            for key, embedding in self.metadata_embeddings.items():
                if key not in metadata:
                    raise KeyError(f"Missing metadata field '{key}' for conditioning")
                conditioning_inputs.append(embedding(metadata[key].long()))
        else:
            for _ in self.metadata_embeddings.values():
                conditioning_inputs.append(torch.zeros_like(pooled_style))

        fused = torch.cat(conditioning_inputs, dim=-1)
        return self.fusion(fused)
