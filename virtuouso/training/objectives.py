"""Training utilities for the multimodal music model."""
from __future__ import annotations

from typing import Dict

import torch

from ..models.multimodal_music_model import DecoderOutput, MultiModalMusicModel


def compute_training_step(
    model: MultiModalMusicModel,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run a forward and loss computation for a single batch."""

    decoder_output, aux_outputs = model(
        audio_spectrogram=batch["audio_spectrogram"],
        midi_tokens=batch["midi_tokens"],
        tgt_states=batch["tgt_states"],
        available_modalities=batch["available_modalities"],
        text_prompt=batch.get("text_prompt"),
        metadata=batch.get("metadata"),
    )

    losses = model.compute_losses(
        decoder_output=decoder_output,
        aux_outputs=aux_outputs,
        target_audio_tokens=batch["target_audio_tokens"],
        target_midi_tokens=batch["target_midi_tokens"],
        beat_labels=batch["beat_labels"],
        downbeat_labels=batch["downbeat_labels"],
    )
    return losses
