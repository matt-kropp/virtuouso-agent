"""Multimodal music model with aligned dual encoders and gated fusion decoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..modules.conditioning import GlobalConditioning
from ..modules.positional import SharedSinusoidalPositionalEncoding


@dataclass
class DecoderOutput:
    audio_logits: torch.Tensor
    midi_logits: torch.Tensor
    alignment_scores: Dict[str, torch.Tensor]


class SpectrogramEncoder(nn.Module):
    """Encodes log-mel spectrogram frames into latent representations."""

    def __init__(self, in_channels: int, model_dim: int, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(in_channels, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, spectrogram: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        x = self.proj(spectrogram) + pos_embed
        return self.encoder(x)


class MIDIEventEncoder(nn.Module):
    """Encodes tokenized MIDI events into latent sequences."""

    def __init__(self, vocab_size: int, model_dim: int, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) + pos_embed
        return self.encoder(x)


class CrossModalAligner(nn.Module):
    """Bidirectional cross-attention to align audio and MIDI latents."""

    def __init__(self, model_dim: int, num_heads: int = 8):
        super().__init__()
        self.audio_to_midi = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.midi_to_audio = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.norm_audio = nn.LayerNorm(model_dim)
        self.norm_midi = nn.LayerNorm(model_dim)

    def forward(
        self, audio_latents: torch.Tensor, midi_latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        midi_ctx, audio_midi_scores = self.audio_to_midi(
            query=midi_latents, key=audio_latents, value=audio_latents, need_weights=True
        )
        audio_ctx, midi_audio_scores = self.midi_to_audio(
            query=audio_latents, key=midi_latents, value=midi_latents, need_weights=True
        )
        audio_aligned = self.norm_audio(audio_latents + audio_ctx)
        midi_aligned = self.norm_midi(midi_latents + midi_ctx)
        return audio_aligned, midi_aligned, {
            "audio_to_midi": audio_midi_scores,
            "midi_to_audio": midi_audio_scores,
        }


class FusionDecoder(nn.Module):
    """Decodes joint future tokens conditioned on audio/MIDI availability."""

    def __init__(self, model_dim: int, audio_vocab: int, midi_vocab: int, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.audio_head = nn.Linear(model_dim, audio_vocab)
        self.midi_head = nn.Linear(model_dim, midi_vocab)
        self.gating = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2),
        )

    def forward(
        self,
        tgt_states: torch.Tensor,
        memory_audio: torch.Tensor,
        memory_midi: torch.Tensor,
        available_modalities: torch.Tensor,
    ) -> DecoderOutput:
        """Run the fusion decoder with modality-aware gating.

        Args:
            tgt_states: Autoregressive input states (B, T, D).
            memory_audio: Encoder memory for audio (B, S_a, D).
            memory_midi: Encoder memory for MIDI (B, S_m, D).
            available_modalities: Binary flags of shape (B, 2) indicating whether
                audio [0] and MIDI [1] streams are present.
        """

        fused_memory = torch.cat([memory_audio.mean(dim=1), memory_midi.mean(dim=1)], dim=-1)
        gate_logits = self.gating(fused_memory)
        gate_weights = torch.sigmoid(gate_logits) * available_modalities
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        weight_audio = gate_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        weight_midi = gate_weights[:, 1].unsqueeze(-1).unsqueeze(-1)

        weighted_audio = memory_audio * weight_audio
        weighted_midi = memory_midi * weight_midi

        blended_memory = torch.cat([weighted_audio, weighted_midi], dim=1)

        decoded = self.decoder(tgt_states, blended_memory)
        audio_logits = self.audio_head(decoded)
        midi_logits = self.midi_head(decoded)

        return DecoderOutput(audio_logits=audio_logits, midi_logits=midi_logits, alignment_scores={})


class AudioCodecWrapper(nn.Module):
    """Wrapper around a vector-quantized codec such as EnCodec 48 kHz."""

    def __init__(self, codec: nn.Module, codebook_size: int, hop_length: int):
        super().__init__()
        self.codec = codec
        self.codebook_size = codebook_size
        self.hop_length = hop_length

        if hasattr(codec, "quantizer") and hasattr(codec.quantizer, "codebook"):
            cb_size = codec.quantizer.codebook.weight.size(0)
            if cb_size != codebook_size:
                raise ValueError(
                    f"Codec codebook size {cb_size} does not match expected {codebook_size}."
                )

        if hasattr(codec, "hop_length") and codec.hop_length != hop_length:
            raise ValueError(
                f"Codec hop length {codec.hop_length} does not match expected {hop_length}."
            )

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        quantized, _ = self.codec.encode(audio)
        return quantized

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(tokens)


class MultiModalMusicModel(nn.Module):
    def __init__(
        self,
        spectrogram_channels: int,
        midi_vocab: int,
        audio_vocab: int,
        model_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        codec: Optional[nn.Module] = None,
        codebook_size: int = 1024,
        hop_length: int = 320,
        metadata_embedding_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.audio_encoder = SpectrogramEncoder(
            in_channels=spectrogram_channels, model_dim=model_dim, num_layers=num_layers, num_heads=num_heads
        )
        self.midi_encoder = MIDIEventEncoder(
            vocab_size=midi_vocab, model_dim=model_dim, num_layers=num_layers, num_heads=num_heads
        )
        self.positional_encoding = SharedSinusoidalPositionalEncoding(model_dim)
        self.cross_aligner = CrossModalAligner(model_dim=model_dim, num_heads=num_heads)
        self.decoder = FusionDecoder(
            model_dim=model_dim, audio_vocab=audio_vocab, midi_vocab=midi_vocab, num_layers=num_layers, num_heads=num_heads
        )
        self.codec_wrapper = (
            AudioCodecWrapper(codec, codebook_size=codebook_size, hop_length=hop_length)
            if codec is not None
            else None
        )
        self.conditioning = GlobalConditioning(
            latent_dim=model_dim,
            metadata_embedding_sizes=metadata_embedding_sizes,
            conditioning_dim=model_dim,
        )

        self.beat_head = nn.Linear(model_dim, 2)  # beat / non-beat
        self.downbeat_head = nn.Linear(model_dim, 2)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        audio_spectrogram: torch.Tensor,
        midi_tokens: torch.Tensor,
        tgt_states: torch.Tensor,
        available_modalities: torch.Tensor,
        text_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[DecoderOutput, Dict[str, torch.Tensor]]:
        batch_size, audio_len, _ = audio_spectrogram.shape
        _, midi_len = midi_tokens.shape

        audio_pe = self.positional_encoding(audio_len, device=audio_spectrogram.device)
        midi_pe = self.positional_encoding(midi_len, device=midi_tokens.device)

        audio_latents = self.audio_encoder(audio_spectrogram, audio_pe)
        midi_latents = self.midi_encoder(midi_tokens, midi_pe)

        audio_aligned, midi_aligned, alignment_scores = self.cross_aligner(audio_latents, midi_latents)

        conditioning = self.conditioning(
            audio_latents=audio_aligned,
            midi_latents=midi_aligned,
            text_prompt=text_prompt,
            metadata=metadata,
        )

        conditioning_expanded = conditioning.unsqueeze(1).expand(-1, tgt_states.size(1), -1)
        decoder_input = tgt_states + conditioning_expanded

        decoder_output = self.decoder(
            tgt_states=decoder_input,
            memory_audio=audio_aligned,
            memory_midi=midi_aligned,
            available_modalities=available_modalities,
        )

        decoder_output.alignment_scores = alignment_scores

        aux_outputs = self._compute_auxiliary_predictions(audio_aligned, midi_aligned)

        return decoder_output, aux_outputs

    def _compute_auxiliary_predictions(
        self, audio_latents: torch.Tensor, midi_latents: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        fused = (audio_latents + midi_latents) / 2
        beat_logits = self.beat_head(fused)
        downbeat_logits = self.downbeat_head(fused)

        audio_pooled = audio_latents.mean(dim=1)
        midi_pooled = midi_latents.mean(dim=1)
        contrastive_logits = torch.matmul(
            F.normalize(audio_pooled, dim=-1), F.normalize(midi_pooled, dim=-1).t()
        ) / self.temperature.exp()

        return {
            "beat_logits": beat_logits,
            "downbeat_logits": downbeat_logits,
            "contrastive_logits": contrastive_logits,
        }

    def compute_losses(
        self,
        decoder_output: DecoderOutput,
        aux_outputs: Dict[str, torch.Tensor],
        target_audio_tokens: torch.Tensor,
        target_midi_tokens: torch.Tensor,
        beat_labels: torch.Tensor,
        downbeat_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        losses["audio"] = F.cross_entropy(
            decoder_output.audio_logits.reshape(-1, decoder_output.audio_logits.size(-1)),
            target_audio_tokens.reshape(-1),
        )
        losses["midi"] = F.cross_entropy(
            decoder_output.midi_logits.reshape(-1, decoder_output.midi_logits.size(-1)),
            target_midi_tokens.reshape(-1),
        )

        contrastive_logits = aux_outputs["contrastive_logits"]
        contrastive_labels = torch.arange(contrastive_logits.size(0), device=contrastive_logits.device)
        losses["alignment_contrastive"] = F.cross_entropy(contrastive_logits, contrastive_labels)

        beat_logits = aux_outputs["beat_logits"].reshape(-1, 2)
        downbeat_logits = aux_outputs["downbeat_logits"].reshape(-1, 2)
        losses["beat"] = F.cross_entropy(beat_logits, beat_labels.reshape(-1))
        losses["downbeat"] = F.cross_entropy(downbeat_logits, downbeat_labels.reshape(-1))

        losses["total"] = sum(losses.values())
        return losses

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self.codec_wrapper is None:
            raise RuntimeError("No codec configured for audio encoding")
        return self.codec_wrapper.encode(audio)

    def decode_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.codec_wrapper is None:
            raise RuntimeError("No codec configured for audio decoding")
        return self.codec_wrapper.decode(tokens)
