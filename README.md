# Virtuouso Multimodal Music Model

This repository provides a reference implementation of a multimodal music model
that aligns audio spectrograms with symbolic MIDI events and performs joint
prediction over both modalities.

## Key features

- Dual encoders for audio spectrograms and MIDI event tokens sharing a common
  sinusoidal positional embedding to keep both streams time-aligned.
- Cross-attention alignment block that exchanges context between modalities to
  produce synchronized latent streams.
- Gated fusion decoder capable of conditioning on audio-only, MIDI-only, or
  combined inputs while autoregressively predicting future audio codec tokens
  and MIDI events.
- Vector-quantized audio codec wrapper (e.g., EnCodec 48 kHz) ensuring codebook
  size and hop length are consistent with the decoder's discrete acoustic token
  interface.
- Training objectives covering joint autoregressive losses, contrastive
  alignment between modalities, and beat/downbeat auxiliary supervision.
- Global conditioning stack that incorporates metadata or optional text prompts
  to steer generation toward desired style, composer, or instrumentation.

See `virtuouso/models/multimodal_music_model.py` for the main model definition
and `virtuouso/training/objectives.py` for training utilities.
