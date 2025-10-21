# Piano Dataset Expansion Pipeline

This document outlines the full pipeline for assembling a comprehensive piano performance dataset that combines existing paired audio/MIDI corpora with newly transcribed material. The steps cover sourcing, licensing, transcription, alignment, metadata normalization, quality assurance, and dataset partitioning.

## 1. Core Paired Datasets and Licenses

| Dataset | Version | Availability | License | Notes |
| --- | --- | --- | --- | --- |
| MAESTRO | v3.0.0 | Google Magenta | Creative Commons Attribution 4.0 | 200 hours of paired Disklavier performances across concert repertoire. |
| Yamaha Disklavier Performances | Various | Yamaha/ENS Music Lab | Custom academic use agreement | Requires institutional request; verify redistribution terms before use. |
| Vienna 4+ Piano | 2020 release | Vienna Symphonic Library | Commercial EULA | Licensed virtual instrument recordings; ensure usage complies with VSL EULA. |
| ASAP (Academic Sankt Augustin Piano) | 2021 release | Télécom Paris/LIMSI | Creative Commons Attribution-NonCommercial 4.0 | Includes precise score alignments; non-commercial clause applies. |
| GiantMIDI-Piano | 1.2 | GitHub (by Shulei Zhang) | Creative Commons Attribution-NonCommercial 4.0 | Automatically transcribed virtuoso repertoire; double-check composer copyright status per region. |

*Action items*
1. Archive dataset manifests (checksums, download URLs, size) in `data/catalog/core_datasets.yaml`.
2. Store license text or links in `licenses/` and track acceptance requirements.

## 2. Expanded Audio Collection and Transcription

1. **Sourcing**
   - Focus on public-domain or commercially cleared recordings: solo piano concertos (solo stems when available), art song accompaniments, chamber works with piano, and film/game scores.
   - Maintain a spreadsheet with recording metadata (source, release year, performers, rights holder) to verify licensing.
2. **Ingestion**
   - Normalize formats to 48 kHz, 24-bit WAV; capture stereo channels where possible.
   - Use `ffmpeg` batch scripts with loudness normalization (`-filter:a loudnorm`) and silence trimming (`-af silenceremove`).
3. **Transcription**
   - Run dual-model transcription for redundancy:
     - [Onsets & Frames](https://github.com/magenta/magenta/tree/main/magenta/models/onsets_frames_transcription) fine-tuned on MAESTRO.
     - [Google MT3](https://github.com/magenta/mt3) with piano checkpoint.
   - Export both MIDI hypotheses and compute confidence metrics (frame F1, onset precision, velocity correlation).
   - Fuse outputs using consensus heuristics (e.g., time-aligned note voting) or feed into a lightweight refinement model.
4. **Post-processing**
   - Apply sustain pedal inference and deduplication of overlapping notes.
   - Quantize tempos to remove transcription jitter while retaining expressive timing.

## 3. Metadata Normalization and Alignment

1. **Metadata schema**
   - Fields: `composer`, `work_title`, `movement`, `catalog_number`, `key`, `time_signature`, `tempo_bpm`, `difficulty_level`, `source_dataset`, `recording_date`, `license`, `alignment_quality`.
   - Store per-piece metadata in `data/metadata/*.json` with ISO-8601 timestamps.
2. **Alignment workflow**
   - Run dynamic time warping (DTW) using chroma features (e.g., `librosa.feature.chroma_cqt`) to align MIDI to audio.
   - Re-synthesize MIDI via high-quality piano VST to verify alignment.
   - Flag items with DTW cost above threshold for manual review.
3. **Manual QA**
   - Stratify sample by dataset, composer era, and difficulty.
   - Use annotation tool (e.g., Audacity, Sonic Visualiser with MIDI overlay) for human validation.
   - Record QA notes and final alignment status in `data/qa/records.csv`.

## 4. Dataset Partitioning

1. **Stratification dimensions**
   - Composer identity (unique IDs per composer).
   - Historical era (Baroque, Classical, Romantic, Impressionist, Modern, Contemporary).
   - Difficulty level (graded using e.g., RCM/ABRSM mapping or virtuosic heuristics).
2. **Splits**
   - **Train**: 70% of pieces, ensuring no overlap in exact work across splits.
   - **Validation**: 15%, balanced across eras and difficulty.
   - **Test**: 10%, withheld pieces with representation from each composer group.
   - **Blind evaluation**: 5%, includes mixed instrumentation tracks (piano + ensemble) and novel sources reserved for final benchmarking.
3. **Implementation**
   - Use deterministic split script (`scripts/make_splits.py`) seeded with metadata.
   - Output JSON manifests per split listing file paths, metadata, and alignment quality scores.
   - Log summary statistics (duration, note counts, composer coverage) for each split.

## Governance and Versioning

- Track dataset versions using DVC or Git LFS for manifests; store large binaries in cloud object storage (S3/GS buckets).
- Maintain `CHANGELOG.md` entries for new data releases and QC notes.
- Establish access controls and audit logs for commercial content.

## Compliance Checklist

- [ ] Licenses reviewed and stored.
- [ ] Transcription models validated against MAESTRO hold-out.
- [ ] Metadata schema enforced via JSON Schema validation.
- [ ] Alignment QA thresholds met (<5% manual fixes).
- [ ] Split reproducibility verified via checksum of manifest files.
