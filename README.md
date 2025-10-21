# Virtuouso Dataset Pipeline

This repository documents the workflow for assembling a rich piano performance dataset that combines curated paired audio/MIDI corpora with new transcriptions.

## Documentation

- [Piano Dataset Expansion Pipeline](docs/data_pipeline.md)

## Data Catalog

- Core dataset manifest: `data/catalog/core_datasets.yaml`
- Metadata JSON stubs: `data/metadata/`
- QA records: `data/qa/`
- License references: `licenses/`

## Tooling

- `scripts/make_splits.py`: generate reproducible stratified train/validation/test/blind splits from normalized metadata manifests.

## Getting Started

1. Populate `data/metadata/` with per-piece JSON files following the schema described in the documentation.
2. Run transcription pipelines for additional audio sources as detailed in the docs.
3. Execute `python scripts/make_splits.py data/metadata data/splits` once metadata has been normalized.
4. Review `data/splits/summary.yaml` to ensure coverage balance before finalizing the release.
