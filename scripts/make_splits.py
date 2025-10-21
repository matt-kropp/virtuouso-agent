"""Utility to generate stratified dataset splits for piano corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


@dataclass(frozen=True)
class Piece:
    piece_id: str
    composer: str
    era: str
    difficulty: str
    metadata_path: Path
    duration_seconds: float
    notes: int


SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.10,
    "blind": 0.05,
}


def load_metadata(metadata_dir: Path) -> List[Piece]:
    """Load piece metadata from JSON files."""
    pieces: List[Piece] = []
    for json_path in sorted(metadata_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        pieces.append(
            Piece(
                piece_id=payload["piece_id"],
                composer=payload["composer"],
                era=payload["era"],
                difficulty=payload["difficulty_level"],
                metadata_path=json_path,
                duration_seconds=float(payload["duration_seconds"]),
                notes=int(payload["note_count"]),
            )
        )
    return pieces


def stratify(pieces: Iterable[Piece]) -> Dict[Tuple[str, str, str], List[Piece]]:
    """Group pieces by composer, era, and difficulty for stratified sampling."""
    buckets: Dict[Tuple[str, str, str], List[Piece]] = defaultdict(list)
    for piece in pieces:
        key = (piece.composer, piece.era, piece.difficulty)
        buckets[key].append(piece)
    return buckets


def _bucket_seed(key: Tuple[str, str, str], seed: int) -> int:
    """Create a deterministic seed per bucket using a stable hash."""
    digest = hashlib.blake2b(
        repr((key, seed)).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def allocate_bucket(pieces: List[Piece], seed: int) -> Dict[str, List[Piece]]:
    """Allocate a bucket of pieces into the desired splits deterministically."""
    pieces = list(pieces)
    random.Random(seed).shuffle(pieces)
    counts = {split: int(len(pieces) * ratio) for split, ratio in SPLIT_RATIOS.items()}

    # Ensure allocation sums to bucket length by distributing remainder greedily.
    assigned_total = sum(counts.values())
    remainder = len(pieces) - assigned_total
    split_cycle = ["train", "validation", "test", "blind"]
    idx = 0
    while remainder > 0:
        counts[split_cycle[idx % len(split_cycle)]] += 1
        idx += 1
        remainder -= 1

    allocation: Dict[str, List[Piece]] = {split: [] for split in SPLIT_RATIOS}
    cursor = 0
    for split in split_cycle:
        next_cursor = cursor + counts[split]
        allocation[split].extend(pieces[cursor:next_cursor])
        cursor = next_cursor
    return allocation


def split_dataset(pieces: List[Piece], seed: int) -> Dict[str, List[Piece]]:
    buckets = stratify(pieces)
    splits: Dict[str, List[Piece]] = {split: [] for split in SPLIT_RATIOS}
    for key, bucket_pieces in buckets.items():
        bucket_seed = _bucket_seed(key, seed)
        allocation = allocate_bucket(bucket_pieces, seed=bucket_seed)
        for split, assigned in allocation.items():
            splits[split].extend(assigned)
    return splits


def summarize_split(split_pieces: List[Piece]) -> Dict[str, float]:
    duration = sum(piece.duration_seconds for piece in split_pieces)
    notes = sum(piece.notes for piece in split_pieces)
    composers = {piece.composer for piece in split_pieces}
    return {
        "total_pieces": len(split_pieces),
        "duration_hours": duration / 3600.0,
        "note_events": notes,
        "unique_composers": len(composers),
    }


def save_manifests(splits: Dict[str, List[Piece]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, pieces in splits.items():
        manifest = []
        for piece in pieces:
            manifest.append(
                {
                    "piece_id": piece.piece_id,
                    "metadata": str(piece.metadata_path),
                    "composer": piece.composer,
                    "era": piece.era,
                    "difficulty_level": piece.difficulty,
                    "duration_seconds": piece.duration_seconds,
                    "note_count": piece.notes,
                }
            )
        manifest_path = output_dir / f"{split}.json"
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        summary[split] = summarize_split(pieces)

    summary_path = output_dir / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_dir", type=Path, help="Directory with per-piece metadata JSON files")
    parser.add_argument("output_dir", type=Path, help="Directory to write split manifests")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for deterministic shuffling")
    args = parser.parse_args()

    pieces = load_metadata(args.metadata_dir)
    if not pieces:
        raise SystemExit("No metadata JSON files found; aborting.")

    splits = split_dataset(pieces, seed=args.seed)
    save_manifests(splits, args.output_dir)


if __name__ == "__main__":
    main()
