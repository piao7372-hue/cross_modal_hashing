# NUS-WIDE Canonical Raw Alignment Policy

## Scope Note (2026-04 Sync)
1. This file is a NUS-WIDE cleaning-stage alignment policy artifact.
2. It is a historical sub-stage document and does not define the current full-project phase.
3. Current implementation status is determined by repository code/config first, and aligned with `docs/project_status.md` plus `docs/semantic_cache_contract.md`.

## Purpose
This document defines the canonical alignment policy for NUS-WIDE raw data cleaning in this repository.
It applies to dataset cleaning only and does not cover feature extraction, training, evaluation, or model code.

## Canonical Metadata Sources
The canonical raw alignment must use only the following files:

1. `data/raw/nuswide/NUS_WID_Tags/All_Tags.txt`
2. `data/raw/nuswide/NUS_WID_Tags/AllTags81.txt`
3. `data/raw/nuswide/NUS_WID_Tags/AllTags1k.txt`
4. `data/raw/nuswide/Groundtruth/AllLabels/Labels_*.txt`

These files are treated as the canonical full-set metadata space.

## Non-Canonical Sources
`Groundtruth/TrainTestLabels` must not be used as the main label source for canonical raw alignment.

Reason:

1. `Labels_lake_Train.txt` is confirmed to contain a source anomaly (`-1130179`).
2. Train/Test label files are only reference material for comparison, not for canonical alignment.

## Canonical Alignment Key
The canonical primary key comes from the first token (image id) in `All_Tags.txt`.

For each metadata row:

1. Parse `image_id` from `All_Tags.txt` first token.
2. Build candidate image files by matching `images/*_{image_id}.jpg`.
3. Align `AllTags81`, `AllTags1k`, and `Labels_*.txt` by canonical row index.

## Duplicate Image ID Policy
Duplicate `image_id` means one id appears in multiple metadata rows and multiple image files.

### Rule A: Safe Duplicate
If duplicate rows for the same id have both:

1. identical tag text in `All_Tags.txt`
2. identical 81-d label vector from `Labels_*.txt`

then keep all occurrences and apply deterministic one-to-one mapping:

1. sort image filenames ascending
2. sort duplicate metadata rows by row index ascending
3. pair in order

### Rule B: Ambiguous Duplicate
If duplicate rows for the same id differ in either:

1. tag text, or
2. 81-d label vector

then drop all occurrences of that id and record:

1. `drop_reason = ambiguous_duplicate_id`

## Pre-Confirmed Ambiguous IDs
Based on repository scan results, the following ids are pre-confirmed ambiguous and must be dropped:

1. `1100787682`
2. `2728487708`
3. `702409954`

## Legacy Subset Policy
`data/raw/nuswide/legacy_subset/*` is reference-only and must not participate in canonical raw alignment.

It can be used only for historical comparison/reporting, not for authoritative row matching.

## Expected Cleaning Outputs
NUS-WIDE cleaning output naming must be fixed as:

1. `data/processed/nuswide/clean_manifest.jsonl`
2. `data/processed/nuswide/dropped_samples.jsonl`
3. `outputs/nuswide/clean_stats.json`
4. `outputs/nuswide/cleaning_report.md`
