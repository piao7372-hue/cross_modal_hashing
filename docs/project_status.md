# Project Status

## Current phase
Project is still in `data cleaning / data layer only`.
Out of scope at this stage: split, feature extraction, training, evaluation, model, loss.
MSCOCO must not be started in the current phase.

## NUS-WIDE status
NUS-WIDE cleaning is completed and frozen.
Rule: do not modify NUS-WIDE cleaning logic unless a clear bug is found.

## MIRFlickr status
MIRFlickr cleaning first pass is completed and frozen.

Completed and integrated:
1. MIRFlickr cleaner implementation
2. dataset config integration
3. pipeline registration and dispatch

Execution status:
1. `--dataset mirflickr25k --dry-run`: passed
2. `--dataset mirflickr25k` full non-dry-run: passed

Frozen counts:
1. `actual_total_rows = 25000`
2. `actual_clean_rows = 25000`
3. `dropped_records = 0`
4. `drop_reason_breakdown = {}`
5. `empty_text_rows = 2128`
6. `annotation dedup statistics inside label files = 0`

Manual acceptance closure:
1. completed for head / middle / tail samples
2. sampling source: `data/processed/mirflickr25k/clean_manifest.jsonl`
3. sampling basis: current file order
4. fixed positions: line `1 / 12500 / 25000`
5. explicitly not sampled by numeric `image_id` sorting

Dropped-sample manual audit:
1. currently not applicable because full run has `dropped_records = 0`
2. retained as future-triggered / still_open item

Traceability:
1. MIRFlickr completion commit: `b6cf66a`

## Constraints reminder
1. Do not start MSCOCO.
2. Do not apply paper ra MIRFlickr `<20 labels` filtering as raw cleaning rule.
3. Keep this document as status summary only; do not use it to introduce new cleaning rules.
