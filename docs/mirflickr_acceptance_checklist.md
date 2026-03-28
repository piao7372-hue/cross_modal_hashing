# MIRFlickr Raw Cleaning Acceptance Checklist (Draft)

## Current phase
1. Scope is dataset cleaning/data layer only.
2. Out of scope: split, feature extraction, training, evaluation, model, loss.
3. The ra paper MIRFlickr `<20 labels` filter is experiment-stage only and must not be used as raw cleaning logic.

## Preconditions
1. Raw root exists: `data/raw/mirflickr25k/`.
2. Canonical image root exists: `data/raw/mirflickr25k/mirflickr/`.
3. Canonical text root exists: `data/raw/mirflickr25k/mirflickr/meta/tags_raw/`.
4. Canonical label root exists: `data/raw/mirflickr25k/mirflickr25k_annotations_v080/`.
5. Annotation version marker is present: `v080`.
6. Expected total raw samples: `25000`.
7. Expected canonical image file count: `25000`.
8. Expected canonical text file count: `25000`.
9. Expected annotation file count (excluding README): `38`.

## Config load validation
1. Dataset config loads without schema/key errors.
2. `dataset_name` must be `mirflickr25k`.
3. Canonical text source must be `meta/tags_raw/`.
4. Canonical label source must be `mirflickr25k_annotations_v080/*.txt`.
5. `annotation_source_version` must be `v080`.
6. `*_r1.txt` must map to `relevant_labels_r1`.
7. Empty text policy must be keep+flag (`text_empty=true`), not drop.
8. Output naming must use `mirflickr25k`.

## Canonical alignment validation
1. `sample_id` parses from `im<ID>.jpg`.
2. Image alignment uses canonical image source only.
3. Text alignment uses canonical text source only (`tags_raw`).
4. Label alignment uses `annotations_v080` files excluding `README.txt`.
5. Label namespaces are parallel:
6. `potential_labels`
7. `relevant_labels_r1`
8. No single namespace is treated as the only authoritative master label set.

## Schema validation
1. `clean_manifest.jsonl` schema is consistent across rows.
2. Required fields exist: identity key, image path, canonical text, `text_empty`, label namespaces.
3. Canonical text field preserves raw content (no normalization in raw cleaning).
4. `dropped_samples.jsonl` has traceable sample id and `drop_reason`.
5. `clean_stats.json` includes row counts, drop reason breakdown, and duplicate-annotation dedup record.

## Sample-level validation
1. Manual checks from head/middle/tail: completed.
2. Sampling basis: `data/processed/mirflickr25k/clean_manifest.jsonl` current file order.
3. Fixed sample positions (by manifest line order): head=`1`, middle=`12500`, tail=`25000`.
4. Sampling is NOT based on `image_id` numeric sorting.
5. Sample ids checked:
6. line 1 -> `id=mirflickr25k:1`, `image_id=1`
7. line 12500 -> `id=mirflickr25k:21247`, `image_id=21247`
8. line 25000 -> `id=mirflickr25k:9999`, `image_id=9999`
9. For all 3 samples:
10. canonical image path exists and is aligned.
11. canonical text path (`tags_raw/tags<ID>.txt`) exists and is aligned.
12. `text_empty` matches actual text emptiness.
13. manifest `text` equals raw text file content.
14. annotation/labels are traceable in `mirflickr25k_annotations_v080` and consistent with namespace policy.
15. Alignment anomaly found: none for sampled rows.
16. Dropped samples per reason sampled for audit: current_not_applicable (full run `dropped_records=0`); still_open for future runs when dropped rows appear.

## Stats validation
1. `actual_total_rows`: `25000`.
2. `actual_clean_rows`: `25000`.
3. `dropped_records`: `0`.
4. `drop_reason_breakdown`: `{}`.
5. `empty_text_rows`: `2128`.
6. Annotation dedup statistics inside label files: `0`.
7. Annotation source version reported as `v080`.

## Non-dry-run output validation
1. Required outputs must exist after non-dry-run:
2. `data/processed/mirflickr25k/clean_manifest.jsonl`
3. `data/processed/mirflickr25k/dropped_samples.jsonl`
4. `outputs/mirflickr25k/clean_stats.json`
5. `outputs/mirflickr25k/cleaning_report.md`
6. Files are readable and schema-consistent.

## `--limit 200` small-sample stability validation
1. `--limit 200` run succeeds without schema drift.
2. Clean records written are `<= 200`.
3. Drop records remain well-formed.
4. Empty-text keep+flag behavior is unchanged.
5. Label namespace behavior is unchanged.
6. Repeated runs with same inputs/config keep stable schema and reason taxonomy.

## Non-regression invariants
1. `dataset_name=mirflickr25k` remains unchanged unless policy update is approved.
2. Canonical text source remains `tags_raw`.
3. Canonical label source remains `annotations_v080` with version marker `v080`.
4. `potential_labels` and `relevant_labels_r1` remain parallel namespaces.
5. Empty text remains keep+flag, not drop.
6. Missing canonical text file remains drop (`drop_reason=missing_text_file`).
7. Annotation parse failure remains hard fail.
8. Primary alignment key conflict remains hard fail.
9. Annotation in-file duplicates remain dedup+record.
10. Execution stage strict full coverage remains `false`.
11. Acceptance stage still enforces expected-count coverage checks.

## Explicitly forbidden shortcuts
1. Do not switch canonical text source to `meta/tags/`.
2. Do not apply ra `<20 labels` filtering in raw cleaning.
3. Do not collapse `*_r1.txt` into the only master label set.
4. Do not drop samples solely because text is empty.
5. Do not downgrade annotation parse failure from hard fail to silent continue.
6. Do not ignore primary key conflicts.
7. Do not change output root naming away from `mirflickr25k`.
