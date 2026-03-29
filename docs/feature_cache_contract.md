# Feature Cache Contract (Stage: Feature Extraction / Feature Cache Layer)

## Scope
This document defines the output contract of the feature extraction stage only.
Out of scope:
1. semantic similarity matrix (`S1/S2/S_final`)
2. SCH-KANH backbone
3. loss, training, evaluation

## Fixed Defaults
1. default encoder: `OpenAI CLIP / openai/clip-vit-base-patch32`
2. default data views:
3. `nuswide -> data/processed/nuswide/clean_manifest.jsonl` (canonical)
4. `mirflickr25k -> data/processed/mirflickr25k/clean_manifest.jsonl`
5. `mscoco -> data/processed/mscoco/clean_manifest.jsonl`
6. `nuswide_ra_top10/filtered_manifest.jsonl` is paper-specific opt-in view only

## Sample Index Contract
1. `sample_index` is zero-based and follows manifest non-empty line order.
2. `row_index` is preserved as metadata only.
3. `row_index` must never be used as continuous matrix position.

## Empty Text Contract
1. empty text rows are kept.
2. empty text is encoded via empty string with the same tokenizer/model path.
3. no empty-text drop is allowed.
4. this guarantees `len(manifest) == rows(X_I) == rows(X_T)`.

## Normalization Contract
1. `X_I` and `X_T` are stored as L2-normalized `float32`.
2. this is the primary contract consumed by next stage.
3. raw features are not required by default.

## MSCOCO Image Reuse Contract
1. internal optimization may use `X_I_unique + row_to_image_idx`.
2. external contract always provides row-aligned `X_I` in manifest order.
3. this preserves direct downstream consumption while reducing image re-encoding.

## Output Layout
For dataset `<dataset>` and feature set `<feature_set_id>`:
`data/processed/<dataset>/feature_cache/<feature_set_id>/`

Required files:
1. `X_I.npy`: row-aligned image feature matrix, shape `[N, D]`, L2-normalized
2. `X_T.npy`: row-aligned text feature matrix, shape `[N, D]`, L2-normalized
3. `row_to_image_idx.npy`: row-to-unique-image mapping, shape `[N]`
4. `sample_index.jsonl`: per-row traceability payload
5. `meta.json`: reproducibility metadata

Optional file:
1. `X_I_unique.npy`: unique image feature matrix, shape `[U, D]`

## Reproducibility Metadata
`meta.json` must record at least:
1. contract version
2. dataset/view/feature_set_id
3. manifest path + sha256 + row count
4. sample index basis and `row_index_used_as_position=false`
5. encoder model/tokenizer/preprocess/max_length/device/dtype/deterministic seed
6. normalization type/epsilon
7. empty text policy summary
8. image reuse summary (`U`, mapping filename, unique cache saved or not)

