# MSCOCO Raw Inventory (Planning-Only, Draft)

## 1. Scope And Boundary
- This document records only raw data inventory facts for MSCOCO under `data/raw/mscoco/`.
- This is for data cleaning planning bootstrap only.
- This document does **not** freeze cleaning policy.
- This document does **not** introduce cleaner implementation, dataset config, or pipeline registration.

## 2. Baseline Traceability
- Freeze baseline tag (existing): `data-cleaning-freeze-20260328` -> `f384ed6`.
- Current workspace HEAD at inventory time: `3601bc1`.
- Diff from freeze tag to HEAD at inventory time: `docs/mirflickr_handoff.md` only.

## 3. Actually Discovered Raw Directory Structure
- `data/raw/mscoco/annotations_trainval2014/`
- `data/raw/mscoco/annotations_trainval2014/annotations/`
- `data/raw/mscoco/image_info_test2014/`
- `data/raw/mscoco/image_info_test2014/annotations/`
- `data/raw/mscoco/train2014/`
- `data/raw/mscoco/train2014/train2014/`
- `data/raw/mscoco/val2014/`
- `data/raw/mscoco/val2014/val2014/`
- `data/raw/mscoco/test2014/`

## 4. Existence Check For Required Paths
- `data/raw/mscoco/train2014/train2014/`: exists = `true`
- `data/raw/mscoco/val2014/val2014/`: exists = `true`
- `data/raw/mscoco/test2014/`: exists = `true`
- `data/raw/mscoco/annotations_trainval2014/annotations/captions_train2014.json`: exists = `true`
- `data/raw/mscoco/annotations_trainval2014/annotations/captions_val2014.json`: exists = `true`
- `data/raw/mscoco/annotations_trainval2014/annotations/instances_train2014.json`: exists = `true`
- `data/raw/mscoco/annotations_trainval2014/annotations/instances_val2014.json`: exists = `true`
- `data/raw/mscoco/image_info_test2014/annotations/image_info_test2014.json`: exists = `true`

## 5. Image File Counts
- `train2014/train2014/*.jpg`: `82783`
- `val2014/val2014/*.jpg`: `40504`
- `test2014/*.jpg`: `40775`

## 6. Annotation JSON Basic Readability And Minimal Field-Level Stats
- `captions_train2014.json`
- size_bytes: `66782097`
- top_level_keys: `info, images, licenses, annotations`
- images_count: `82783`
- annotations_count: `414113`
- categories_count: `N/A` (key not present)
- licenses_count: `8`

- `captions_val2014.json`
- size_bytes: `32421077`
- top_level_keys: `info, images, licenses, annotations`
- images_count: `40504`
- annotations_count: `202654`
- categories_count: `N/A` (key not present)
- licenses_count: `8`

- `instances_train2014.json`
- size_bytes: `332556225`
- top_level_keys: `info, images, licenses, annotations, categories`
- images_count: `82783`
- annotations_count: `604907`
- categories_count: `80`
- licenses_count: `8`

- `instances_val2014.json`
- size_bytes: `160682675`
- top_level_keys: `info, images, licenses, annotations, categories`
- images_count: `40504`
- annotations_count: `291875`
- categories_count: `80`
- licenses_count: `8`

- `image_info_test2014.json`
- size_bytes: `9131355`
- top_level_keys: `info, images, licenses, categories`
- images_count: `40775`
- annotations_count: `N/A` (key not present)
- categories_count: `80`
- licenses_count: `8`

## 7. Raw Cleaning Pre-Policy Ambiguities
- `TO_BE_CONFIRMED`: whether `test2014` is included in canonical raw cleaning scope, or inventory-only/reference-only.
- `TO_BE_CONFIRMED`: whether canonical image scope is `train+val` only or `train+val+test`.
- `TO_BE_CONFIRMED`: whether canonical label source should include `instances_*.json`, and if yes, how categories/annotations are projected into cleaning manifest.
- `TO_BE_CONFIRMED`: handling of multi-caption per image in `captions_*.json` (single text field vs list field vs other representation).
- `TO_BE_CONFIRMED`: expected output row unit (per-image or per-image-caption pair).
- `TO_BE_CONFIRMED`: required behavior for images existing in `image_info_test2014.json` without caption annotations.
- `TO_BE_CONFIRMED`: whether `categories` in `image_info_test2014.json` are treated as alignment metadata or reference-only.
- `TO_BE_CONFIRMED`: drop-reason taxonomy for missing image, parse failure, and cross-file mismatch in MSCOCO context.

## 8. Explicit Non-Freeze Statement
- This run completed **raw inventory only**.
- This does **not** mean MSCOCO cleaning policy is finalized/frozen.
- Any future alignment rule, drop policy, schema mapping, or scope decision must be confirmed separately and documented with `TO_BE_CONFIRMED` closure.
