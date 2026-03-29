# MSCOCO Count Freeze Preparation (Pre-Implementation)

## 1. Purpose
This document prepares count-freeze inputs after blocker decisions B1-B8 were confirmed.
It does not start cleaner/config/pipeline implementation.

## 2. Confirmed Policy Inputs Used For Count Prep
1. Canonical scope: `train2014 + val2014` only (`test2014` excluded).
2. Row unit: `per-image-caption`.
3. Label payload: class index list.
4. Alignment key: `(split, image_id)`.
5. Failure boundary: structural corruption = hard-fail; row-level issues = drop.
6. Text policy: empty caption keep+flag; missing caption drop.
7. `image_info_test2014.json` remains reference-only.

## 3. Baseline Raw Counts (Read-Only Verification)
### 3.1 Image-space counts
1. `train2014` images: `82783`
2. `val2014` images: `40504`
3. canonical unique images (`train+val`): `123287`

### 3.2 Caption-space counts
1. `captions_train2014` annotations: `414113`
2. `captions_val2014` annotations: `202654`
3. candidate expected total rows before drop (`per-image-caption`): `616767`
4. caption distinct image ids in train: `82783`
5. caption distinct image ids in val: `40504`
6. caption image ids not found in corresponding split images: train=`0`, val=`0`
7. images without any caption: train=`0`, val=`0`
8. caption annotation id duplicates: train=`0`, val=`0`
9. cross-split caption annotation id overlap: `0`

### 3.3 Instance-space counts
1. `instances_train2014` annotations: `604907`
2. `instances_val2014` annotations: `291875`
3. instance distinct image ids: train=`82081`, val=`40137`
4. instance image ids not found in corresponding split images: train=`0`, val=`0`
5. images without any instance annotation: train=`702`, val=`367`
6. categories count: train=`80`, val=`80`
7. category id set difference: train_not_in_val=`0`, val_not_in_train=`0`
8. cross-split image id overlap (train images vs val images): `0`

## 4. Count-Freeze Candidate Fields
Fields ready to freeze after implementation dry-run/non-dry-run validation:
1. `expected_total_rows` candidate: `616767` (caption rows in canonical scope).
2. `expected_clean_rows`: `TO_BE_FINALIZED_AFTER_DROP_PROFILE`
3. `expected_drop_rows`: `TO_BE_FINALIZED_AFTER_DROP_PROFILE`
4. `expected_drop_reason_breakdown`: `TO_BE_FINALIZED_AFTER_DROP_PROFILE`
5. split-level expected rows:
6. train caption rows=`414113`
7. val caption rows=`202654`

## 5. Pre-Freeze Validation Checklist (No Implementation Yet)
1. Confirm operational interpretation for zero-instance images under current decisions:
2. keep row and emit empty label index list (not drop).
3. Confirm missing-caption detection rule at row level is explicit and deterministic.
4. Confirm drop reason names are closed and stable before code phase.
5. Confirm output schema fields for split key and empty-caption flag.
6. After implementation begins, run dry-run first, then non-dry-run to close expected_clean/drop fields.

## 6. Gate Statement
This document is count-freeze preparation only.
Policy implementation is still gated until expected clean/drop counts and drop taxonomy are frozen from run evidence.
