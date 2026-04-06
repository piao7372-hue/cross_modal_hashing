# MSCOCO Canonical Raw Alignment Policy (Decision-Synced Draft)

## Scope Note (2026-04 Sync)
1. This file is a MSCOCO cleaning-stage alignment policy artifact.
2. It is a historical sub-stage document and does not define the current full-project phase.
3. Current implementation status is determined by repository code/config first, and aligned with `docs/project_status.md` plus `docs/semantic_cache_contract.md`.

## 1. Purpose
This document records the currently effective MSCOCO raw cleaning policy decisions for planning.
Scope is limited to `data cleaning / data layer` and does not authorize implementation by itself.

Decision source:
1. `docs/mscoco_freeze_blockers.md` (B1-B8 confirmed)
2. `docs/mscoco_count_freeze_prep.md` (count freeze preparation evidence)

## 2. Stage Boundary
Out of scope:
1. feature extraction
2. split for training/evaluation protocol
3. training
4. evaluation
5. model/loss implementation

Paper experiment rules must not be used directly as raw cleaning rules.

## 3. Canonical Scope (Confirmed)
Canonical image scope:
1. `data/raw/mscoco/train2014/train2014/*.jpg`
2. `data/raw/mscoco/val2014/val2014/*.jpg`

Explicitly non-canonical in current phase:
1. `data/raw/mscoco/test2014/*.jpg`

## 4. Canonical Text Source And Row Unit (Confirmed)
Canonical text source:
1. `data/raw/mscoco/annotations_trainval2014/annotations/captions_train2014.json`
2. `data/raw/mscoco/annotations_trainval2014/annotations/captions_val2014.json`

Confirmed row unit:
1. `per-image-caption`

Text handling:
1. source-faithful raw caption content
2. no normalization at raw cleaning stage
3. empty caption = keep + flag
4. missing caption mapping = drop

## 5. Canonical Label Source (Confirmed)
Canonical label/annotation source:
1. `data/raw/mscoco/annotations_trainval2014/annotations/instances_train2014.json`
2. `data/raw/mscoco/annotations_trainval2014/annotations/instances_val2014.json`

Label representation:
1. class index list

Category mapping source:
1. `categories` in `instances_*.json`

Images without instances handling (confirmed):
1. observed counts: `images_without_instances_train = 702`, `images_without_instances_val = 367`
2. these samples are kept as valid image-text rows (not dropped)
3. label payload is empty list: `labels = []`
4. required explicit semantics:
5. `has_instances = false`
6. `labels_empty_by_source = true`
7. `image_without_instances` is not a drop reason and does not enter drop taxonomy

## 6. Primary Alignment Key (Confirmed)
Primary alignment key:
1. `(split, image_id)`

Required consistency checks:
1. parse `image_id` from `COCO_<split>_<12-digit-id>.jpg`
2. parsed id must match json `images[].id`

## 7. Drop/Hard-Fail Boundary (Confirmed)
Hard-fail boundary:
1. canonical JSON structural corruption / unreadable source
2. required top-level keys missing in canonical source JSON
3. irrecoverable structural alignment failure (non-row-level)

Drop boundary:
1. row-level issues are dropped with explicit reason

Closed row-level drop reasons (confirmed):
1. `missing_image`
2. `invalid_image_identity`
3. `missing_caption`
4. `caption_parse_failure`
5. `instances_parse_failure`
6. `alignment_key_conflict`
7. explicitly excluded: `image_without_instances`

Drop-record field names (confirmed):
1. `id`
2. `dataset_name`
3. `row_index`
4. `split`
5. `image_id`
6. `caption_ann_id`
7. `candidate_image_paths`
8. `drop_reason`
9. `detail`

## 8. Reference-Only Sources (Confirmed)
Reference-only in this phase:
1. `data/raw/mscoco/image_info_test2014/annotations/image_info_test2014.json`
2. `data/raw/mscoco/annotations_trainval2014/annotations/person_keypoints_train2014.json`
3. `data/raw/mscoco/annotations_trainval2014/annotations/person_keypoints_val2014.json`

`image_info_test2014.json`:
1. does not enter canonical metadata
2. does not participate in current canonical alignment
3. does not participate in current count freeze

## 9. Count Freeze Inputs (Current)
From confirmed scope and row unit:
1. canonical scope = `train+val`
2. row unit = `per-image-caption`
3. `expected_total_rows = 616767`

Supporting evidence:
1. train caption annotations = `414113`
2. val caption annotations = `202654`
3. train/val caption image coverage missing = `0`

Observed instance-side coverage note:
1. images without instances: train=`702`, val=`367`
2. confirmed handling: keep rows with `labels=[]`, `has_instances=false`, `labels_empty_by_source=true`.

## 10. Remaining TO_BE_CONFIRMED Items
1. whether to mirror reference-only metadata into manifest auxiliary fields
2. any optional auxiliary field expansion beyond current confirmed output contract

## 11. Row-Level Unique ID (Confirmed)
Row-level unique id format:
1. `id = mscoco:<split>:<image_id>:<caption_ann_id>`
2. this is the default and confirmed format for `per-image-caption` rows
## 12. Draft-Only Statement
This is a decision-synced policy draft.
It is not an implementation artifact.
The out-of-scope wording above is scoped to this cleaning-policy document itself.
