# MSCOCO Raw Cleaning Acceptance Checklist (Decision-Synced Draft)

## Current phase
1. Scope is dataset cleaning/data layer only.
2. Out of scope: split, feature extraction, training, evaluation, model, loss.
3. This checklist is synchronized with confirmed decisions B1-B8.
4. Paper experiment rules must not be used directly as raw cleaning rules.

## Confirmed baseline decisions (effective)
1. canonical scope = `train2014 + val2014`; `test2014` excluded from canonical.
2. row unit = `per-image-caption`.
3. row-level unique id = `mscoco:<split>:<image_id>:<caption_ann_id>`.
4. label representation = class index list.
5. primary alignment key = `(split, image_id)`.
6. structural corruption = hard-fail; row-level issues = drop.
7. empty caption = keep + flag; missing caption mapping = drop.
8. `image_info_test2014.json` = reference-only (not canonical, not count-freeze input).
9. count freeze sequence = confirm B1-B7 first, then freeze counts.

## Preconditions (raw inventory consistency)
1. Raw root exists: `data/raw/mscoco/`.
2. Canonical image roots exist:
3. `data/raw/mscoco/train2014/train2014/`
4. `data/raw/mscoco/val2014/val2014/`
5. Canonical caption sources exist:
6. `data/raw/mscoco/annotations_trainval2014/annotations/captions_train2014.json`
7. `data/raw/mscoco/annotations_trainval2014/annotations/captions_val2014.json`
8. Canonical instances sources exist:
9. `data/raw/mscoco/annotations_trainval2014/annotations/instances_train2014.json`
10. `data/raw/mscoco/annotations_trainval2014/annotations/instances_val2014.json`
11. Reference-only source exists:
12. `data/raw/mscoco/image_info_test2014/annotations/image_info_test2014.json`

## Acceptance dimensions
### A) Raw image counts
1. `train2014/train2014/*.jpg` count must be `82783`.
2. `val2014/val2014/*.jpg` count must be `40504`.
3. `test2014/*.jpg` count must be `40775` as inventory/reference check only.

### B) Caption source counts and coverage
1. `captions_train2014.json`: `images_count=82783`, `annotations_count=414113`.
2. `captions_val2014.json`: `images_count=40504`, `annotations_count=202654`.
3. Top-level keys must include `info`, `images`, `licenses`, `annotations`.
4. caption image coverage missing must be `0` in train and `0` in val.

### C) Instances source counts and category consistency
1. `instances_train2014.json`: `images_count=82783`, `annotations_count=604907`, `categories_count=80`.
2. `instances_val2014.json`: `images_count=40504`, `annotations_count=291875`, `categories_count=80`.
3. Top-level keys must include `info`, `images`, `licenses`, `annotations`, `categories`.
4. instance image ids not found in split images must be `0` in train and `0` in val.
5. category id sets train/val must be consistent (`train_not_in_val=0`, `val_not_in_train=0`).

### D) Canonical row count
1. canonical scope is fixed to `train+val`.
2. row unit is fixed to `per-image-caption`.
3. `expected_total_rows = 616767`.
4. split-level row expectation:
5. train rows = `414113`
6. val rows = `202654`

### E) Alignment key consistency
1. primary key must be `(split, image_id)`.
2. `COCO_<split>_<12-digit-id>.jpg` parsed id must match json `images[].id`.
3. cross-source joins (images/captions/instances) must be reproducible under split-aware key.

### F) Text handling consistency
1. text must remain source-faithful without normalization.
2. empty caption rows must be retained and explicitly flagged.
3. missing caption mapping must be dropped with explicit reason.

### G) Drop/hard-fail boundary
1. hard-fail (not drop-reason) must include:
2. canonical JSON structural corruption / unreadable source
3. required top-level keys missing in canonical source JSON
4. irrecoverable structural alignment failure (non-row-level)
5. row-level issues must go to dropped samples with reason.
6. closed drop-reason enum must be exactly:
7. `missing_image`, `invalid_image_identity`, `missing_caption`, `caption_parse_failure`, `instances_parse_failure`, `alignment_key_conflict`
8. `image_without_instances` must not appear as a drop reason.
9. drop-record required fields must be exactly:
10. `id`, `dataset_name`, `row_index`, `split`, `image_id`, `caption_ann_id`, `candidate_image_paths`, `drop_reason`, `detail`
11. `drop_reason_breakdown` must be present in clean stats.

### H) Label payload consistency
1. manifest label payload must use class index list.
2. label assignment must be traceable back to `instances_*.json`.
3. observed images without instances are train=`702`, val=`367`.
4. rows with no instance annotations must be kept as valid image-text rows.
5. these rows must carry `labels=[]`.
6. these rows must carry `has_instances=false` and `labels_empty_by_source=true`.
7. this condition is an audit/frozen statistic, not a drop item.

## Manual audit plan (required)
1. head/middle/tail manual checks on final `clean_manifest.jsonl`.
2. for each sampled row verify image split, key fields, caption traceability, label traceability.
3. at least one dropped sample per non-zero drop reason must be audited.
4. zero-count reasons remain `future-triggered / still_open`.

## Full-run freeze targets (after implementation and non-dry-run)
1. `expected_total_rows = 616767` must be auditable against canonical caption sources.
2. `actual_total_rows`
3. `actual_clean_rows`
4. `dropped_records`
5. `drop_reason_breakdown`
6. split-level clean row counts (`train`, `val`)
7. empty-caption counter
8. `images_without_instances_train = 702`
9. `images_without_instances_val = 367`
10. rows-without-instances handling consistency:
11. keep rows, `labels=[]`, `has_instances=false`, `labels_empty_by_source=true`, and not counted in drop taxonomy

## Remaining TO_BE_CONFIRMED items
1. whether reference-only metadata should be mirrored into manifest auxiliary fields
2. any optional auxiliary field expansion beyond current confirmed output contract

## Draft-only statement
1. this is a decision-synced acceptance checklist draft.
2. it is not an implementation run report.
3. cleaner/config/pipeline changes are out of scope in this step.
