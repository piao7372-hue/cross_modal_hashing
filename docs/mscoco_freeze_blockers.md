# MSCOCO Freeze Blockers (Decision-Landed, Pre-Freeze)

## Scope
This file records blocker decisions already confirmed by human review.
It does not start implementation by itself and does not replace count-freeze closure.

References:
1. `docs/mscoco_raw_inventory.md`
2. `docs/mscoco_alignment_policy.md`
3. `docs/mscoco_acceptance_checklist.md`

## Decision Snapshot
1. B1 = A (`train+val` canonical, `test2014` excluded from canonical)
2. B2 = A (`per-image-caption`)
3. B3 = A (label as class index list)
4. B4 = A (primary key = `(split, image_id)`)
5. B5 = A (structural corruption hard-fail; other row-level issues drop)
6. B6 = A (empty caption keep+flag; missing caption drop)
7. B7 = A (`image_info_test2014.json` stays reference-only)
8. B8 = A (freeze expected counts only after B1-B7 are confirmed)
9. B3-detail confirmed: `images without instances` are kept, `labels=[]`, `has_instances=false`, `labels_empty_by_source=true`, and excluded from drop taxonomy
10. Drop/hard-fail boundary details confirmed:
11. hard-fail = canonical JSON structural corruption / unreadable source, required top-level keys missing, irrecoverable structural alignment failure (non-row-level)
12. closed drop-reason enum = `missing_image`, `invalid_image_identity`, `missing_caption`, `caption_parse_failure`, `instances_parse_failure`, `alignment_key_conflict`
13. drop-record fields = `id`, `dataset_name`, `row_index`, `split`, `image_id`, `caption_ann_id`, `candidate_image_paths`, `drop_reason`, `detail`
14. row-level unique id confirmed = `id = mscoco:<split>:<image_id>:<caption_ann_id>`

## Blocker Decisions

### B1 Canonical split scope
- Problem: include `test2014` in canonical scope or not.
- Options: A=`train+val`; B=`train+val+test`.
- Confirmed choice: A.
- Impact:
- canonical row count anchored to train+val only.
- manifest schema does not require canonical test branch now.
- drop taxonomy excludes test-caption related drop reasons.
- implementation path remains two-way split alignment.
- Final status: `CONFIRMED`

### B2 Canonical row unit
- Problem: row unit in canonical manifest.
- Options: A=`per-image-caption`; B=`per-image+captions[]`; C=`per-image+single-caption`.
- Confirmed choice: A.
- Impact:
- canonical row count follows caption annotation volume.
- manifest schema centers on one caption payload per row.
- drop taxonomy evaluates caption-level missing/parse issues.
- implementation aligns on caption annotations as row generator.
- Final status: `CONFIRMED`

### B3 Label representation
- Problem: label payload shape.
- Options: A=class index list; B=multi-hot; C=both.
- Confirmed choice: A.
- Impact:
- canonical row count unaffected directly.
- manifest schema stores label index list fields.
- drop taxonomy focuses on annotation parsing/alignment validity, not vector-shape checks.
- implementation avoids early vectorization coupling.
- Final status: `CONFIRMED`

### B3-detail Images without instances handling
- Problem: how to treat image-text rows with no instance annotations.
- Options:
- Option A: keep rows with empty labels and explicit flags.
- Option B: drop with dedicated reason.
- Confirmed choice: A.
- Impact:
- canonical row count: retained rows remain in clean set.
- manifest schema: requires `labels=[]`, `has_instances=false`, `labels_empty_by_source=true`.
- drop taxonomy: `image_without_instances` excluded as drop reason.
- implementation: requires deterministic flag emission and audit counters.
- Frozen statistics:
- `images_without_instances_train = 702`
- `images_without_instances_val = 367`
- Final status: `CONFIRMED`

### B4 Primary alignment key
- Problem: identity key granularity.
- Options: A=`(split, image_id)`; B=`image_id`.
- Confirmed choice: A.
- Impact:
- canonical row count avoids cross-split key ambiguity.
- manifest schema must preserve split identity field.
- drop taxonomy includes split-aware key conflict handling.
- implementation join logic remains split-scoped and traceable.
- Final status: `CONFIRMED`

### B5 Drop vs hard-fail boundary
- Problem: classify failure handling.
- Options: A=structural hard-fail + row-level drop; B=broader hard-fail.
- Confirmed choice: A.
- Impact:
- canonical row count impacted by row-level drops, not by over-broad aborts.
- manifest schema needs stable dropped-sample reason fields.
- drop taxonomy must define explicit row-level reason set.
- implementation can continue on row-level issues while preserving strict structural integrity.
- hard-fail set is explicitly closed to structural-level failures only.
- closed drop-reason enum and drop-record fields are fixed for implementation entry.
- Final status: `CONFIRMED`

### B6 Empty/missing caption handling
- Problem: retain/drop policy for caption text anomalies.
- Options: A=empty keep+flag, missing drop; B=empty/missing drop; C=empty/missing keep+flag.
- Confirmed choice: A.
- Impact:
- canonical row count keeps empty-caption rows but removes missing-caption rows.
- manifest schema requires empty-caption flag field.
- drop taxonomy must include missing-caption reason.
- implementation must separate empty-text vs missing-caption paths.
- Final status: `CONFIRMED`

### B7 `image_info_test2014.json` role
- Problem: canonical or reference-only.
- Options: A=reference-only; B=canonical metadata for test.
- Confirmed choice: A.
- Impact:
- canonical row count excludes test info file.
- manifest schema does not depend on test-info-only fields.
- drop taxonomy does not require test-info canonical reasons in current phase.
- implementation excludes test-info canonical join.
- Final status: `CONFIRMED`

### B8 Count freeze timing
- Problem: when to freeze expected counts.
- Options: A=after B1-B7; B=before decisions fully settled.
- Confirmed choice: A.
- Impact:
- canonical row count can now move to freeze-prep calculation.
- manifest schema assumptions are stabilized before count freeze.
- drop taxonomy freeze can be aligned with confirmed boundaries.
- implementation can start only after count-freeze closure.
- Final status: `CONFIRMED`

## Non-Blocking Open Item

### N1 Reference-only metadata mirroring
- Problem: whether reference-only channels should be mirrored into manifest auxiliary fields.
- Options: A=do not mirror in first pass; B=mirror selected fields.
- Suggested default: A.
- Impact:
- canonical row count: no direct impact.
- manifest schema: optional extension only.
- drop taxonomy: typically no direct impact.
- implementation: additive enhancement, can be deferred.
- Final status: `TO_BE_CONFIRMED`

## Current Gate Statement
Decision blockers B1-B8 are landed as confirmed choices.
MSCOCO still remains pre-implementation until count-freeze preparation is closed.
