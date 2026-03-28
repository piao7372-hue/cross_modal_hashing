# MIRFlickr Canonical Raw Alignment Policy (Draft)

## 1. Purpose
This document defines the canonical raw alignment policy for MIRFlickr in this repository.
It applies to dataset cleaning/data layer only.

## 2. Current Phase / Out Of Scope
Current phase is dataset cleaning and filtering.
The following are out of scope for this policy:

1. feature extraction
2. data split
3. training
4. evaluation
5. model and loss implementation

Important boundary:
The ra paper MIRFlickr rule "filter pairs with <20 labels" is an experiment-stage rule and is NOT a raw cleaning rule in this policy.

## 3. Raw Inventory Reference
This draft is based on current observed raw inventory under:

1. `data/raw/mirflickr25k/mirflickr/`
2. `data/raw/mirflickr25k/mirflickr25k_annotations_v080/`

Inventory reference document:

1. `docs/mirflickr_raw_inventory.md` (TO_BE_CONFIRMED if filename/content changes)

## 4. Canonical Image Source
Canonical image source is frozen as:

1. `data/raw/mirflickr25k/mirflickr/im<ID>.jpg`

where `<ID>` is an integer sample id in the MIRFlickr25k id space.

## 5. Canonical Text/Tag Source
Canonical text/tag source is frozen as:

1. `data/raw/mirflickr25k/mirflickr/meta/tags_raw/tags<ID>.txt`

`meta/tags/` is not canonical in this draft and is treated as non-canonical/reference-only.

## 6. Canonical Label/Annotation Source
Canonical label source is frozen as:

1. `data/raw/mirflickr25k/mirflickr25k_annotations_v080/*.txt` (except `README.txt`)

Version provenance must be explicitly preserved:

1. annotation_source_version = `v080`

## 7. Non-Canonical Or Reference-Only Sources
The following are reference-only by default in this draft:

1. `data/raw/mirflickr25k/mirflickr/meta/tags/`
2. `data/raw/mirflickr25k/mirflickr/meta/exif/`
3. `data/raw/mirflickr25k/mirflickr/meta/exif_raw/`
4. `data/raw/mirflickr25k/mirflickr/meta/license/`
5. `data/raw/mirflickr25k/mirflickr/doc/common_tags.txt`
6. `data/raw/mirflickr25k/mirflickr/doc/owner_ids.txt`

Whether some of these fields are mirrored into manifest payload remains TO_BE_CONFIRMED.

## 8. Sample Identity Definition
Identity key:

1. `sample_id = <ID>` (integer id extracted from canonical image filename `im<ID>.jpg`)

Cross-file alignment uses the same `<ID>` in:

1. image file `im<ID>.jpg`
2. text file `tags<ID>.txt` from canonical text source
3. annotation line values from `mirflickr25k_annotations_v080/*.txt`

## 9. Image Alignment Rule
For each `sample_id=<ID>`:

1. canonical image must exist at `mirflickr/im<ID>.jpg`
2. filename must parse into a valid integer id
3. invalid parse or missing image is dropped with explicit drop reason

## 10. Text/Tag Alignment Rule
For each `sample_id=<ID>`:

1. resolve text only from `meta/tags_raw/tags<ID>.txt`
2. retain raw text content as canonical text field without normalization
3. text read/parse failures are handled by drop policy

Text normalization policy is frozen:

1. no normalization at raw cleaning stage
2. canonical text is source-faithful raw content

## 11. Label Handling Rule
Label handling uses `mirflickr25k_annotations_v080` with two parallel namespaces:

1. `potential_labels`: from files without `_r1` suffix
2. `relevant_labels_r1`: from files with `_r1` suffix

Frozen decisions:

1. `*_r1.txt` is preserved as separate namespace (`relevant_labels_r1`)
2. `potential_labels` and `relevant_labels_r1` are parallel independent namespaces
3. no single namespace is treated as the only authoritative master label set
4. the annotation package (`annotations_v080`) remains canonical label source overall

## 12. EXIF / License / Doc Positioning
Current draft positioning:

1. `exif/exif_raw/license/doc` are metadata/reference channels
2. they are not primary alignment keys
3. inclusion into `clean_manifest.jsonl` fields is TO_BE_CONFIRMED

## 13. Empty-Text Policy
Frozen policy:

1. empty text samples are kept in raw cleaning stage
2. do not drop solely because text is empty
3. mark with suggested field: `text_empty=true`

## 14. Duplicate / Missing / Parse Failure Policy
Frozen rules:

1. missing canonical image -> drop (`drop_reason=missing_image`)
2. invalid image id parse -> drop (`drop_reason=invalid_image_id`)
3. missing canonical text file -> drop (`drop_reason=missing_text_file`)
4. text read/parse failure -> drop (`drop_reason=text_parse_failure`)
5. annotation parse failure -> hard fail
6. primary alignment key conflict (`sample_id` conflict) -> hard fail
7. duplicate rows inside the same annotation file -> deduplicate and record in stats/report

## 15. Coverage Validation Rule
Frozen rule:

1. execution stage strict full coverage = `false`
2. acceptance stage must check coverage against expected counts

Expected counts remain TO_BE_FILLED in acceptance artifacts.

## 16. Output Contract
Frozen naming:

1. dataset_name = `mirflickr25k`
2. `data/processed/mirflickr25k/clean_manifest.jsonl`
3. `data/processed/mirflickr25k/dropped_samples.jsonl`
4. `outputs/mirflickr25k/clean_stats.json`
5. `outputs/mirflickr25k/cleaning_report.md`

## 17. Open Questions
Items not frozen yet:

1. whether and how to include `exif/exif_raw/license/doc` fields in manifest schema
2. whether to enforce extra id-space sanity checks beyond acceptance expected-count checks

All above remain TO_BE_CONFIRMED.

## 18. Freeze Rule
Freeze behavior for this draft:

1. do not implement cleaner code from this document yet
2. do not start split/training/feature extraction from this document
3. any item marked TO_BE_CONFIRMED must not be silently treated as finalized
4. once unresolved items are confirmed, update this document explicitly before implementation
