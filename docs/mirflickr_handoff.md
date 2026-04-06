# MIRFlickr Handoff

## 文档作用域说明（2026-04 状态同步）
- 本文档是 MIRFlickr 数据清洗子阶段的 handoff 记录，用于保留该子阶段的历史状态。
- 本文档不代表当前全项目阶段，不单独定义项目当前主路径。
- 当前实现状态以仓库代码与配置为准，并结合 `docs/project_status.md` 与 `docs/semantic_cache_contract.md` 统一判断。

## 1. 当前阶段
- 本文档记录时对应阶段为 data cleaning / data layer only（历史子阶段口径）。

## 2. 已完成的关键工作
- MIRFlickr raw inventory 已完成。
- canonical alignment policy 已冻结。
- acceptance checklist 已基本完成。
- dataset config 已冻结。
- MIRFlickr cleaner 已实现并接入 pipeline。
- `--dataset mirflickr25k --dry-run` 已走通。
- `--dataset mirflickr25k` 全量非 dry-run 已跑通。

## 3. 已冻结的 canonical policy
- `dataset_name=mirflickr25k`
- canonical image source = `data/raw/mirflickr25k/mirflickr/im<ID>.jpg`
- canonical text source = `data/raw/mirflickr25k/mirflickr/meta/tags_raw/tags<ID>.txt`
- canonical label source = `data/raw/mirflickr25k/mirflickr25k_annotations_v080/*.txt`（排除 `README.txt`）
- `annotation_source_version = v080`
- `potential_labels` 与 `relevant_labels_r1` 为并列独立命名空间
- empty text 保留并标记 `text_empty=true`
- `missing_text_file = drop`
- `annotation_parse_failure = hard fail`
- primary alignment key conflict = hard fail
- annotation 文件内重复行 = deduplicate and record

## 4. 已冻结的验收计数
- `actual_total_rows = 25000`
- `actual_clean_rows = 25000`
- `dropped_records = 0`
- `drop_reason_breakdown = {}`
- `empty_text_rows = 2128`
- `annotation dedup statistics inside label files = 0`

## 5. 人工验收状态（闭环已完成）
- head/middle/tail 样本的人审一致性抽检已完成。
  - 取样基准：按 `data/processed/mirflickr25k/clean_manifest.jsonl` 的当前文件顺序抽样。
  - 固定位置：第 `1 / 12500 / 25000` 条。
  - 说明：本次取样不是按 `image_id` 数值排序。
  - 抽检位置与样本：
    - line 1 -> `id=mirflickr25k:1`, `image_id=1`
    - line 12500 -> `id=mirflickr25k:21247`, `image_id=21247`
    - line 25000 -> `id=mirflickr25k:9999`, `image_id=9999`
  - 抽检结论：
    - image/text 路径对齐正常
    - `text_empty` 与实际一致
    - annotation 映射可回查且命名空间一致
    - 未发现 alignment 异常
- dropped 样本抽检：当前不适用（full run `dropped_records=0`），保留为 future-triggered / still_open 项，待未来出现 dropped 样本时补检。

## 6. 当前明确不能做的事
- 不进入 split。
- 不进入 feature extraction。
- 不进入 training / evaluation / model / loss。
- 不把论文 ra 的 `<20 labels` 过滤当作 raw cleaning 规则。
- 不启动 MSCOCO。

## 7. 下一个线程接手后应该先做什么
- 先读取 `docs/mirflickr_handoff.md`。
- 然后只补人工抽检闭环，不改清洗逻辑。
