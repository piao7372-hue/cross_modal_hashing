# NUS-WIDE 清洗验收清单（v1）

## 文档作用域说明（2026-04 状态同步）
- 本文档是 NUS-WIDE 数据清洗子阶段的验收记录。
- 本文档属于历史子阶段文档，不代表当前全项目阶段。
- 当前实现状态以仓库代码与配置为准，并结合 `docs/project_status.md` 与 `docs/semantic_cache_contract.md` 统一判断。

## 1) 本次清洗输入文件
- `data/raw/nuswide/images/*.jpg`
- `data/raw/nuswide/NUS_WID_Tags/All_Tags.txt`
- `data/raw/nuswide/NUS_WID_Tags/AllTags81.txt`
- `data/raw/nuswide/NUS_WID_Tags/AllTags1k.txt`
- `data/raw/nuswide/Groundtruth/AllLabels/Labels_*.txt`
- `data/raw/nuswide/ConceptsList/Concepts81.txt`

## 2) 采用的 Canonical Sources
- `All_Tags.txt`（主对齐键 `image_id` + 原始文本 token）
- `AllTags81.txt`（81 维 tag 向量）
- `AllTags1k.txt`（1000 维 tag 向量）
- `Groundtruth/AllLabels/Labels_*.txt`（81 维标签向量）

明确未作为 canonical 对齐源：
- `Groundtruth/TrainTestLabels/*`
- `legacy_subset/*`

## 3) 删除策略
- 非重复 `image_id`：若无法唯一回连图像则删除（`drop_reason=missing_image` 或歧义原因）。
- 重复 `image_id`：
  - 若重复行 tag 与 81 维 label 都相同：保留，按“文件名升序 + metadata 行序升序”一一匹配。
  - 若任一不同：整组删除，`drop_reason=ambiguous_duplicate_id`。
- 预确认强制删除 id：
  - `1100787682`
  - `2728487708`
  - `702409954`

## 4) 最终样本数
- `expected_total_rows = 269648`
- `actual_total_rows = 269648`
- `expected_clean_rows_after_preconfirmed_ambiguous_drop = 269642`
- `actual_clean_rows = 269642`
- `dropped_records = 6`（全部为 `ambiguous_duplicate_id`）

## 5) 已知风险
- 安全重复组当前采用确定性规则完成匹配，但并非来自官方显式映射表。
- 当前输出 `clean_manifest.jsonl` 较大（全量约数百 MB），后续迭代需关注 I/O 成本。
- `TrainTestLabels` 中已确认存在源异常（`Labels_lake_Train.txt`），因此未参与 canonical 对齐。

## 6) 建议人工检查项（输出文件与字段）
重点检查以下 4 个输出文件：
- `data/processed/nuswide/clean_manifest.jsonl`
- `data/processed/nuswide/dropped_samples.jsonl`
- `outputs/nuswide/clean_stats.json`
- `outputs/nuswide/cleaning_report.md`

建议人工抽查字段：
- `clean_manifest.jsonl`：
  - `id`, `row_index`, `image_id`, `image_path`
  - `text_raw_tokens`, `text`
  - `tags_1k`（长度 1000）、`labels_81`（长度 81）
- `dropped_samples.jsonl`：
  - `image_id`, `candidate_image_paths`, `drop_reason`, `detail`, `row_index`
- `clean_stats.json`：
  - `actual_total_rows`, `actual_clean_rows`
  - `safe_duplicate_ids`, `ambiguous_duplicate_ids`
  - `drop_reason_breakdown`
- `cleaning_report.md`：
  - canonical source 列表
  - disabled source 列表
  - `Labels_lake_Train` anomaly 排除声明
