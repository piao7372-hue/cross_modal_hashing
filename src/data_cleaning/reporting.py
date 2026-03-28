from __future__ import annotations

from .records import CleaningStats


def build_cleaning_report_markdown(
    stats: CleaningStats,
    canonical_sources: dict[str, str],
    disabled_sources: list[str],
    notes: list[str],
) -> str:
    title_name = "NUS-WIDE" if stats.dataset_name.lower() == "nuswide" else stats.dataset_name.upper()
    lines = [
        f"# {title_name} Cleaning Report",
        "",
        "## Summary",
        f"- expected_total_rows: {stats.expected_total_rows}",
        f"- actual_total_rows: {stats.actual_total_rows}",
        "- expected_clean_rows_after_preconfirmed_ambiguous_drop: "
        f"{stats.expected_clean_rows_after_preconfirmed_ambiguous_drop}",
        f"- actual_clean_rows: {stats.actual_clean_rows}",
        f"- clean_records_written: {stats.clean_records_written}",
        f"- dropped_records_written: {stats.dropped_records_written}",
        "",
        "## Duplicate Handling",
        f"- safe_duplicate_resolution_policy: {stats.safe_duplicate_resolution_policy}",
        f"- safe_duplicate_ids: {stats.safe_duplicate_ids}",
        f"- ambiguous_duplicate_ids: {stats.ambiguous_duplicate_ids}",
        f"- ambiguous_duplicate_drop_count: {stats.ambiguous_duplicate_drop_count}",
        "",
        "## Quality Counters",
        f"- missing_image_count: {stats.missing_image_count}",
        f"- parse_failure_count: {stats.parse_failure_count}",
        f"- drop_reason_breakdown: {stats.drop_reason_breakdown}",
        "",
        "## Canonical Sources Used",
    ]
    for key, value in canonical_sources.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Disabled Sources (Not Used for Canonical Alignment)",
        ]
    )
    for value in disabled_sources:
        lines.append(f"- `{value}`")
    lines.extend(["", "## Notes"])
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"
