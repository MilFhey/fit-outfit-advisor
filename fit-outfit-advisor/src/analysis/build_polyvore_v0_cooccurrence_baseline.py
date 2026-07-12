from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from src.analysis.analyze_polyvore_v0_schema_mapping import (
    CATEGORIES_FILE,
    DEFAULT_RAW_ROOT_CANDIDATES,
    METADATA_FILE,
    SPLIT_FILES,
    extract_item_id_candidates,
    load_categories_lookup,
    load_json,
    metadata_labels,
)
from src.config.paths import OUTFIT_V1_CONFIG_PATH, PROJECT_ROOT
from src.mappings.polyvore_mapping import (
    build_compatible_role_pair_set,
    compatible_role_pair_key,
    load_outfit_v1_config,
    map_polyvore_label_to_fashion,
    validate_outfit_v1_config,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "polyvore_v0_cooccurrence_baseline.json"


def exact_item_pair_key(item_id_a: Any, item_id_b: Any) -> tuple[str, str]:
    return tuple(sorted((str(item_id_a), str(item_id_b))))


def resolve_raw_root(raw_root: Path | None) -> Path | None:
    if raw_root is not None:
        return raw_root
    env_value = os.environ.get("POLYVORE_RAW_ROOT")
    if env_value:
        return Path(env_value)
    for candidate in DEFAULT_RAW_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def map_metadata_item(
    metadata: dict[str, Any],
    *,
    categories_lookup: dict[str, str],
    config: dict[str, Any],
) -> dict[str, str] | None:
    row = {"metadata": metadata}
    field_priority = {
        "semantic_category": 0,
        "category_id_name": 1,
        "catgeories": 2,
        "title": 3,
        "url_name": 4,
    }
    labels = sorted(
        metadata_labels(row, categories_lookup),
        key=lambda item: field_priority.get(item["field"], 99),
    )
    for label_payload in labels:
        mapped = map_polyvore_label_to_fashion(label_payload["label"], config)
        if mapped is not None:
            mapped["source_field"] = label_payload["field"]
            return mapped
    return None


def load_mapped_split_items(
    split_path: Path,
    *,
    metadata: dict[str, Any],
    categories_lookup: dict[str, str],
    config: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    records = load_json(split_path)
    rows: list[dict[str, str]] = []
    diagnostics = {
        "outfit_count": len(records),
        "raw_item_count": 0,
        "linked_item_count": 0,
        "mapped_item_count": 0,
        "mapped_outfit_count": 0,
        "unmapped_label_examples": [],
    }
    unmapped_examples: list[str] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        outfit_id = str(record.get("set_id", ""))
        outfit_mapped_count = 0
        for item in record.get("items", []) or []:
            diagnostics["raw_item_count"] += 1
            candidates = extract_item_id_candidates(item, set_id=outfit_id)
            item_id = next((candidate for candidate in candidates if candidate in metadata), None)
            if item_id is None:
                continue
            diagnostics["linked_item_count"] += 1
            item_metadata = metadata[item_id]
            if not isinstance(item_metadata, dict):
                continue
            mapped = map_metadata_item(
                item_metadata,
                categories_lookup=categories_lookup,
                config=config,
            )
            if mapped is None:
                for label_payload in metadata_labels({"metadata": item_metadata}, categories_lookup):
                    if label_payload["field"] in {"semantic_category", "category_id_name", "catgeories"}:
                        unmapped_examples.append(str(label_payload["label"]))
                        break
                continue
            rows.append(
                {
                    "outfit_id": outfit_id,
                    "item_id": str(item_id),
                    "product_type_v0": mapped["product_type_v0"],
                    "canonical_category": mapped["canonical_category"],
                    "outfit_role": mapped["outfit_role"],
                    "source_field": mapped["source_field"],
                }
            )
            outfit_mapped_count += 1
        if outfit_mapped_count >= 2:
            diagnostics["mapped_outfit_count"] += 1

    diagnostics["mapped_item_count"] = len(rows)
    diagnostics["unmapped_label_examples"] = list(dict.fromkeys(unmapped_examples))[:20]
    return rows, diagnostics


def build_split_cooccurrence(
    rows: list[dict[str, str]],
    *,
    config: dict[str, Any],
) -> tuple[Counter[tuple[str, str]], Counter[str], Counter[tuple[str, str]], set[tuple[str, str]]]:
    compatible_roles = build_compatible_role_pair_set(config)
    by_outfit: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_outfit[row["outfit_id"]].append(row)

    directed_counts: Counter[tuple[str, str]] = Counter()
    input_counts: Counter[str] = Counter()
    role_pair_counts: Counter[tuple[str, str]] = Counter()
    exact_positive_pair_keys: set[tuple[str, str]] = set()

    for outfit_rows in by_outfit.values():
        for left, right in combinations(outfit_rows, 2):
            role_key = compatible_role_pair_key(left["outfit_role"], right["outfit_role"])
            if role_key not in compatible_roles:
                continue
            pair_key = exact_item_pair_key(left["item_id"], right["item_id"])
            exact_positive_pair_keys.add(pair_key)
            role_pair_counts[role_key] += 1
            for input_item, candidate_item in ((left, right), (right, left)):
                input_product = input_item["product_type_v0"]
                candidate_product = candidate_item["product_type_v0"]
                directed_counts[(input_product, candidate_product)] += 1
                input_counts[input_product] += 1

    return directed_counts, input_counts, role_pair_counts, exact_positive_pair_keys


def recommendations_from_counts(
    directed_counts: Counter[tuple[str, str]],
    input_counts: Counter[str],
    *,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (input_product, candidate_product), count in directed_counts.items():
        if input_product == candidate_product:
            continue
        grouped[input_product].append((candidate_product, count))

    recommendations: dict[str, list[dict[str, Any]]] = {}
    for input_product, candidates in grouped.items():
        total = max(input_counts[input_product], 1)
        recommendations[input_product] = [
            {
                "product_type_v0": candidate,
                "cooccurrence_count": count,
                "raw_compatibility_score": round(count / total, 6),
            }
            for candidate, count in sorted(candidates, key=lambda item: (-item[1], item[0]))[:top_k]
        ]
    return recommendations


def positive_pair_leakage(split_pair_keys: dict[str, set[tuple[str, str]]]) -> dict[str, Any]:
    split_names = list(split_pair_keys)
    overlaps: dict[str, int] = {}
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = split_pair_keys[left_name].intersection(split_pair_keys[right_name])
            overlaps[f"{left_name}__{right_name}"] = len(overlap)
    return {
        "exact_positive_pair_overlap_counts": overlaps,
        "has_exact_positive_pair_leakage": any(count > 0 for count in overlaps.values()),
    }


def build_cooccurrence_baseline(
    *,
    raw_root: Path | None = None,
    config_path: Path = OUTFIT_V1_CONFIG_PATH,
) -> dict[str, Any]:
    resolved_raw_root = resolve_raw_root(raw_root)
    config = load_outfit_v1_config(config_path)
    validate_outfit_v1_config(config, require_ready=True)

    report: dict[str, Any] = {
        "version": "polyvore_v0_cooccurrence_baseline",
        "training_executed": False,
        "tensorflow_used": False,
        "streamlit_integration_executed": False,
        "raw_root": str(resolved_raw_root) if resolved_raw_root else None,
        "config_path": str(config_path),
        "config_status": config.get("status"),
        "model_status": "experimental_only",
        "baseline_ready": False,
        "reason": "",
    }

    if resolved_raw_root is None or not resolved_raw_root.exists():
        report["reason"] = "raw_files_missing_requires_colab_or_drive_raw_root"
        return report

    metadata_path = resolved_raw_root / METADATA_FILE
    categories_path = resolved_raw_root / CATEGORIES_FILE
    missing_files = [
        str(path.relative_to(resolved_raw_root))
        for path in [metadata_path, categories_path, *(resolved_raw_root / split for split in SPLIT_FILES)]
        if not path.exists()
    ]
    if missing_files:
        report["reason"] = "required_raw_files_missing"
        report["missing_files"] = missing_files
        return report

    metadata = load_json(metadata_path)
    categories_lookup = load_categories_lookup(resolved_raw_root)

    split_diagnostics: dict[str, Any] = {}
    split_recommendations: dict[str, Any] = {}
    split_pair_keys: dict[str, set[tuple[str, str]]] = {}
    aggregate_directed_counts: Counter[tuple[str, str]] = Counter()
    aggregate_input_counts: Counter[str] = Counter()
    aggregate_role_pair_counts: Counter[tuple[str, str]] = Counter()

    for relative_split in SPLIT_FILES:
        split_name = relative_split.replace("/", "_").replace(".json", "")
        rows, diagnostics = load_mapped_split_items(
            resolved_raw_root / relative_split,
            metadata=metadata,
            categories_lookup=categories_lookup,
            config=config,
        )
        directed_counts, input_counts, role_pair_counts, pair_keys = build_split_cooccurrence(
            rows,
            config=config,
        )
        split_diagnostics[split_name] = {
            **diagnostics,
            "directed_pair_count": sum(directed_counts.values()),
            "unique_directed_product_pair_count": len(directed_counts),
            "exact_positive_item_pair_count": len(pair_keys),
            "role_pair_counts": {
                "|".join(key): count for key, count in role_pair_counts.most_common()
            },
        }
        split_recommendations[split_name] = recommendations_from_counts(
            directed_counts,
            input_counts,
        )
        split_pair_keys[split_name] = pair_keys
        aggregate_directed_counts.update(directed_counts)
        aggregate_input_counts.update(input_counts)
        aggregate_role_pair_counts.update(role_pair_counts)

    leakage = positive_pair_leakage(split_pair_keys)
    report.update(
        {
            "baseline_ready": True,
            "reason": "cooccurrence_baseline_built_from_raw_metadata",
            "split_diagnostics": split_diagnostics,
            "leakage": leakage,
            "aggregate": {
                "directed_pair_count": sum(aggregate_directed_counts.values()),
                "unique_directed_product_pair_count": len(aggregate_directed_counts),
                "role_pair_counts": {
                    "|".join(key): count
                    for key, count in aggregate_role_pair_counts.most_common()
                },
                "recommendations_by_product_type": recommendations_from_counts(
                    aggregate_directed_counts,
                    aggregate_input_counts,
                ),
            },
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Polyvore V0 product-type cooccurrence baseline."
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=OUTFIT_V1_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_cooccurrence_baseline(raw_root=args.raw_root, config_path=args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {args.output}")
    print(f"baseline_ready={report['baseline_ready']}")
    print(f"reason={report['reason']}")


if __name__ == "__main__":
    main()
