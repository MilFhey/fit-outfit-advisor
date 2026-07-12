from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_RANKING_TOP_KS = (1, 3, 5, 10)


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
    observations = build_positive_pair_observations(rows, config=config)
    return build_cooccurrence_from_observations(observations)


def build_cooccurrence_from_observations(
    observations: list[dict[str, Any]],
) -> tuple[Counter[tuple[str, str]], Counter[str], Counter[tuple[str, str]], set[tuple[str, str]]]:
    directed_counts: Counter[tuple[str, str]] = Counter()
    input_counts: Counter[str] = Counter()
    role_pair_counts: Counter[tuple[str, str]] = Counter()
    exact_positive_pair_keys: set[tuple[str, str]] = set()
    counted_role_pair_occurrences: set[tuple[str, tuple[str, str], tuple[str, str]]] = set()

    for observation in observations:
        input_product = observation["input_product_type"]
        candidate_product = observation["candidate_product_type"]
        directed_counts[(input_product, candidate_product)] += 1
        input_counts[input_product] += 1
        role_pair_occurrence = (
            observation["outfit_id"],
            observation["pair_key"],
            observation["role_pair_key"],
        )
        if role_pair_occurrence not in counted_role_pair_occurrences:
            role_pair_counts[observation["role_pair_key"]] += 1
            counted_role_pair_occurrences.add(role_pair_occurrence)
        exact_positive_pair_keys.add(observation["pair_key"])

    return directed_counts, input_counts, role_pair_counts, exact_positive_pair_keys


def build_positive_pair_observations(
    rows: list[dict[str, str]],
    *,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    compatible_roles = build_compatible_role_pair_set(config)
    by_outfit: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_outfit[row["outfit_id"]].append(row)

    observations: list[dict[str, Any]] = []

    for outfit_rows in by_outfit.values():
        for left, right in combinations(outfit_rows, 2):
            role_key = compatible_role_pair_key(left["outfit_role"], right["outfit_role"])
            if role_key not in compatible_roles:
                continue
            pair_key = exact_item_pair_key(left["item_id"], right["item_id"])
            for input_item, candidate_item in ((left, right), (right, left)):
                observations.append(
                    {
                        "pair_key": pair_key,
                        "outfit_id": input_item["outfit_id"],
                        "role_pair_key": role_key,
                        "input_product_type": input_item["product_type_v0"],
                        "candidate_product_type": candidate_item["product_type_v0"],
                    }
                )

    return observations


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


def split_config_name(relative_split: str) -> str:
    return relative_split.split("/", maxsplit=1)[0]


def split_short_name(relative_split: str) -> str:
    return Path(relative_split).stem


def aggregate_payload(
    directed_counts: Counter[tuple[str, str]],
    input_counts: Counter[str],
    role_pair_counts: Counter[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "directed_pair_count": sum(directed_counts.values()),
        "unique_directed_product_pair_count": len(directed_counts),
        "role_pair_counts": {
            "|".join(key): count for key, count in role_pair_counts.most_common()
        },
        "recommendations_by_product_type": recommendations_from_counts(
            directed_counts,
            input_counts,
        ),
    }


def _empty_ranking_metrics() -> dict[str, Any]:
    return {
        "raw_directed_pair_count": 0,
        "filtered_train_overlap_directed_pair_count": 0,
        "evaluable_directed_pair_count": 0,
        "ranking_hit_count_by_k": {str(k): 0 for k in DEFAULT_RANKING_TOP_KS},
        "precision_at_k": {str(k): None for k in DEFAULT_RANKING_TOP_KS},
        "recall_at_k": {str(k): None for k in DEFAULT_RANKING_TOP_KS},
        "ndcg_at_k": {str(k): None for k in DEFAULT_RANKING_TOP_KS},
        "mrr": None,
    }


def evaluate_filtered_ranking(
    observations: list[dict[str, Any]],
    *,
    train_pair_keys: set[tuple[str, str]],
    recommendations_by_product_type: dict[str, list[dict[str, Any]]],
    top_ks: tuple[int, ...] = DEFAULT_RANKING_TOP_KS,
) -> dict[str, Any]:
    rankings = {
        input_product: {
            row["product_type_v0"]: rank
            for rank, row in enumerate(recommendations, start=1)
        }
        for input_product, recommendations in recommendations_by_product_type.items()
    }
    filtered = [
        observation
        for observation in observations
        if observation["pair_key"] not in train_pair_keys
    ]
    if not observations:
        return _empty_ranking_metrics()

    hit_counts = {k: 0 for k in top_ks}
    ndcg_sums = {k: 0.0 for k in top_ks}
    reciprocal_rank_sum = 0.0
    found_rank_count = 0

    for observation in filtered:
        candidate_rank = rankings.get(observation["input_product_type"], {}).get(
            observation["candidate_product_type"]
        )
        if candidate_rank is None:
            continue
        found_rank_count += 1
        reciprocal_rank_sum += 1 / candidate_rank
        for k in top_ks:
            if candidate_rank <= k:
                hit_counts[k] += 1
                ndcg_sums[k] += 1 / math.log2(candidate_rank + 1)

    evaluable_count = len(filtered)
    if evaluable_count == 0:
        metrics = _empty_ranking_metrics()
        metrics["raw_directed_pair_count"] = len(observations)
        metrics["filtered_train_overlap_directed_pair_count"] = len(observations)
        return metrics

    return {
        "raw_directed_pair_count": len(observations),
        "filtered_train_overlap_directed_pair_count": len(observations) - evaluable_count,
        "evaluable_directed_pair_count": evaluable_count,
        "ranking_found_directed_pair_count": found_rank_count,
        "ranking_hit_count_by_k": {str(k): hit_counts[k] for k in top_ks},
        "precision_at_k": {
            str(k): round(hit_counts[k] / (evaluable_count * k), 6)
            for k in top_ks
        },
        "recall_at_k": {
            str(k): round(hit_counts[k] / evaluable_count, 6)
            for k in top_ks
        },
        "ndcg_at_k": {
            str(k): round(ndcg_sums[k] / evaluable_count, 6)
            for k in top_ks
        },
        "mrr": round(reciprocal_rank_sum / evaluable_count, 6),
    }


def train_eval_positive_pair_leakage(
    config_split_pair_keys: dict[str, set[tuple[str, str]]],
) -> dict[str, Any]:
    train_keys = config_split_pair_keys.get("train", set())
    overlaps: dict[str, int] = {}
    for split_name, pair_keys in config_split_pair_keys.items():
        if split_name == "train":
            continue
        overlaps[f"train__{split_name}"] = len(train_keys.intersection(pair_keys))
    return {
        "exact_positive_pair_overlap_counts": overlaps,
        "has_train_eval_positive_pair_leakage": any(
            count > 0 for count in overlaps.values()
        ),
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
    split_baselines: dict[str, Any] = {}
    split_pair_keys: dict[str, set[tuple[str, str]]] = {}
    split_pair_keys_by_config: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(dict)
    split_observations_by_config: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    aggregate_directed_counts: Counter[tuple[str, str]] = Counter()
    aggregate_input_counts: Counter[str] = Counter()
    aggregate_role_pair_counts: Counter[tuple[str, str]] = Counter()
    config_directed_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    config_input_counts: dict[str, Counter[str]] = defaultdict(Counter)
    config_role_pair_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    for relative_split in SPLIT_FILES:
        config_name = split_config_name(relative_split)
        short_name = split_short_name(relative_split)
        split_name = relative_split.replace("/", "_").replace(".json", "")
        rows, diagnostics = load_mapped_split_items(
            resolved_raw_root / relative_split,
            metadata=metadata,
            categories_lookup=categories_lookup,
            config=config,
        )
        observations = build_positive_pair_observations(rows, config=config)
        directed_counts, input_counts, role_pair_counts, pair_keys = build_cooccurrence_from_observations(
            observations
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
        split_baselines[split_name] = aggregate_payload(
            directed_counts,
            input_counts,
            role_pair_counts,
        )
        split_pair_keys[split_name] = pair_keys
        split_pair_keys_by_config[config_name][short_name] = pair_keys
        split_observations_by_config[config_name][short_name] = observations
        aggregate_directed_counts.update(directed_counts)
        aggregate_input_counts.update(input_counts)
        aggregate_role_pair_counts.update(role_pair_counts)
        config_directed_counts[config_name].update(directed_counts)
        config_input_counts[config_name].update(input_counts)
        config_role_pair_counts[config_name].update(role_pair_counts)

    leakage_by_config = {
        config_name: positive_pair_leakage(config_split_pair_keys)
        for config_name, config_split_pair_keys in split_pair_keys_by_config.items()
    }
    has_within_config_leakage = any(
        payload["has_exact_positive_pair_leakage"]
        for payload in leakage_by_config.values()
    )
    aggregate_by_config = {
        config_name: aggregate_payload(
            config_directed_counts[config_name],
            config_input_counts[config_name],
            config_role_pair_counts[config_name],
        )
        for config_name in sorted(config_directed_counts)
    }
    primary_config = "disjoint" if "disjoint" in aggregate_by_config else next(iter(aggregate_by_config))
    primary_training_split = f"{primary_config}_train"
    if primary_training_split not in split_baselines:
        primary_training_split = next(iter(split_baselines))
    primary_train_eval_leakage = train_eval_positive_pair_leakage(
        split_pair_keys_by_config.get(primary_config, {})
    )
    primary_train_pair_keys = split_pair_keys_by_config.get(primary_config, {}).get("train", set())
    primary_recommendations = split_baselines[primary_training_split][
        "recommendations_by_product_type"
    ]
    leakage_filtered_evaluation = {
        split_name: evaluate_filtered_ranking(
            observations,
            train_pair_keys=primary_train_pair_keys,
            recommendations_by_product_type=primary_recommendations,
        )
        for split_name, observations in split_observations_by_config.get(primary_config, {}).items()
        if split_name != "train"
    }
    leakage_filtered_evaluation_ready = any(
        payload["evaluable_directed_pair_count"] > 0
        for payload in leakage_filtered_evaluation.values()
    )
    cross_config_diagnostic = positive_pair_leakage(split_pair_keys)
    leakage = {
        "by_config": leakage_by_config,
        "has_within_config_positive_pair_leakage": has_within_config_leakage,
        "primary_train_eval": primary_train_eval_leakage,
        "has_primary_train_eval_positive_pair_leakage": primary_train_eval_leakage[
            "has_train_eval_positive_pair_leakage"
        ],
        "cross_config_diagnostic": {
            **cross_config_diagnostic,
            "decision_note": (
                "Cross-config overlaps are diagnostic only because disjoint and "
                "nondisjoint are alternative dataset configurations."
            ),
        },
    }
    report.update(
        {
            "baseline_ready": True,
            "reason": "cooccurrence_baseline_built_from_raw_metadata",
            "split_diagnostics": split_diagnostics,
            "split_recommendations": split_recommendations,
            "split_baselines": split_baselines,
            "leakage": leakage,
            "primary_config": primary_config,
            "primary_training_split": primary_training_split,
            "primary_baseline": split_baselines[primary_training_split],
            "evaluation_ready_without_leakage": not primary_train_eval_leakage[
                "has_train_eval_positive_pair_leakage"
            ],
            "leakage_filtered_evaluation_ready": leakage_filtered_evaluation_ready,
            "leakage_filtered_evaluation": leakage_filtered_evaluation,
            "baseline_decision": (
                "train_only_baseline_ready_for_evaluation"
                if not primary_train_eval_leakage["has_train_eval_positive_pair_leakage"]
                else (
                    "train_only_baseline_ready_with_leakage_filtered_evaluation"
                    if leakage_filtered_evaluation_ready
                    else "train_only_baseline_built_evaluation_requires_leakage_filter"
                )
            ),
            "aggregate_by_config": aggregate_by_config,
            "aggregate": aggregate_payload(
                aggregate_directed_counts,
                aggregate_input_counts,
                aggregate_role_pair_counts,
            ),
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
