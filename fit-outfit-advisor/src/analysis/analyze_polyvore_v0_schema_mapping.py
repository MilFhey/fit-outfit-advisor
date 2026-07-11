from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config.paths import (
    FASHION_V1_CLASSES_PATH,
    PROJECT_ROOT,
)
from src.mappings.fashion_v1_mapping import (
    FASHION_PRODUCT_TYPES_V0,
    load_fashion_v1_class_config,
    map_product_type_to_canonical_category,
)


DEFAULT_RAW_ROOT_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "polyvore" / "raw_hf_files",
    PROJECT_ROOT / "data" / "raw" / "mvasil_polyvore_outfits_raw_files",
)
DEFAULT_DATASET_AUDIT_PATH = PROJECT_ROOT / "reports" / "polyvore_v0_dataset_audit.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "polyvore_v0_schema_mapping_audit.json"

SPLIT_FILES = (
    "disjoint/train.json",
    "disjoint/valid.json",
    "disjoint/test.json",
    "nondisjoint/train.json",
    "nondisjoint/valid.json",
    "nondisjoint/test.json",
)

METADATA_FILE = "polyvore_item_metadata.json"
CATEGORIES_FILE = "categories.csv"

EXCLUSION_KEYWORDS = {
    "beauty": "beauty/cosmetics outside Fashion V1 product_type_v0",
    "makeup": "beauty/cosmetics outside Fashion V1 product_type_v0",
    "lipstick": "beauty/cosmetics outside Fashion V1 product_type_v0",
    "perfume": "fragrance outside outfit roles",
    "fragrance": "fragrance outside outfit roles",
    "nail": "beauty/cosmetics outside Fashion V1 product_type_v0",
    "hair": "hair products outside outfit roles",
    "phone": "electronics outside outfit roles",
    "case": "generic case/accessory too ambiguous for Fashion V1",
    "home": "home/decor outside outfit roles",
    "decor": "home/decor outside outfit roles",
    "furniture": "home/decor outside outfit roles",
    "hat": "headwear not retained in Fashion V1.1",
    "cap": "cap/headwear excluded from Fashion V1.1",
    "scarf": "scarf has no faithful Fashion V1 product_type_v0",
    "glove": "gloves have no faithful Fashion V1 product_type_v0",
    "sock": "hosiery has no faithful Fashion V1 product_type_v0",
    "hosiery": "hosiery has no faithful Fashion V1 product_type_v0",
    "legging": "leggings have no faithful Fashion V1 product_type_v0",
    "skirt": "skirt has no faithful Fashion V1 product_type_v0",
    "boot": "boots have no faithful Fashion V1 product_type_v0",
    "bra": "underwear outside Fashion V1 product_type_v0",
    "lingerie": "underwear outside Fashion V1 product_type_v0",
    "swim": "swimwear outside Fashion V1 product_type_v0",
}

MAPPING_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("sports_shoes", "high", ("running shoe", "training shoe", "athletic shoe", "sneaker")),
    ("heels", "high", ("heel", "pump", "stiletto")),
    ("sandals", "high", ("sandal",)),
    ("flip_flops", "high", ("flip flop", "thong sandal")),
    ("dress_shoes", "medium", ("flat", "loafer", "oxford", "formal shoe")),
    ("casual_shoes", "medium", ("shoe", "footwear")),
    ("outerwear", "high", ("coat", "jacket", "blazer", "cardigan", "sweater", "sweatshirt")),
    ("jeans", "high", ("jean", "denim")),
    ("shorts", "high", ("short",)),
    ("trousers", "medium", ("trouser", "pant", "chino", "slack")),
    ("dress", "high", ("dress", "gown")),
    ("dress", "medium", ("jumpsuit", "romper")),
    ("tshirt", "high", ("t shirt", "tee", "tshirt")),
    ("shirt", "high", ("shirt", "blouse", "button down")),
    ("top", "medium", ("top", "tunic", "kurta", "camisole", "tank")),
    ("bag", "high", ("bag", "handbag", "backpack", "clutch", "purse", "tote")),
    ("watch", "high", ("watch",)),
    ("sunglasses", "high", ("sunglass", "eyewear")),
    ("wallet", "high", ("wallet",)),
    ("belt", "high", ("belt",)),
    ("jewellery", "high", ("jewelry", "jewellery", "earring", "necklace", "bracelet", "ring", "bangle", "pendant")),
)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, dict):
        values: list[str] = []
        for nested_value in value.values():
            values.extend(flatten_values(nested_value))
        return values
    if isinstance(value, list):
        values = []
        for nested_value in value:
            values.extend(flatten_values(nested_value))
        return values
    return [str(value)]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_dataset_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def summarize_dataset_audit(dataset_audit: dict[str, Any]) -> dict[str, Any]:
    hf_split_rows = []
    for row in dataset_audit.get("hf_split_reports", []):
        hf_split_rows.append(
            {
                "config": row.get("config"),
                "split": row.get("split"),
                "row_count": row.get("row_count"),
                "sample_shape": row.get("sample_shape"),
                "id_like_columns": row.get("id_like_columns", []),
                "category_like_columns": row.get("category_like_columns", []),
                "outfit_like_columns": row.get("outfit_like_columns", []),
            }
        )

    raw_key_files = {}
    for row in dataset_audit.get("raw_metadata_reports", []):
        repo_file = row.get("repo_file")
        if repo_file in {
            "categories.csv",
            "polyvore_item_metadata.json",
            "polyvore_outfit_titles.json",
            *SPLIT_FILES,
        }:
            raw_key_files[repo_file] = {
                "readable": row.get("readable"),
                "kind": row.get("kind"),
                "record_count": row.get("record_count"),
                "sample_shape": row.get("sample_shape"),
                "sample_keys": row.get("sample_keys"),
                "nested_sample_keys": row.get("nested_sample_keys"),
            }

    return {
        "loader_only_cooccurrence_possible": dataset_audit.get(
            "loader_only_cooccurrence_possible"
        ),
        "cooccurrence_baseline_possible": dataset_audit.get(
            "cooccurrence_baseline_possible"
        ),
        "dataset_exploitable_for_outfit_v0": dataset_audit.get(
            "dataset_exploitable_for_outfit_v0"
        ),
        "audit_decision": dataset_audit.get("audit_decision"),
        "audit_notes": dataset_audit.get("audit_notes", []),
        "hf_split_rows": hf_split_rows,
        "raw_key_files": raw_key_files,
    }


def load_categories_lookup(raw_root: Path) -> dict[str, str]:
    path = raw_root / CATEGORIES_FILE
    if not path.exists():
        return {}

    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            cells = [cell.strip() for cell in row if cell and cell.strip()]
            if not cells:
                continue
            category_id = next((cell for cell in cells if cell.isdigit()), None)
            label = next((cell for cell in reversed(cells) if not cell.isdigit()), None)
            if category_id and label and normalize_text(label) not in {"undefined", "nan"}:
                lookup[category_id] = label
    return lookup


def detect_product_type(label: str, fashion_config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_text(label)
    if not normalized:
        return {
            "status": "excluded",
            "reason": "empty label",
        }

    for keyword, reason in EXCLUSION_KEYWORDS.items():
        if keyword in normalized.split() or keyword in normalized:
            return {
                "status": "excluded",
                "reason": reason,
            }

    for product_type, confidence, keywords in MAPPING_RULES:
        if product_type not in FASHION_PRODUCT_TYPES_V0:
            continue
        if any(keyword in normalized for keyword in keywords):
            canonical_category = map_product_type_to_canonical_category(
                product_type,
                fashion_config,
            )
            return {
                "status": "mapped",
                "product_type_v0": product_type,
                "canonical_category": canonical_category,
                "outfit_role": canonical_category,
                "confidence": confidence,
                "matched_keywords": [keyword for keyword in keywords if keyword in normalized],
            }

    return {
        "status": "excluded",
        "reason": "no faithful Fashion V1 product_type_v0 mapping found",
    }


def extract_item_id_candidates(item: Any, set_id: Any | None = None) -> list[str]:
    candidates: list[str] = []
    if isinstance(item, (str, int)):
        candidates.append(str(item))
    elif isinstance(item, dict):
        for key in ("item_id", "itemid", "id", "product_id"):
            if item.get(key) is not None:
                candidates.append(str(item[key]))
        if set_id is not None and item.get("index") is not None:
            candidates.append(f"{set_id}_{item['index']}")
    return list(dict.fromkeys(candidates))


def inspect_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    records = load_json(path)
    first_record = records[0] if records else {}
    first_items = first_record.get("items", []) if isinstance(first_record, dict) else []
    first_item = first_items[0] if first_items else {}

    item_ids: list[str] = []
    item_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        set_id = record.get("set_id")
        for item in record.get("items", []) or []:
            item_count += 1
            item_ids.extend(extract_item_id_candidates(item, set_id=set_id))

    return {
        "exists": True,
        "record_count": len(records),
        "item_count": item_count,
        "top_level_keys": sorted(first_record.keys()) if isinstance(first_record, dict) else [],
        "first_item_keys": sorted(first_item.keys()) if isinstance(first_item, dict) else [],
        "first_item_sample": first_item,
        "unique_item_id_candidate_count": len(set(item_ids)),
        "item_id_candidate_examples": list(dict.fromkeys(item_ids))[:10],
    }


def collect_linked_metadata(
    split_paths: list[Path],
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    linked_rows: list[dict[str, Any]] = []
    linkage = {
        "split_item_count": 0,
        "candidate_id_count": 0,
        "linked_item_count": 0,
        "unlinked_candidate_examples": [],
    }
    unlinked: list[str] = []

    for path in split_paths:
        if not path.exists():
            continue
        split_name = path.parent.name + "/" + path.name
        for record in load_json(path):
            if not isinstance(record, dict):
                continue
            set_id = record.get("set_id")
            for item in record.get("items", []) or []:
                linkage["split_item_count"] += 1
                candidates = extract_item_id_candidates(item, set_id=set_id)
                linkage["candidate_id_count"] += len(candidates)
                matched_id = next((candidate for candidate in candidates if candidate in metadata), None)
                if matched_id is None:
                    unlinked.extend(candidates[:1])
                    continue
                payload = metadata[matched_id]
                if isinstance(payload, dict):
                    linked_rows.append(
                        {
                            "split": split_name,
                            "set_id": set_id,
                            "item_id": matched_id,
                            "metadata": payload,
                        }
                    )

    linkage["linked_item_count"] = len(linked_rows)
    linkage["unlinked_candidate_examples"] = list(dict.fromkeys(unlinked))[:20]
    return linked_rows, linkage


def metadata_labels(row: dict[str, Any], categories_lookup: dict[str, str]) -> list[dict[str, str]]:
    metadata = row["metadata"]
    labels: list[dict[str, str]] = []
    for field in ("semantic_category", "catgeories", "title", "url_name"):
        for value in flatten_values(metadata.get(field)):
            normalized = normalize_text(value)
            if normalized:
                labels.append({"field": field, "label": value})

    category_id = metadata.get("category_id")
    if category_id is not None:
        category_id_text = str(category_id)
        labels.append({"field": "category_id", "label": category_id_text})
        category_label = categories_lookup.get(category_id_text)
        if category_label:
            labels.append({"field": "category_id_name", "label": category_label})
    return labels


def build_distributions(
    linked_rows: list[dict[str, Any]],
    categories_lookup: dict[str, str],
    fashion_config: dict[str, Any],
    *,
    top_n: int = 50,
) -> dict[str, Any]:
    field_counters: dict[str, Counter[str]] = defaultdict(Counter)
    mapping_counter: Counter[tuple[str, str, str, str]] = Counter()
    exclusion_counter: Counter[tuple[str, str, str]] = Counter()

    for row in linked_rows:
        for label_payload in metadata_labels(row, categories_lookup):
            field = label_payload["field"]
            label = label_payload["label"]
            normalized = normalize_text(label)
            field_counters[field][normalized] += 1

            if field == "category_id":
                continue
            mapping = detect_product_type(label, fashion_config)
            if mapping["status"] == "mapped":
                mapping_counter[
                    (
                        field,
                        normalized,
                        mapping["product_type_v0"],
                        mapping["canonical_category"],
                    )
                ] += 1
            else:
                exclusion_counter[(field, normalized, mapping["reason"])] += 1

    mapping_proposal = [
        {
            "source_field": field,
            "polyvore_label": label,
            "product_type_v0": product_type,
            "canonical_category": canonical_category,
            "outfit_role": canonical_category,
            "support_count": count,
        }
        for (field, label, product_type, canonical_category), count in mapping_counter.most_common(top_n)
    ]
    exclusions = [
        {
            "source_field": field,
            "polyvore_label": label,
            "reason": reason,
            "support_count": count,
        }
        for (field, label, reason), count in exclusion_counter.most_common(top_n)
    ]

    return {
        "field_distributions": {
            field: [
                {"value": value, "count": count}
                for value, count in counter.most_common(top_n)
            ]
            for field, counter in sorted(field_counters.items())
        },
        "mapping_proposal": mapping_proposal,
        "excluded_labels": exclusions,
    }


def mapping_rules_report(fashion_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product_type, confidence, keywords in MAPPING_RULES:
        canonical_category = map_product_type_to_canonical_category(product_type, fashion_config)
        rows.append(
            {
                "keywords": list(keywords),
                "product_type_v0": product_type,
                "canonical_category": canonical_category,
                "outfit_role": canonical_category,
                "confidence": confidence,
            }
        )
    return rows


def build_schema_mapping_audit(
    *,
    raw_root: Path | None = None,
    dataset_audit_path: Path = DEFAULT_DATASET_AUDIT_PATH,
    fashion_config_path: Path = FASHION_V1_CLASSES_PATH,
) -> dict[str, Any]:
    resolved_raw_root = resolve_raw_root(raw_root)
    fashion_config = load_fashion_v1_class_config(fashion_config_path)
    dataset_audit = load_dataset_audit(dataset_audit_path)

    report: dict[str, Any] = {
        "version": "polyvore_v0_schema_mapping_audit",
        "training_executed": False,
        "streamlit_integration_executed": False,
        "raw_root": str(resolved_raw_root) if resolved_raw_root else None,
        "dataset_audit_path": str(dataset_audit_path),
        "input_dataset_audit_decision": dataset_audit.get("audit_decision"),
        "input_raw_metadata_ready_for_schema_audit": dataset_audit.get(
            "raw_metadata_ready_for_schema_audit"
        ),
        "input_dataset_audit_summary": summarize_dataset_audit(dataset_audit),
        "taxonomy_alignment": {
            "source": "fashion_v1",
            "allowed_roles": sorted(set(fashion_config.get("canonical_mapping", {}).values())),
            "allowed_product_types": list(fashion_config.get("product_type_mapping", {}).keys()),
        },
        "mapping_rules": mapping_rules_report(fashion_config),
        "excluded_keyword_policy": [
            {"keyword": keyword, "reason": reason}
            for keyword, reason in sorted(EXCLUSION_KEYWORDS.items())
        ],
        "source_files": {},
        "split_structure": {},
        "linkage": {},
        "field_distributions": {},
        "mapping_proposal": [],
        "excluded_labels": [],
        "audit_decision": "raw_files_missing_requires_colab_or_drive_raw_root",
        "audit_notes": [],
        "next_decision": (
            "Run this script with --raw-root pointing to the cached Hugging Face raw files "
            "before building the cooccurrence baseline."
        ),
    }

    if resolved_raw_root is None or not resolved_raw_root.exists():
        report["source_files"] = {
            "raw_root_exists": False,
            "required_files": [METADATA_FILE, CATEGORIES_FILE, *SPLIT_FILES],
        }
        report["audit_notes"].append(
            "Local raw Polyvore files are absent; distributions and linked mapping support "
            "cannot be computed in this workspace."
        )
        return report

    source_files = {
        relative_path: {
            "exists": (resolved_raw_root / relative_path).exists(),
            "path": str(resolved_raw_root / relative_path),
        }
        for relative_path in (METADATA_FILE, CATEGORIES_FILE, *SPLIT_FILES)
    }
    report["source_files"] = source_files

    split_paths = [resolved_raw_root / relative_path for relative_path in SPLIT_FILES]
    report["split_structure"] = {
        relative_path: inspect_split(resolved_raw_root / relative_path)
        for relative_path in SPLIT_FILES
    }

    metadata_path = resolved_raw_root / METADATA_FILE
    if not metadata_path.exists():
        report["audit_notes"].append("polyvore_item_metadata.json is absent.")
        return report

    metadata = load_json(metadata_path)
    categories_lookup = load_categories_lookup(resolved_raw_root)
    linked_rows, linkage = collect_linked_metadata(split_paths, metadata)
    report["linkage"] = linkage
    report["categories_lookup_count"] = len(categories_lookup)
    distributions = build_distributions(linked_rows, categories_lookup, fashion_config)
    report.update(distributions)

    if linked_rows:
        report["audit_decision"] = "schema_mapping_ready_for_manual_review"
        report["next_decision"] = (
            "Review mapping_proposal and excluded_labels, then promote selected labels into "
            "config/outfit_v1_config.json before the cooccurrence baseline."
        )
    else:
        report["audit_decision"] = "raw_files_readable_but_items_not_linked_to_metadata"
        report["audit_notes"].append(
            "Split item ids did not link to polyvore_item_metadata.json with the supported id heuristics."
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Polyvore raw schema and propose mapping to Fashion V1 taxonomy."
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--dataset-audit", type=Path, default=DEFAULT_DATASET_AUDIT_PATH)
    parser.add_argument("--fashion-config", type=Path, default=FASHION_V1_CLASSES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_schema_mapping_audit(
        raw_root=args.raw_root,
        dataset_audit_path=args.dataset_audit,
        fashion_config_path=args.fashion_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {args.output}")
    print(f"audit_decision={report['audit_decision']}")


if __name__ == "__main__":
    main()
