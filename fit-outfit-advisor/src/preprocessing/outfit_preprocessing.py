from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from src.mappings.polyvore_mapping import (
    build_compatible_role_pair_set,
    compatible_role_pair_key,
)


OUTFIT_PAIR_FEATURE_COLUMNS = [
    "input_product_type",
    "input_canonical_category",
    "input_outfit_role",
    "candidate_product_type",
    "candidate_canonical_category",
    "candidate_outfit_role",
]


def exact_item_pair_key(item_id_a: Any, item_id_b: Any) -> tuple[str, str]:
    return tuple(sorted((str(item_id_a), str(item_id_b))))


def build_positive_outfit_pairs(items: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required_columns = {
        "outfit_id",
        "item_id",
        "product_type_v0",
        "canonical_category",
        "outfit_role",
    }
    missing = required_columns.difference(items.columns)
    if missing:
        raise ValueError(f"Colonnes outfit absentes: {sorted(missing)}")

    compatible_pairs = build_compatible_role_pair_set(config)
    rows: list[dict[str, Any]] = []
    for outfit_id, outfit_items in items.groupby("outfit_id"):
        for left, right in combinations(outfit_items.to_dict("records"), 2):
            role_key = compatible_role_pair_key(left["outfit_role"], right["outfit_role"])
            if role_key not in compatible_pairs:
                continue
            for input_item, candidate_item in ((left, right), (right, left)):
                rows.append(
                    {
                        "outfit_id": outfit_id,
                        "input_item_id": input_item["item_id"],
                        "candidate_item_id": candidate_item["item_id"],
                        "input_product_type": input_item["product_type_v0"],
                        "input_canonical_category": input_item["canonical_category"],
                        "input_outfit_role": input_item["outfit_role"],
                        "candidate_product_type": candidate_item["product_type_v0"],
                        "candidate_canonical_category": candidate_item["canonical_category"],
                        "candidate_outfit_role": candidate_item["outfit_role"],
                        "pair_key": exact_item_pair_key(input_item["item_id"], candidate_item["item_id"]),
                        "label": 1,
                    }
                )
    return pd.DataFrame(rows)


def generate_hard_negative_pairs(
    positive_pairs: pd.DataFrame,
    items: pd.DataFrame,
    *,
    positive_pair_keys: set[tuple[str, str]] | None = None,
    ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate hard negatives by replacing candidates with same-role items."""
    if positive_pairs.empty:
        return positive_pairs.copy()

    rng = __import__("random").Random(seed)
    positive_pair_keys = positive_pair_keys or set(positive_pairs["pair_key"])
    target_count = int(round(len(positive_pairs) * ratio))
    rows: list[dict[str, Any]] = []
    item_records = items.to_dict("records")

    for positive in positive_pairs.sample(frac=1.0, random_state=seed).to_dict("records"):
        if len(rows) >= target_count:
            break
        candidates = [
            item
            for item in item_records
            if item["outfit_role"] == positive["candidate_outfit_role"]
            and item["outfit_id"] != positive["outfit_id"]
            and str(item["item_id"]) != str(positive["input_item_id"])
        ]
        same_family = [
            item
            for item in candidates
            if item["canonical_category"] == positive["candidate_canonical_category"]
        ]
        candidates = same_family or candidates
        rng.shuffle(candidates)
        for candidate in candidates:
            pair_key = exact_item_pair_key(positive["input_item_id"], candidate["item_id"])
            if pair_key in positive_pair_keys:
                continue
            rows.append(
                {
                    "outfit_id": positive["outfit_id"],
                    "input_item_id": positive["input_item_id"],
                    "candidate_item_id": candidate["item_id"],
                    "input_product_type": positive["input_product_type"],
                    "input_canonical_category": positive["input_canonical_category"],
                    "input_outfit_role": positive["input_outfit_role"],
                    "candidate_product_type": candidate["product_type_v0"],
                    "candidate_canonical_category": candidate["canonical_category"],
                    "candidate_outfit_role": candidate["outfit_role"],
                    "pair_key": pair_key,
                    "label": 0,
                }
            )
            break

    return pd.DataFrame(rows)


def build_outfit_feature_frame(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = set(OUTFIT_PAIR_FEATURE_COLUMNS + ["label"]).difference(pairs.columns)
    if missing:
        raise ValueError(f"Colonnes de paires absentes: {sorted(missing)}")
    features = pairs[OUTFIT_PAIR_FEATURE_COLUMNS].copy()
    target = pairs["label"].astype(int).copy()
    return features, target


def split_outfits_by_id(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    outfit_ids = pd.Series(frame["outfit_id"].dropna().unique())
    train_validation_ids, test_ids = train_test_split(
        outfit_ids,
        test_size=test_size,
        random_state=seed,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train_ids, validation_ids = train_test_split(
        train_validation_ids,
        test_size=relative_validation_size,
        random_state=seed,
    )
    train = frame[frame["outfit_id"].isin(set(train_ids))].copy()
    validation = frame[frame["outfit_id"].isin(set(validation_ids))].copy()
    test = frame[frame["outfit_id"].isin(set(test_ids))].copy()
    return train, validation, test


def assert_no_positive_pair_leakage(split_pairs: dict[str, pd.DataFrame]) -> None:
    positive_keys = {
        split_name: set(frame.loc[frame["label"] == 1, "pair_key"])
        for split_name, frame in split_pairs.items()
    }
    split_names = list(positive_keys)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            overlap = positive_keys[left_name].intersection(positive_keys[right_name])
            if overlap:
                raise ValueError(
                    f"Paires positives exactes en fuite entre {left_name} et {right_name}: "
                    f"{sorted(overlap)[:5]}"
                )
