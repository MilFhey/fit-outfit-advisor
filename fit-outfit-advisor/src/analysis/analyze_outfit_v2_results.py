from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config.paths import OUTFIT_V1_DIR, OUTFIT_V2_DIR, REPORTS_DIR


OUTFIT_V2_ANALYSIS_PATH = REPORTS_DIR / "outfit_v2_results_analysis.json"
COOCCURRENCE_BASELINE_PATH = REPORTS_DIR / "polyvore_v0_cooccurrence_baseline.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _metric(payload: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def build_outfit_v2_analysis(
    *,
    baseline_path: Path = COOCCURRENCE_BASELINE_PATH,
    outfit_v1_dir: Path = OUTFIT_V1_DIR,
    outfit_v2_dir: Path = OUTFIT_V2_DIR,
) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    v1_metrics = _read_json(outfit_v1_dir / "metrics.json") or _read_json(REPORTS_DIR / "outfit_v1_training_analysis.json")
    v2_metrics = _read_json(outfit_v2_dir / "metrics.json")
    v2_metadata = _read_json(outfit_v2_dir / "metadata.json")

    baseline_test = (baseline or {}).get("leakage_filtered_evaluation", {}).get("test", {})
    v2_test = (v2_metrics or {}).get("test_metrics", {})
    v2_ranking = v2_test.get("ranking", {}) if isinstance(v2_test, dict) else {}
    baseline_recall_at_3 = _metric({"root": baseline_test}, "root", "recall_at_k", "3", default=0.0)
    v2_recall_at_3 = _metric({"root": v2_ranking}, "root", "recall_at_k", "3", default=0.0)
    v2_roc_auc = _metric({"root": v2_test}, "root", "roc_auc", default=0.0)
    v2_macro_f1 = _metric({"root": v2_test}, "root", "macro_f1", default=0.0)

    promotion_checks = {
        "roc_auc_at_least_0_60": bool((v2_roc_auc or 0.0) >= 0.60),
        "recall_at_3_beats_or_matches_baseline": bool((v2_recall_at_3 or 0.0) >= (baseline_recall_at_3 or 1.0)),
        "uses_image_embeddings": bool((v2_metadata or {}).get("uses_image_embeddings") is True),
        "uses_color_features": bool((v2_metadata or {}).get("uses_color_features") is True),
        "no_direct_id_features": "item_id" in (v2_metadata or {}).get("forbidden_direct_features", []),
    }
    promotable = all(promotion_checks.values())
    decision = "promote_manually_to_models_outfit_active" if promotable else "keep_experimental_only"

    return {
        "version": "outfit_v2_analysis",
        "decision": decision,
        "promotion_checks": promotion_checks,
        "comparison": {
            "cooccurrence_v0": {
                "mrr_test": _metric({"root": baseline_test}, "root", "mrr"),
                "recall_at_3_test": baseline_recall_at_3,
                "status": (baseline or {}).get("model_status"),
            },
            "outfit_v1_tensorflow_mlp": {
                "roc_auc_test": _metric(v1_metrics, "test_metrics", "tensorflow_mlp", "roc_auc"),
                "macro_f1_test": _metric(v1_metrics, "test_metrics", "tensorflow_mlp", "macro_f1"),
                "status": (v1_metrics or {}).get("model_status", "experimental_only"),
            },
            "outfit_v2_multimodal": {
                "roc_auc_test": v2_roc_auc,
                "macro_f1_test": v2_macro_f1,
                "mrr_test": _metric({"root": v2_ranking}, "root", "mrr"),
                "recall_at_3_test": v2_recall_at_3,
                "status": (v2_metadata or {}).get("model_status", "missing"),
            },
        },
        "interpretation": (
            "Outfit V2 est promotable seulement si le modele multimodal bat clairement la baseline "
            "sur les metriques de ranking tout en conservant un ROC AUC exploitable. Sinon, le MVP "
            "garde V2 comme experience TensorFlow avancee et sert la baseline fail-closed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Outfit V2 metrics and promotion readiness.")
    parser.add_argument("--baseline-path", type=Path, default=COOCCURRENCE_BASELINE_PATH)
    parser.add_argument("--outfit-v1-dir", type=Path, default=OUTFIT_V1_DIR)
    parser.add_argument("--outfit-v2-dir", type=Path, default=OUTFIT_V2_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTFIT_V2_ANALYSIS_PATH)
    args = parser.parse_args()

    report = build_outfit_v2_analysis(
        baseline_path=args.baseline_path,
        outfit_v1_dir=args.outfit_v1_dir,
        outfit_v2_dir=args.outfit_v2_dir,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "output_path": str(args.output_path)}, indent=2))


if __name__ == "__main__":
    main()
