from pathlib import Path
import joblib
import pandas as pd
import yaml

from ml.calibration import apply_calibrators
from ml.costs import evaluate_decision_policy
from ml.features import build_feature_pipeline
from ml.metrics import evaluate_all_scorers


def load_config(config_path: str = "config/train.yaml")-> dict:
    """Load evaluation configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_models(model_dir: str = "artifacts/models")-> dict:
    """
    Load the three trained scorer models.
    """
    model_path = Path(model_dir)
    models = {}
    for scorer_name in ("identity","defect","adherence"):
        path = (model_path/f"{scorer_name}.joblib")

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        models[scorer_name] = joblib.load(
            path
        )

    return models


def load_calibrators(calibrator_dir: str = "artifacts/models")-> dict:
    """
    Load the calibration models for all scorers.
    """
    calibrator_path = Path(calibrator_dir)
    calibrators = {}
    for scorer_name in (
        "identity",
        "defect",
        "adherence",
    ):
        path = (
            calibrator_path
            / f"{scorer_name}_calibrator.joblib"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Calibrator not found: {path}"
            )

        calibrators[scorer_name] = joblib.load(
            path
        )

    return calibrators

def evaluate_models(
    test_df: pd.DataFrame,
    models: dict,
    calibrators: dict,
    config_path: str = "config/train.yaml",
) -> dict:
    """
    Evaluate all scorers on the test dataset.

    Test data is used only for final evaluation.
    Calibration has already been fitted using validation data.
    """
    features, labels = build_feature_pipeline(
        test_df
    )
    calibrated_probabilities = (
        apply_calibrators(
            models=models,
            calibrators=calibrators,
            features=features,
        )
    )
    config = load_config(
        config_path
    )
    evaluation_config = config[
        "evaluation"
    ]

    metrics = evaluate_all_scorers(
        probabilities=calibrated_probabilities,
        labels=labels,
        target_precision=evaluation_config[
            "target_precision"
        ],
        bootstrap_samples=evaluation_config[
            "bootstrap_samples"
        ],
        seed=config["seed"],
    )

    decision_metrics = (
        evaluate_decision_policy(
            probabilities=calibrated_probabilities,
            labels=labels,
            config_path=config_path,
        )
    )

    return {
        "scorer_metrics": metrics,
        "decision_metrics": decision_metrics,
    }

def format_results(
    results: dict,
) -> str:
    """
    Convert evaluation results into a readable text report.
    """
    lines = []
    lines.append("Generation Quality Gate Evaluation")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Scorer Metrics")
    lines.append("-" * 40)
    for scorer_name, metrics in results["scorer_metrics"].items():
        lines.append(f"\n{scorer_name.upper()}")
        lines.append(f"PR-AUC: "f"{metrics['pr_auc']:.4f}")
        pr_auc_ci = metrics[
            "pr_auc_ci"
        ]
        lines.append(
            f"PR-AUC 95% CI: "
            f"({pr_auc_ci[0]:.4f}, "
            f"{pr_auc_ci[1]:.4f})"
        )
        lines.append(
            f"Recall @ target precision: "
            f"{metrics['recall_at_precision']:.4f}"
        )
        recall_ci = metrics[
            "recall_ci"
        ]
        lines.append(
            f"Recall 95% CI: "
            f"({recall_ci[0]:.4f}, "
            f"{recall_ci[1]:.4f})"
        )
        lines.append(
            f"ECE: "
            f"{metrics['ece']:.4f}"
        )
    decision_metrics = results[
        "decision_metrics"
    ]
    lines.append("")
    lines.append(
        "Decision Policy"
    )
    lines.append(
        "-" * 40
    )
    lines.append(
        f"Expected cost / 1000 shots: "
        f"{decision_metrics['expected_cost_per_1000']:.2f}"
    )
    lines.append(
        f"Total cost: "
        f"{decision_metrics['total_cost']:.2f}"
    )
    lines.append(
        f"Samples: "
        f"{decision_metrics['n_samples']}"
    )
    lines.append(
        "Decision counts:"
    )
    for decision, count in (
        decision_metrics[
            "decision_counts"
        ].items()
    ):
        lines.append(
            f"  {decision}: {count}"
        )

    return "\n".join(lines)


def save_results(
    results: dict,
    output_path: str = "artifacts/evaluation.txt",
) -> None:
    """
    Save the evaluation report.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True,exist_ok=True)
    report = format_results(results)
    output_file.write_text(
        report,
        encoding="utf-8",
    )