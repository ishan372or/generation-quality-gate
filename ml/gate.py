import yaml


def load_config(config_path: str = "config/train.yaml")-> dict:
    """Load promotion-gate configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def check_scorer_gate(metrics: dict,min_pr_auc: float,min_recall: float)-> dict:
    pr_auc_passed = (metrics["pr_auc"] >= min_pr_auc)
    recall_passed = (metrics["recall_at_precision"] >= min_recall)
    return {
        "pr_auc_passed": pr_auc_passed,
        "recall_passed": recall_passed,
        "passed": (
            pr_auc_passed
            and recall_passed
        ),
    }

def check_slice_regression(
    candidate_metrics: dict,
    baseline_metrics: dict,
    max_regression: float,
) -> dict:
    candidate_pr_auc = candidate_metrics["pr_auc"]
    baseline_pr_auc = baseline_metrics["pr_auc"]

    candidate_recall = (
        candidate_metrics["recall_at_precision"]
    )
    baseline_recall = (
        baseline_metrics["recall_at_precision"]
    )

    pr_auc_regression = (baseline_pr_auc- candidate_pr_auc)

    recall_regression = (baseline_recall- candidate_recall)
    passed = (pr_auc_regression <= max_regression and recall_regression <= max_regression)
    return {
        "pr_auc_regression": float(
            pr_auc_regression
        ),
        "recall_regression": float(
            recall_regression
        ),
        "passed": passed,
    }

def evaluate_promotion_gate(
    candidate_metrics: dict,
    baseline_metrics: dict | None = None,
    config_path: str = "config/train.yaml",
) -> dict:
    config = load_config(
        config_path
    )
    gate_config = config["promotion_gate"]
    min_pr_auc = gate_config["min_pr_auc"]
    min_recall = gate_config["min_recall"]
    max_slice_regression = gate_config["max_slice_regression"]
    scorer_results = {}

    all_scorers_passed = True
    for scorer_name, metrics in candidate_metrics.items():
        result = check_scorer_gate(
            metrics=metrics,
            min_pr_auc=min_pr_auc,
            min_recall=min_recall,
        )

        scorer_results[scorer_name] = result
        if not result["passed"]:
            all_scorers_passed = False
    regression_results = {}
    if baseline_metrics is not None:
        for scorer_name in candidate_metrics:
            if scorer_name not in baseline_metrics:
                raise ValueError(
                    f"Missing baseline metrics for "
                    f"scorer: {scorer_name}"
                )
            regression_results[scorer_name] = (
                check_slice_regression(candidate_metrics=candidate_metrics[scorer_name],
                    baseline_metrics=baseline_metrics[scorer_name],
                    max_regression=max_slice_regression
                )
            )
        regression_passed = all(
            result["passed"]
            for result in regression_results.values()
        )
    else:
        regression_passed = True
    promoted = (
        all_scorers_passed
        and regression_passed
    )
    return {
        "promoted": promoted,
        "scorers": scorer_results,
        "regression": regression_results,
        "thresholds": {
            "min_pr_auc": min_pr_auc,
            "min_recall": min_recall,
            "max_slice_regression": max_slice_regression,
        },
    }