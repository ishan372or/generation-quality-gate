import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)

def compute_pr_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Compute Precision-Recall AUC.

    PR-AUC is useful when the positive class is relatively rare,
    such as defective shots in our dataset.
    """
    return float(average_precision_score(labels,probabilities))

def compute_recall_at_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_precision: float = 0.90,
) -> float:
    """
    Compute the highest recall achievable while maintaining
    at least the requested precision.
    """

    precision, recall, _ = precision_recall_curve(
        labels,
        probabilities,
    )

    valid_recall = recall[
        precision >= target_precision
    ]

    if len(valid_recall) == 0:
        return 0.0

    return float(
        np.max(valid_recall)
    )


def compute_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how closely predicted probabilities
    match the actual fraction of positive examples.

    Example:

        predictions around 0.8
        actual positive rate around 0.8

    means the model is well calibrated.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    bin_edges = np.linspace(0.0,1.0,n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )
        if not np.any(mask):
            continue
        bin_probabilities = probabilities[mask]
        bin_labels = labels[mask]
        mean_probability = np.mean(
            bin_probabilities
        )
        actual_positive_rate = np.mean(
            bin_labels
        )
        bin_error = abs(mean_probability- actual_positive_rate)
        bin_weight = (np.sum(mask)/ len(probabilities))
        ece += (bin_weight* bin_error)
    return float(ece)


def bootstrap_confidence_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric_function,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Estimate a confidence interval for a metric using bootstrap
    resampling.

    The same metric function can be used for PR-AUC,
    recall at precision, or another evaluation metric.
    """

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    if len(labels) != len(probabilities):
        raise ValueError(
            "Labels and probabilities must have the same length."
        )

    if len(labels) == 0:
        raise ValueError(
            "Cannot bootstrap an empty dataset."
        )

    rng = np.random.default_rng(seed)

    scores = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            len(labels),
            size=len(labels),
        )

        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[
            indices
        ]

        try:
            score = metric_function(
                sampled_labels,
                sampled_probabilities,
            )

            scores.append(score)

        except ValueError:
            continue

    if len(scores) == 0:
        raise ValueError(
            "Could not compute bootstrap confidence interval."
        )

    alpha = 1.0 - confidence

    lower_percentile = (
        100 * (alpha / 2)
    )

    upper_percentile = (
        100 * (1 - alpha / 2)
    )

    lower = np.percentile(
        scores,
        lower_percentile,
    )

    upper = np.percentile(
        scores,
        upper_percentile,
    )

    return float(lower), float(upper)


def evaluate_scorer(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_precision: float = 0.90,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict:
    """
    Evaluate one scorer using the required metrics.
    """

    pr_auc = compute_pr_auc(
        labels,
        probabilities,
    )

    recall_at_precision = compute_recall_at_precision(
        labels,
        probabilities,
        target_precision=target_precision,
    )

    ece = compute_ece(
        labels,
        probabilities,
    )

    pr_auc_ci = bootstrap_confidence_interval(
        labels=labels,
        probabilities=probabilities,
        metric_function=compute_pr_auc,
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )

    recall_ci = bootstrap_confidence_interval(
        labels=labels,
        probabilities=probabilities,
        metric_function=lambda y, p: compute_recall_at_precision(
            y,
            p,
            target_precision=target_precision,
        ),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )

    return {
        "pr_auc": pr_auc,
        "pr_auc_ci": pr_auc_ci,
        "recall_at_precision": recall_at_precision,
        "recall_ci": recall_ci,
        "ece": ece,
    }


def evaluate_all_scorers(
    probabilities: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    target_precision: float = 0.90,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Evaluate identity, defect, and adherence scorers.
    """

    results = {}

    for scorer_name in probabilities:

        if scorer_name not in labels:
            raise ValueError(
                f"Missing labels for scorer: {scorer_name}"
            )

        results[scorer_name] = evaluate_scorer(
            labels=labels[scorer_name],
            probabilities=probabilities[scorer_name],
            target_precision=target_precision,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )

    return results