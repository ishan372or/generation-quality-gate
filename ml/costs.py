import numpy as np
import yaml

DECISIONS = ("KEEP","REGENERATE","REVIEW")

def load_config(config_path: str = "config/train.yaml")-> dict:
    """Load decision configuration from YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def compute_expected_cost(probability_bad: float,action: str,costs: dict) -> float:
    """
    Compute the expected cost of taking an action.
    probability_bad:
        Probability that the shot should NOT be kept.
    Actions:
        KEEP
            Risk paying the false-keep cost.
        REGENERATE
            Risk paying the false-regenerate cost.
        REVIEW
            Always incurs the review cost.
    """
    probability_bad = float(
        np.clip(
            probability_bad,
            0.0,
            1.0,
        )
    )
    probability_good = 1.0 - probability_bad
    if action == "KEEP":
        return (
            probability_bad
            * costs["false_keep"]
            + probability_good
            * costs["correct_keep"]
        )
    if action == "REGENERATE":
        return (
            probability_good
            * costs["false_regenerate"]
            + probability_bad
            * costs["correct_regenerate"]
        )
    if action == "REVIEW":
        return costs["review"]

    raise ValueError(
        f"Unknown decision: {action}"
    )

def choose_decision(
    probability_bad: float,
    costs: dict,
) -> tuple[str, dict[str, float]]:
    """
    Choose the action with the lowest expected cost.

    Returns
    -------
    decision:
        KEEP, REGENERATE, or REVIEW.

    expected_costs:
        Expected cost for every possible action.
    """

    expected_costs = {
        action: compute_expected_cost(
            probability_bad=probability_bad,
            action=action,
            costs=costs,
        )
        for action in DECISIONS
    }

    decision = min(
        expected_costs,
        key=expected_costs.get
    )
    return decision, expected_costs

def combine_quality_scores(
    identity_probability: float,
    defect_probability: float,
    adherence_probability: float,
) -> dict[str, float]:
    """
    Convert the three scorer probabilities into a common
    quality representation.
    identity_probability:
        Probability that identity matches.
    defect_probability:
        Probability that the shot contains a defect.
    adherence_probability:
        Probability that the shot adheres to the reference.
    The returned values represent the probability that each
    quality dimension is good.
    """
    return {
        "identity_good": float(
            np.clip(
                identity_probability,
                0.0,
                1.0,
            )
        ),
        "defect_good": float(
            np.clip(
                1.0 - defect_probability,
                0.0,
                1.0,
            )
        ),
        "adherence_good": float(
            np.clip(
                adherence_probability,
                0.0,
                1.0,
            )
        ),
    }
def compute_overall_bad_probability(
    identity_probability: float,
    defect_probability: float,
    adherence_probability: float,
) -> float:
    """
    Combine the three quality scorers into one probability
    representing the likelihood that a shot should not be kept.

    The worst quality dimension dominates the decision.

    This is conservative: a shot with a severe identity failure
    or defect should not be considered good just because its
    other scores are high.
    """
    quality_scores = combine_quality_scores(
        identity_probability=identity_probability,
        defect_probability=defect_probability,
        adherence_probability=adherence_probability,
    )

    overall_quality = min(quality_scores.values())
    return float(1.0 - overall_quality)

def score_shot_decision(
    identity_probability: float,
    defect_probability: float,
    adherence_probability: float,
    config_path: str = "config/train.yaml",
) -> dict:
    config = load_config(
        config_path
    )
    costs = config["decision"]["costs"]
    quality_scores = combine_quality_scores(
        identity_probability=identity_probability,
        defect_probability=defect_probability,
        adherence_probability=adherence_probability,
    )
    overall_bad_probability = (
        compute_overall_bad_probability(
            identity_probability=identity_probability,
            defect_probability=defect_probability,
            adherence_probability=adherence_probability,
        )
    )
    decision, expected_costs = choose_decision(
        probability_bad=overall_bad_probability,
        costs=costs,
    )
    return {
        "decision": decision,
        "overall_bad_probability": overall_bad_probability,
        "quality_scores": quality_scores,
        "expected_costs": expected_costs,
    }

def evaluate_decision_policy(
    probabilities: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    config_path: str = "config/train.yaml",
) -> dict:
    config = load_config(config_path)

    costs = config["decision"]["costs"]

    identity_probabilities = probabilities["identity"]
    defect_probabilities = probabilities["defect"]
    adherence_probabilities = probabilities["adherence"]

    identity_labels = labels["identity"]
    defect_labels = labels["defect"]
    adherence_labels = labels["adherence"]

    n_samples = len(identity_probabilities)

    if not (
        len(defect_probabilities) == n_samples
        and len(adherence_probabilities) == n_samples
        and len(identity_labels) == n_samples
        and len(defect_labels) == n_samples
        and len(adherence_labels) == n_samples
    ):
        raise ValueError(
            "All probability and label arrays must have "
            "the same length."
        )

    decision_counts = {
        action: 0
        for action in DECISIONS
    }

    total_cost = 0.0

    for index in range(n_samples):

        decision_result = score_shot_decision(
            identity_probability=identity_probabilities[index],
            defect_probability=defect_probabilities[index],
            adherence_probability=adherence_probabilities[index],
            config_path=config_path,
        )

        decision = decision_result["decision"]

        decision_counts[decision] += 1
        shot_is_good = (
            identity_labels[index] == 1
            and defect_labels[index] == 0
            and adherence_labels[index] == 1
        )

        if decision == "KEEP":
            if not shot_is_good:
                total_cost += costs["false_keep"]

            else:
                total_cost += costs["correct_keep"]

        elif decision == "REGENERATE":
            if shot_is_good:
                total_cost += costs["false_regenerate"]

            else:
                total_cost += costs["correct_regenerate"]

        elif decision == "REVIEW":
            total_cost += costs["review"]

    expected_cost_per_1000 = (
        total_cost / n_samples
    ) * 1000

    return {
        "expected_cost_per_1000": float(
            expected_cost_per_1000
        ),
        "total_cost": float(
            total_cost
        ),
        "decision_counts": decision_counts,
        "n_samples": n_samples,
    }