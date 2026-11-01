import numpy as np

from ml.costs import choose_decision, compute_overall_bad_probability


class ScoringService:
    def __init__(self, models, calibrators):
        self.models = models
        self.calibrators = calibrators

    def _predict(self, scorer_name, features):
        model = self.models[scorer_name]
        calibrator = self.calibrators[scorer_name]
        raw_probability = model.predict_proba(features)[0, 1]
        calibrated_probability = calibrator.predict([raw_probability])[0]

        return float(np.clip(calibrated_probability, 0.0, 1.0))

    def score(self, features, costs):
        identity_score = self._predict("identity",features["identity"])
        defect_score = self._predict("defect",features["defect"])
        adherence_score = self._predict("adherence",features["adherence"])
        probabilities = {
            "identity": identity_score,
            "defect": defect_score,
            "adherence": adherence_score,
        }
        probability_bad = compute_overall_bad_probability(
            identity_probability=identity_score,
            defect_probability=defect_score,
            adherence_probability=adherence_score,
        )

        decision, _ = choose_decision(
            probability_bad=probability_bad,
            costs=costs,
        )

        return {
            "identity_score": identity_score,
            "defect_score": defect_score,
            "adherence_score": adherence_score,
            "probability_bad": probability_bad,
            "decision": decision,
        }
