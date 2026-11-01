import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression

def load_config(config_path: str = "config/train.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def fit_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray
) -> IsotonicRegression:
    """
    Fit an isotonic calibration model.
    probabilities:
        Raw probabilities produced by the trained scorer.

    labels:
        True labels from the validation set.
    """
    calibrator = IsotonicRegression(y_min=0.0,y_max=1.0,out_of_bounds="clip",)
    calibrator.fit(probabilities,labels,)
    return calibrator

def calibrate_probability(calibrator: IsotonicRegression,probabilities: np.ndarray) -> np.ndarray:
    """
    Convert raw model probabilities into calibrated
    probabilities.
    """
    calibrated = calibrator.predict(probabilities)
    return np.asarray(calibrated,dtype=np.float32)

def fit_all_calibrators(models: dict,validation_features: dict[str, np.ndarray],validation_labels: dict[str, np.ndarray]) -> dict[str, IsotonicRegression]:
    """
    Fit one calibrator for each scorer using validation data.

    The trained models are NOT modified.
    The calibrators operate on their output probabilities.
    """
    calibrators = {}
    for name, model in models.items():
        raw_probabilities = model.predict_proba(validation_features[name])[:, 1]
        calibrators[name] = fit_calibrator(probabilities=raw_probabilities,labels=validation_labels[name])
    return calibrators

def apply_calibrators(
    models: dict,
    calibrators: dict,
    features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Generate calibrated probabilities for all scorers.
    """
    calibrated_probabilities = {}
    for name, model in models.items():
        raw_probabilities = model.predict_proba(features[name])[:, 1]
        calibrated_probabilities[name] = (calibrate_probability(calibrators[name],raw_probabilities))
    return calibrated_probabilities


