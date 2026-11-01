from pathlib import Path
import joblib
import numpy as np
import yaml
from xgboost import XGBClassifier


def load_config(config_path: str = "config/train.yaml") -> dict:
    """Load training configuration from YAML."""

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_model(model_config: dict,seed: int)-> XGBClassifier:
    return XGBClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        learning_rate=model_config["learning_rate"],
        random_state=seed,
        eval_metric="logloss",
    )


def train_scorer(X: np.ndarray,y: np.ndarray,model_config: dict,seed: int)-> XGBClassifier:
    model =build_model(model_config=model_config,seed=seed,)
    model.fit(X, y)

    return model


def train_all_scorers(train_features: dict[str, np.ndarray],train_labels: dict[str, np.ndarray],config_path: str = "config/train.yaml",
) -> dict[str, XGBClassifier]:
    """
    Train all three quality scorers.
    Returns:
        {
            "identity": trained identity model,
            "defect": trained defect model,
            "adherence": trained adherence model,
        }
    """

    config = load_config(config_path)
    seed = config["seed"]
    model_configs = config["models"]
    models = {}
    models["identity"] = train_scorer(
        X=train_features["identity"],
        y=train_labels["identity"],
        model_config=model_configs["identity"],
        seed=seed,
    )
    models["defect"] = train_scorer(
        X=train_features["defect"],
        y=train_labels["defect"],
        model_config=model_configs["defect"],
        seed=seed,
    )
    models["adherence"] = train_scorer(
        X=train_features["adherence"],
        y=train_labels["adherence"],
        model_config=model_configs["adherence"],
        seed=seed,
    )

    return models

def save_models(models: dict[str, XGBClassifier],output_dir: str = "artifacts/models",) -> None:
    """
    Save trained scorers to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True,)
    for name, model in models.items():
        model_path = output_path / f"{name}.joblib"
        joblib.dump(model,model_path)
