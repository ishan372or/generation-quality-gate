from pathlib import Path

import pandas as pd

from ml.datagen import generate_dataset
from ml.features import build_feature_pipeline
from ml.train import save_models, train_all_scorers

REQUIRED_FEEDBACK_COLUMNS = [
    "project_id",
    "character_id",
    "shot_id",
    "feedback",
]

def validate_feedback(feedback_df: pd.DataFrame) -> None:
    """Validate the feedback dataset."""
    missing_columns = [
        column for column in REQUIRED_FEEDBACK_COLUMNS
        if column not in feedback_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feedback columns: {missing_columns}")

    valid_feedback = {
        "accept",
        "retry",
        "review_good",
        "review_bad",
    }

    invalid_values = set(feedback_df["feedback"].dropna()) - valid_feedback

    if invalid_values:
        raise ValueError(f"Invalid feedback values: {invalid_values}")


def convert_feedback_to_labels(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """Convert user/reviewer feedback into weak labels."""
    validate_feedback(feedback_df)
    df = feedback_df.copy()
    df["weak_quality_label"] = (
        df["feedback"].isin(["accept", "review_good"]).astype(int)
    )
    df["is_gold_label"] = (
        df["feedback"].isin(["review_good", "review_bad"]).astype(int)
    )
    return df

def merge_feedback_with_dataset(
    dataset_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match feedback with the corresponding generated shots."""

    feedback_df = convert_feedback_to_labels(feedback_df)

    return dataset_df.merge(
        feedback_df[
            [
                "project_id",
                "character_id",
                "shot_id",
                "weak_quality_label",
                "is_gold_label",
            ]
        ],
        on=["project_id", "character_id", "shot_id"],
        how="inner",
    )

def create_retraining_dataset(
    dataset_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a retraining dataset using feedback labels."""

    merged_df = merge_feedback_with_dataset(dataset_df, feedback_df)

    if merged_df.empty:
        raise ValueError("No matching feedback records were found.")

    merged_df["retraining_label"] = merged_df["weak_quality_label"]

    gold_mask = merged_df["is_gold_label"] == 1

    merged_df.loc[gold_mask, "retraining_label"] = (
        merged_df.loc[gold_mask, "feedback"] == "review_good"
    ).astype(int)

    return merged_df

def retrain_candidate(
    retraining_df: pd.DataFrame,
    output_dir: str = "artifacts/models/candidate",
    train_config_path: str = "config/train.yaml",
) -> dict:
    """Train and save a candidate model without replacing production."""

    if retraining_df.empty:
        raise ValueError("Cannot retrain using an empty dataset.")

    features, labels = build_feature_pipeline(retraining_df)
    models = train_all_scorers(
        train_features=features,
        train_labels=labels,
        config_path=train_config_path,
    )
    save_models(
        models=models,
        output_dir=output_dir,
    )
    return models

def prepare_retraining_data(
    feedback_path: str,
    data_config_path: str = "config/data.yaml",
) -> pd.DataFrame:
    """Load feedback and combine it with the reproducible dataset."""

    feedback_file = Path(feedback_path)

    if not feedback_file.exists():
        raise FileNotFoundError(f"Feedback file not found: {feedback_file}")

    feedback_df = pd.read_csv(feedback_file)
    dataset_df = generate_dataset(config_path=data_config_path)

    return create_retraining_dataset(
        dataset_df=dataset_df,
        feedback_df=feedback_df,
    )
