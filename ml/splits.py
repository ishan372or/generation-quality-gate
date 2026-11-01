import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import GroupShuffleSplit

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


def load_config(config_path: str) -> dict:
    """Load dataset configuration from YAML."""

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def split_dataset(
    df: pd.DataFrame,
    config_path: str = "config/data.yaml",
):
    config = load_config(config_path)

    split_config = config["splits"]

    train_ratio = split_config["train"]
    validation_ratio = split_config["validation"]
    test_ratio = split_config["test"]
    total_ratio = (train_ratio+validation_ratio+test_ratio)

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, "
            f"got {total_ratio}"
        )

    if len(df) == 0:
        raise ValueError("Cannot split an empty dataset.")

    if "project_id" not in df.columns:
        raise ValueError(
            "Dataset must contain 'project_id' for grouped splitting."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=(validation_ratio + test_ratio),
        random_state=config["seed"],
    )

    train_indices, temp_indices = next(
        splitter.split(
            df,
            groups=df["project_id"],
        )
    )

    train_df = df.iloc[train_indices].reset_index(drop=True)
    temp_df = df.iloc[temp_indices].reset_index(drop=True)

    validation_fraction = (validation_ratio/(validation_ratio + test_ratio))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=(1.0 - validation_fraction),
        random_state=config["seed"],
    )

    validation_indices, test_indices = next(
        splitter.split(
            temp_df,
            groups=temp_df["project_id"],
        )
    )

    validation_df = temp_df.iloc[validation_indices].reset_index(drop=True)

    test_df = temp_df.iloc[test_indices].reset_index(drop=True)

    train_projects = set(train_df["project_id"])
    validation_projects = set(validation_df["project_id"])
    test_projects = set(test_df["project_id"])

    if train_projects & validation_projects:
        raise ValueError(
            "Project leakage detected between train and validation."
        )

    if train_projects & test_projects:
        raise ValueError(
            "Project leakage detected between train and test."
        )

    if validation_projects & test_projects:
        raise ValueError(
            "Project leakage detected between validation and test."
        )

    return train_df, validation_df, test_df


