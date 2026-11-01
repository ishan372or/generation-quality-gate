import numpy as np
import pandas as pd

def _to_array(value) -> np.ndarray:
    """Convert a stored feature list into a NumPy array."""
    return np.asarray(value,dtype=np.float32)

def calculate_psi(reference: np.ndarray,current: np.ndarray,n_bins: int = 10) -> float:
    reference = np.asarray(reference,dtype=np.float64)
    current = np.asarray(current,dtype=np.float64)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError(
            "Reference and current data cannot be empty."
        )
    bin_edges = np.quantile(reference,
        np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        )
    )
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    reference_counts, _ = np.histogram(reference,bins=bin_edges)

    current_counts, _ = np.histogram(current,bins=bin_edges)
    reference_proportions = (reference_counts/len(reference))
    current_proportions = (current_counts/len(current))
    epsilon = 1e-6
    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )
    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )
    psi = np.sum((current_proportions- reference_proportions)* np.log(current_proportions/ reference_proportions))
    return float(psi)


def calculate_feature_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, float]:
    results = {}
    feature_groups = {
        "defect": "defect_features",
        "adherence": "adherence_features",
    }
    for group_name, column_name in feature_groups.items():
        reference_features = np.stack(
            reference_df[column_name].map(_to_array)
        )
        current_features = np.stack(
            current_df[column_name].map(_to_array)
        )
        n_features = reference_features.shape[1]
        feature_psis = []
        for feature_index in range(n_features):
            psi = calculate_psi(
                reference=reference_features[:, feature_index],
                current=current_features[:, feature_index]
            )
            feature_psis.append(psi)
        results[group_name] = float(np.mean(feature_psis))
    return results

def calculate_embedding_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column_name: str,
) -> float:
    reference_embeddings = np.stack(
        reference_df[column_name].map(_to_array)
    )

    current_embeddings = np.stack(
        current_df[column_name].map(_to_array)
    )

    if (
        reference_embeddings.shape[1]
        != current_embeddings.shape[1]
    ):
        raise ValueError(
            "Reference and current embeddings "
            "must have the same dimension."
        )

    feature_psis = []

    for feature_index in range(
        reference_embeddings.shape[1]
    ):

        psi = calculate_psi(
            reference=reference_embeddings[
                :, feature_index
            ],
            current=current_embeddings[
                :, feature_index
            ],
        )

        feature_psis.append(psi)

    return float(
        np.mean(feature_psis)
    )
    
def classify_drift(
    psi: float,
) -> str:
    """
    Convert PSI into a simple drift status.
    """
    if psi < 0.10:
        return "LOW"
    if psi < 0.25:
        return "MODERATE"
    return "HIGH"

def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    """
    Run drift detection between reference and current data.
    """
    defect_psi, adherence_psi = (
        calculate_feature_psi(
            reference_df=reference_df,
            current_df=current_df,
        ).values()
    )
    frame_embedding_psi = calculate_embedding_psi(
        reference_df=reference_df,
        current_df=current_df,
        column_name="frame_embedding",
    )
    reference_embedding_psi = calculate_embedding_psi(
        reference_df=reference_df,
        current_df=current_df,
        column_name="reference_embedding",
    )
    results = {
        "defect_features": {
            "psi": defect_psi,
            "status": classify_drift(defect_psi)
        },
        "adherence_features": {
            "psi": adherence_psi,
            "status": classify_drift(adherence_psi)
        },
        "frame_embedding": {
            "psi": frame_embedding_psi,
            "status": classify_drift(frame_embedding_psi)
        },
        "reference_embedding": {
            "psi": reference_embedding_psi,
            "status": classify_drift(reference_embedding_psi)
        }
    }

    return results