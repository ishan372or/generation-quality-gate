import numpy as np
import pandas as pd

from ml.schema import FEATURE_SCHEMA


def _to_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)

def build_identity_features(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Build features for the identity scorer.

    The model should learn identity similarity rather than
    relying only on raw cosine similarity.

    Features:
        - cosine similarity
        - absolute embedding difference
        - element-wise product
    """
    frame_embeddings = np.stack(df["frame_embedding"].map(_to_array))
    reference_embeddings = np.stack(df["reference_embedding"].map(_to_array))
    frame_norm = np.linalg.norm(
        frame_embeddings,
        axis=1,
        keepdims=True,
    )
    reference_norm = np.linalg.norm(
        reference_embeddings,
        axis=1,
        keepdims=True,
    )
    frame_normalized = (frame_embeddings/np.clip(frame_norm, 1e-8, None))
    reference_normalized = (reference_embeddings/np.clip(reference_norm, 1e-8, None))
    cosine_similarity = np.sum(
        frame_normalized * reference_normalized,
        axis=1,
        keepdims=True,
    )
    absolute_difference = np.abs(frame_embeddings-reference_embeddings)
    elementwise_product = (frame_embeddings*reference_embeddings)
    features = np.concatenate([cosine_similarity,absolute_difference,elementwise_product],axis=1,)

    return features

def build_defect_features(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Build features for the defect scorer.

    Returns the 12 defect features defined
    by FEATURE_SCHEMA.
    """

    features = np.stack(df["defect_features"].map(_to_array))
    if features.shape[1] != len(FEATURE_SCHEMA.defect_features):
        raise ValueError(
            "Defect feature dimension does not match "
            "FEATURE_SCHEMA."
        )

    return features

def build_adherence_features(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Build features for the adherence scorer.

    Returns the 8 adherence-axis features defined
    by FEATURE_SCHEMA.
    """

    features = np.stack(df["adherence_features"].map(_to_array))

    if features.shape[1] != len(FEATURE_SCHEMA.adherence_axes):
        raise ValueError(
            "Adherence feature dimension does not match "
            "FEATURE_SCHEMA."
        )

    return features

def build_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "identity": df["identity_match"].to_numpy(
            dtype=np.int32
        ),
        "defect": df["has_defect"].to_numpy(
            dtype=np.int32
        ),
        "adherence": df["adherence_good"].to_numpy(
            dtype=np.int32
        ),
    }
    
def build_features(df:pd.DataFrame,)-> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if df.empty:
        raise ValueError(
            "Cannot build features from an empty dataset."
        )

    FEATURE_SCHEMA.validate()
    features = {
        "identity": build_identity_features(df),
        "defect": build_defect_features(df),
        "adherence": build_adherence_features(df),
    }
    labels = build_labels(df)
    n_samples = len(df)

    for name, feature_matrix in features.items():
        if feature_matrix.shape[0] != n_samples:
            raise ValueError(
                f"{name} feature matrix contains "
                f"{feature_matrix.shape[0]} samples, "
                f"expected {n_samples}."
            )
        if labels[name].shape[0] != n_samples:
            raise ValueError(
                f"{name} labels contain "
                f"{labels[name].shape[0]} samples, "
                f"expected {n_samples}."
            )

    return features, labels


build_feature_pipeline = build_features
