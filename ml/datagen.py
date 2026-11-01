import numpy as np
import pandas as pd
import yaml

from ml.schema import FEATURE_SCHEMA

def load_config(config_path: str) -> dict:
    """Load dataset configuration from YAML file"""
    with open(config_path,'r') as file:
        return yaml.safe_load(file)

def generate_data(config_path: str = "config/data.yaml", shifted: bool = False)->pd.DataFrame:
    FEATURE_SCHEMA.validate()
    config= load_config(config_path)
    
    seed=config["seed"]
    dataset_config=config["dataset"]
    
    n_projects = dataset_config["n_projects"]
    characters_per_project = dataset_config["characters_per_project"]
    shots_per_character = dataset_config["shots_per_character"]
    defect_rate = dataset_config["defect_rate"]
    label_noise_rate = dataset_config["label_noise_rate"]
    hard_case_rate = dataset_config.get("hard_case_rate", dataset_config.get("hard_cases_rate"))
    lighting_shift_rate = dataset_config["lighting_shift_rate"]
    lookalike_rate = dataset_config["lookalike_rate"]
    identity_mismatch_rate = dataset_config["identity_mismatch_rate"]
    
    if shifted:
        shifted_config = config["shifted_dataset"]
        
        if not shifted_config["enabled"]:
            raise ValueError("Shifted dataset generation is disabled in data.yaml")
        
        defect_rate = shifted_config["defect_rate"]
        lighting_shift_rate = shifted_config["lighting_shift_rate"]
        
    rng=np.random.default_rng(seed)
    rows=[]
    
    for project_id in range(n_projects):
        character_references={}
        for character_id in range(characters_per_project):
            is_lookalike=(character_id>0 and rng.random()<lookalike_rate)
            if is_lookalike:
                similar_character_id=rng.integers(0,character_id)
                base_embedding=character_references[similar_character_id]
                reference_embedding=base_embedding+rng.normal(0,0.2,size=FEATURE_SCHEMA.embedding_dim)
            else:
                reference_embedding=rng.normal(0,1,size=FEATURE_SCHEMA.embedding_dim)
            character_references[character_id] = reference_embedding
                
        for character_id in range(characters_per_project):
            true_reference_embedding=character_references[character_id]
            for shot_id in range(shots_per_character):
                is_hard_case=(rng.random()<hard_case_rate)
                has_lighting_shift=(rng.random()<lighting_shift_rate)
                has_defect=(rng.random()<defect_rate)
                identity_match=1
                reference_embedding=true_reference_embedding
                is_identity_mismatch = (rng.random() < identity_mismatch_rate)
                if is_identity_mismatch:
                    other_character_ids = [candidate for candidate in range(characters_per_project)if candidate != character_id]
                    wrong_character_id = rng.choice(other_character_ids)
                    reference_embedding = character_references[
                        wrong_character_id
                    ]
                    identity_match = 0
                    
                frame_noise_scale=0.15
                if is_hard_case:
                    frame_noise_scale+=0.15
                    
                if has_lighting_shift:
                    frame_noise_scale+=0.1
                    
                frame_embedding = (
                    true_reference_embedding
                    + rng.normal(
                        loc=0.0,
                        scale=frame_noise_scale,
                        size=FEATURE_SCHEMA.embedding_dim,
                    )
                )
                
                if has_defect:
                    frame_embedding += rng.normal(
                        loc=0.0,
                        scale=0.50,
                        size=FEATURE_SCHEMA.embedding_dim,
                    )
                defect_features = rng.random(len(FEATURE_SCHEMA.defect_features))
                if has_defect:
                    defect_features = np.clip(
                        defect_features + 0.50,
                        0.0,
                        1.0,
                    )

                if is_hard_case:
                    defect_features = (
                        0.75 * defect_features
                        + 0.25 * 0.5
                    )

                if has_lighting_shift:
                    lighting_index = (
                        FEATURE_SCHEMA.defect_features.index(
                            "lighting_error"
                        )
                    )
                    defect_features[lighting_index] = np.clip(
                        defect_features[lighting_index] + 0.30,
                        0.0,
                        1.0,
                    )

                adherence_features = rng.random(
                    len(FEATURE_SCHEMA.adherence_axes)
                )
                if is_hard_case:
                    adherence_features = (
                        0.70 * adherence_features
                        + 0.30 * 0.5
                    )

                if has_lighting_shift:
                    lighting_index = (
                        FEATURE_SCHEMA.adherence_axes.index(
                            "lighting"
                        )
                    )

                    adherence_features[lighting_index] = np.clip(
                        adherence_features[lighting_index] - 0.25,
                        0.0,
                        1.0,
                    )
                adherence_good = int(
                    np.mean(adherence_features) >= 0.5
                )
                rows.append(
                    {
                        "project_id": project_id,
                        "character_id": character_id,
                        "shot_id": shot_id,
                        "frame_embedding": (
                            frame_embedding.tolist()
                        ),
                        "reference_embedding": (
                            reference_embedding.tolist()
                        ),
                        "defect_features": (
                            defect_features.tolist()
                        ),
                        "adherence_features": (
                            adherence_features.tolist()
                        ),
                        "identity_match": identity_match,
                        "has_defect": int(has_defect),
                        "adherence_good": adherence_good,
                        "is_hard_case": int(is_hard_case),
                        "is_lookalike": int(
                            any(
                                np.allclose(
                                    reference_embedding,
                                    character_references[
                                        other_id
                                    ],
                                    atol=0.5,
                                )
                                for other_id in character_references
                                if other_id != character_id
                            )
                        ),
                        "lighting_shift": int(
                            has_lighting_shift
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    
    n_noisy = int(len(df) * label_noise_rate)
    if n_noisy > 0:
        noisy_indices = rng.choice(df.index,size=n_noisy,replace=False)
        df.loc[noisy_indices, "has_defect"] = (1- df.loc[noisy_indices, "has_defect"])

    return df


generate_dataset = generate_data

