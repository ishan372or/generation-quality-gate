from dataclasses import dataclass, field

feature_schema_version="fs1"
FEATURE_SCHEMA_VERSION = feature_schema_version

embedding_dim=64

DEFECT_FEATURES = [
    "blur",
    "noise",
    "artifact",
    "face_distortion",
    "extra_limb",
    "missing_limb",
    "bad_hands",
    "bad_text",
    "flicker",
    "compression",
    "lighting_error",
    "background_error",
]

ADHERENCE_AXES = [
    "subject",
    "emotion",
    "optics",
    "motion",
    "lighting",
    "style",
    "audio",
    "continuity",
]

@dataclass
class FeatureSchema:
    version: str = feature_schema_version
    embedding_dim: int = embedding_dim
    
    defect_features: list[str]= field(default_factory=lambda: DEFECT_FEATURES)
    adherence_axes: list[str]= field(default_factory=lambda: ADHERENCE_AXES)
    
    def validate(self):
        if self.version != feature_schema_version:
            raise ValueError(f"Unsupported feature schema version: {self.version}")
        if self.embedding_dim != embedding_dim:
            raise ValueError(f"Unsupported embedding dimension: {self.embedding_dim}")
        if len(self.defect_features) != len(DEFECT_FEATURES):
            raise ValueError(f"Defect features list length mismatch: {len(self.defect_features)} vs {len(DEFECT_FEATURES)}")
        if len(self.adherence_axes) != len(ADHERENCE_AXES):
            raise ValueError(f"Adherence axes list length mismatch: {len(self.adherence_axes)} vs {len(ADHERENCE_AXES)}")
        
FEATURE_SCHEMA = FeatureSchema()
FEATURE_SCHEMA.validate()
