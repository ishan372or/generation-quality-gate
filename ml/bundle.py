from pathlib import Path
import hashlib
import json
import shutil

import joblib

from ml.schema import FEATURE_SCHEMA_VERSION

SCORERS = ("identity", "defect", "adherence")

def calculate_sha256(path: str) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def create_bundle(
    model_dir: str = "artifacts/models",
    output_dir: str = "artifacts/bundle",
    model_version: str = "v1",
    dataset_hash: str = "unknown",
) -> dict:
    model_path = Path(model_dir)
    bundle_path = Path(output_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)
    files = {}
    for scorer in SCORERS:
        model_file = model_path / f"{scorer}.joblib"
        calibrator_file = model_path / f"{scorer}_calibrator.joblib"

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        if not calibrator_file.exists():
            raise FileNotFoundError(f"Calibrator not found: {calibrator_file}")

        shutil.copy2(model_file, bundle_path / model_file.name)
        shutil.copy2(calibrator_file, bundle_path / calibrator_file.name)

        files[model_file.name] = calculate_sha256(model_file)
        files[calibrator_file.name] = calculate_sha256(calibrator_file)

    metadata = {
        "model_version": model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "scorers": list(SCORERS),
        "files": files,
    }

    metadata_path = bundle_path / "metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    metadata["metadata_sha256"] = calculate_sha256(metadata_path)

    return metadata

def validate_bundle(bundle_dir: str) -> dict:
    bundle_path = Path(bundle_dir)
    metadata_path = bundle_path / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path) as file:
        metadata = json.load(file)

    if metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema version mismatch.")

    for filename, expected_hash in metadata["files"].items():
        file_path = bundle_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Bundle file not found: {file_path}")

        actual_hash = calculate_sha256(file_path)

        if actual_hash != expected_hash:
            raise ValueError(f"SHA256 mismatch for {filename}")

    return metadata

def load_bundle(bundle_dir: str) -> dict:
    metadata = validate_bundle(bundle_dir)
    bundle_path = Path(bundle_dir)

    models = {}
    calibrators = {}

    for scorer in SCORERS:
        models[scorer] = joblib.load(
            bundle_path / f"{scorer}.joblib"
        )
        calibrators[scorer] = joblib.load(
            bundle_path / f"{scorer}_calibrator.joblib"
        )

    return {
        "model_version": metadata["model_version"],
        "feature_schema_version": metadata["feature_schema_version"],
        "models": models,
        "calibrators": calibrators,
    }
