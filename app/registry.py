from pathlib import Path
import json
import shutil
from ml.bundle import validate_bundle, load_bundle
from app.db import SessionLocal
from app.models import ModelVersion
import hashlib
import json

class ModelRegistry:
    def __init__(self, registry_dir="artifacts/registry", pin_file="artifacts/model_pin.json"):
        self.registry_dir=Path(registry_dir)
        self.pin_file=Path(pin_file)
        self.registry_dir.mkdir(parents=True,exist_ok=True)

    def register(self,bundle_dir):
        bundle_dir=Path(bundle_dir)
        metadata=validate_bundle(bundle_dir)
        version=metadata["model_version"]
        target_dir=self.registry_dir/version

        if not target_dir.exists():
            shutil.copytree(bundle_dir,target_dir)
        self._register_metadata(metadata, target_dir)
        return metadata

    @staticmethod
    def _bundle_sha256(bundle_dir):
        digest = hashlib.sha256()
        for path in sorted(Path(bundle_dir).glob("*.joblib")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _register_metadata(self, metadata, bundle_dir):
        sha256 = self._bundle_sha256(bundle_dir)
        db = SessionLocal()
        try:
            record = db.query(ModelVersion).filter_by(version=metadata["model_version"]).one_or_none()
            if record is not None:
                if record.sha256 != sha256 or record.feature_schema_version != metadata["feature_schema_version"]:
                    raise ValueError("Registered model digest or feature schema does not match bundle.")
                return
            record = ModelVersion(
                name="generation-quality-gate",
                version=metadata["model_version"],
                status="available",
                sha256=sha256,
                feature_schema_version=metadata["feature_schema_version"],
                embedding_version="embedding-v1",
                dataset_hash=metadata.get("dataset_hash", "unknown"),
                bundle_path=str(bundle_dir),
                metrics_json=json.dumps(metadata.get("metrics", {})),
            )
            db.add(record)
            db.commit()
        finally:
            db.close()

    def register_or_validate(self, version="v1", bundle_dir="artifacts/bundle"):
        metadata = validate_bundle(bundle_dir)
        if metadata["model_version"] != version:
            raise ValueError(f"Bundle version {metadata['model_version']} does not match {version}.")
        target = self.registry_dir / version
        if not target.exists():
            return self.register(bundle_dir)
        self._register_metadata(metadata, target)
        return metadata

    def get(self,version):
        bundle_dir=self.registry_dir/version

        if not bundle_dir.exists():
            raise FileNotFoundError(f"Model version not found: {version}")

        return load_bundle(bundle_dir)

    def pin(self,version):
        model=self.get(version)
        self.pin_file.parent.mkdir(parents=True,exist_ok=True)
        self.pin_file.write_text(json.dumps({"model_version":version},indent=2),encoding="utf-8")

        return model

    def get_pinned(self):
        if not self.pin_file.exists():
            raise RuntimeError("No model version is currently pinned.")

        data=json.loads(self.pin_file.read_text(encoding="utf-8"))
        version=data["model_version"]

        return self.get(version)
