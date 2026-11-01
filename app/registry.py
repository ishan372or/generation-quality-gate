from pathlib import Path
import json
import shutil
from ml.bundle import validate_bundle, load_bundle

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

        if target_dir.exists():
            raise ValueError(f"Model version already registered: {version}")

        shutil.copytree(bundle_dir,target_dir)
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