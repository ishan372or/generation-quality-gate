from app.registry import ModelRegistry

class ModelLoader:
    def __init__(self,registry=None):
        self.registry=registry or ModelRegistry()
        self._models={}

    def load(self,version):
        if version not in self._models:
            self._models[version]=self.registry.get(version)
        return self._models[version]

    def clear(self):
        self._models.clear()