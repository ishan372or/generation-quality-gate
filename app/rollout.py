import hashlib

class RolloutManager:
    STAGES=("production","shadow","canary","promoted","rollback")
    def __init__(self,registry):
        self.registry=registry
        self.current_version=None
        self.candidate_version=None
        self.stage="production"
        self.canary_percentage=0.0

    def set_production(self,version):
        self.registry.get(version)
        self.current_version=version
        self.stage="production"
        self.canary_percentage=0.0

    def set_shadow(self,version):
        self.registry.get(version)
        self.candidate_version=version
        self.stage="shadow"
        self.canary_percentage=0.0

    def start_canary(self,percentage):
        if self.candidate_version is None:
            raise RuntimeError("No candidate model available.")

        if not 0.0 < percentage <= 100.0:
            raise ValueError("Canary percentage must be between 0 and 100.")

        self.stage="canary"
        self.canary_percentage=percentage

    def should_use_candidate(self,project_id):
        if self.stage=="shadow":
            return False

        if self.stage=="canary":
            value=int(hashlib.sha256(project_id.encode("utf-8")).hexdigest(),16)
            bucket=value%100
            return bucket<self.canary_percentage

        if self.stage=="promoted":
            return True

        return False

    def promote(self):
        if self.candidate_version is None:
            raise RuntimeError("No candidate model available.")

        self.current_version=self.candidate_version
        self.candidate_version=None
        self.stage="promoted"
        self.canary_percentage=100.0

    def rollback(self):
        if self.current_version is None:
            raise RuntimeError("No production model available.")

        self.candidate_version=None
        self.stage="rollback"
        self.canary_percentage=0.0