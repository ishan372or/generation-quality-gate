import hashlib
import json
import threading
import os


class ScoreCache:
    """Thread-safe score cache with single-flight computation."""

    def __init__(self, redis_url=None):
        self._values = {}
        self._inflight = {}
        self._lock = threading.RLock()
        self._redis = None
        redis_url = redis_url or os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._redis = client
            except Exception:
                self._redis = None

    @staticmethod
    def make_key(model_version, feature_schema_version, features, parameters=None):
        payload = {
            "model_version": model_version,
            "feature_schema_version": feature_schema_version,
            "features": features,
            "parameters": parameters or {},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key):
        if self._redis is not None:
            value = self._redis.get(key)
            if value is not None:
                return json.loads(value)
        with self._lock:
            return self._values.get(key)

    def set(self, key, value):
        if self._redis is not None:
            self._redis.set(key, json.dumps(value))
        with self._lock:
            self._values[key] = value

    def get_or_compute(self, key, compute):
        with self._lock:
            cached = self.get(key)
            if cached is not None:
                return cached
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False

        if not owner:
            event.wait()
            with self._lock:
                value = self._values.get(key)
                error = getattr(event, "error", None)
            if error is not None:
                raise error
            return value

        try:
            value = compute()
            with self._lock:
                self.set(key, value)
            return value
        except Exception as error:
            event.error = error
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()
