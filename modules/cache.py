import json
from pathlib import Path
import hashlib

class Cache:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, key_data) -> str:
        if isinstance(key_data, str):
            return hashlib.md5(key_data.encode()).hexdigest()
        elif isinstance(key_data, (dict, list)):
            return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        else:
            raise TypeError("Unsupported key type for caching")

    def get(self, key_data):
        cache_key = self._get_cache_key(key_data)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
        return None

    def set(self, key_data, data):
        cache_key = self._get_cache_key(key_data)
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
