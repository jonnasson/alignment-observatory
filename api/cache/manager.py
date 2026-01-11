"""
Cache manager for trace and tensor data.

Provides LRU caching with optional disk persistence.
"""

import hashlib
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from api.config import get_settings

settings = get_settings()


class CacheManager:
    """
    LRU cache manager with optional disk persistence.

    Supports:
    - In-memory LRU cache with configurable size
    - Disk persistence for large tensors
    - Compression for storage efficiency
    """

    def __init__(
        self,
        max_memory_items: int = 100,
        cache_dir: Path | None = None,
        use_compression: bool = True,
    ) -> None:
        self._memory_cache: OrderedDict[str, Any] = OrderedDict()
        self._max_items = max_memory_items
        self._cache_dir = cache_dir or settings.trace_cache_dir
        self._use_compression = use_compression

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, *args: Any) -> str:
        """Generate a cache key from arguments."""
        key_data = pickle.dumps(args)
        return hashlib.sha256(key_data).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        """Get an item from cache."""
        # Check memory cache first
        if key in self._memory_cache:
            # Move to end (most recently used)
            self._memory_cache.move_to_end(key)
            return self._memory_cache[key]

        # Check disk cache
        disk_path = self._cache_dir / f"{key}.npz"
        if disk_path.exists():
            data = self._load_from_disk(disk_path)
            # Add to memory cache
            self._add_to_memory(key, data)
            return data

        return None

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """Set an item in cache."""
        self._add_to_memory(key, value)

        if persist:
            self._save_to_disk(key, value)

    def _add_to_memory(self, key: str, value: Any) -> None:
        """Add item to memory cache with LRU eviction."""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        else:
            if len(self._memory_cache) >= self._max_items:
                # Evict oldest item
                self._memory_cache.popitem(last=False)
            self._memory_cache[key] = value

    def _save_to_disk(self, key: str, value: Any) -> None:
        """Save item to disk."""
        disk_path = self._cache_dir / f"{key}.npz"

        if isinstance(value, dict):
            # Handle dict with numpy arrays
            np_arrays = {}
            other_data = {}

            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    np_arrays[k] = v
                else:
                    other_data[k] = v

            np.savez_compressed(
                disk_path,
                **np_arrays,
                _metadata=pickle.dumps(other_data),
            )
        elif isinstance(value, np.ndarray):
            np.savez_compressed(disk_path, data=value)
        else:
            # Pickle other types
            with open(disk_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(value, f)

    def _load_from_disk(self, path: Path) -> Any:
        """Load item from disk."""
        if path.suffix == ".npz":
            loaded = np.load(path, allow_pickle=True)

            if "_metadata" in loaded:
                # Reconstruct dict
                result = {}
                metadata = pickle.loads(loaded["_metadata"].item())
                result.update(metadata)

                for key in loaded.files:
                    if key != "_metadata":
                        result[key] = loaded[key]
                return result
            elif "data" in loaded:
                return loaded["data"]
            else:
                return dict(loaded)

        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)

        return None

    def delete(self, key: str) -> bool:
        """Delete an item from cache."""
        deleted = False

        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True

        # Delete from disk
        for suffix in [".npz", ".pkl"]:
            disk_path = self._cache_dir / f"{key}{suffix}"
            if disk_path.exists():
                disk_path.unlink()
                deleted = True

        return deleted

    def clear(self) -> None:
        """Clear all cached items."""
        self._memory_cache.clear()

        # Clear disk cache
        for path in self._cache_dir.glob("*"):
            if path.is_file():
                path.unlink()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self._cache_dir.glob("*"))
        disk_size = sum(f.stat().st_size for f in disk_files if f.is_file())

        return {
            "memory_items": len(self._memory_cache),
            "max_memory_items": self._max_items,
            "disk_items": len(disk_files),
            "disk_size_mb": disk_size / (1024 * 1024),
            "cache_dir": str(self._cache_dir),
        }


# Global cache instance
_cache = CacheManager()


def get_cache() -> CacheManager:
    """Get the global cache manager."""
    return _cache
