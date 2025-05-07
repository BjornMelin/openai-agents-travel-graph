"""
Caching mechanisms for browser automation in the travel planner system.

This module implements caching for browser automation to improve performance
and reduce the number of web requests, making the travel planning process
more efficient.
"""

import hashlib
import json
import os
import time
from typing import Any

from stagehand.cache import Cache


class FlightSearchCache:
    """Cache for flight search results."""

    def __init__(self, cache_dir: str | None = None, expiry_hours: int = 6):
        """
        Initialize flight search cache.

        Args:
            cache_dir: Directory to store cache files
            expiry_hours: Hours after which cache entries expire
        """
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cache", "flight_search"
            )

        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_key(self, search_params: dict[str, Any]) -> str:
        """
        Generate a cache key from search parameters.

        Args:
            search_params: Flight search parameters

        Returns:
            Cache key string
        """
        # Create a stable representation of the search parameters
        param_str = json.dumps(search_params, sort_keys=True)
        # Create a hash for the parameters
        hash_obj = hashlib.md5(param_str.encode())
        return hash_obj.hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, search_params: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get cached flight search results if available and not expired.

        Args:
            search_params: Flight search parameters

        Returns:
            Cached results or None if not found or expired
        """
        key = self._generate_key(search_params)
        cache_path = self._get_cache_path(key)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path) as f:
                cache_data = json.load(f)

            # Check if cache has expired
            timestamp = cache_data.get("timestamp", 0)
            if time.time() - timestamp > self.expiry_hours * 3600:
                # Cache expired
                return None

            return cache_data.get("results")
        except Exception:
            # If any error occurs reading the cache, return None
            return None

    def set(self, search_params: dict[str, Any], results: dict[str, Any]):
        """
        Cache flight search results.

        Args:
            search_params: Flight search parameters
            results: Search results to cache
        """
        key = self._generate_key(search_params)
        cache_path = self._get_cache_path(key)

        cache_data = {
            "timestamp": time.time(),
            "search_params": search_params,
            "results": results,
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            # Log but don't raise exception for cache failures
            print(f"Error caching flight results: {e!s}")

    def clear_expired(self):
        """Clear all expired cache entries."""
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith(".json"):
                continue

            cache_path = os.path.join(self.cache_dir, filename)
            try:
                with open(cache_path) as f:
                    cache_data = json.load(f)

                timestamp = cache_data.get("timestamp", 0)
                if time.time() - timestamp > self.expiry_hours * 3600:
                    # Cache expired, delete file
                    os.remove(cache_path)
            except Exception:
                # If any error occurs reading the cache, skip it
                continue

    def clear_all(self):
        """Clear all cache entries."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                os.remove(os.path.join(self.cache_dir, filename))


class CustomStagehandCache(Cache):
    """Custom cache implementation for Stagehand actions."""

    def __init__(self, cache_dir: str, expiry_hours: int = 24):
        """
        Initialize custom Stagehand cache.

        Args:
            cache_dir: Directory to store cache files
            expiry_hours: Hours after which cache entries expire
        """
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_key(self, action: str, inputs: dict[str, Any]) -> str:
        """
        Generate a cache key from action and inputs.

        Args:
            action: Action name
            inputs: Action inputs

        Returns:
            Cache key string
        """
        # Create a stable representation of the action and inputs
        input_str = json.dumps({"action": action, "inputs": inputs}, sort_keys=True)
        # Create a hash for the input string
        hash_obj = hashlib.md5(input_str.encode())
        return hash_obj.hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{key}.json")

    async def get(self, action: str, inputs: dict[str, Any]) -> Any | None:
        """
        Get cached result for an action if available and not expired.

        Args:
            action: Action name
            inputs: Action inputs

        Returns:
            Cached result or None if not found or expired
        """
        key = self._generate_key(action, inputs)
        cache_path = self._get_cache_path(key)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path) as f:
                cache_data = json.load(f)

            # Check if cache has expired
            timestamp = cache_data.get("timestamp", 0)
            if time.time() - timestamp > self.expiry_hours * 3600:
                # Cache expired
                return None

            return cache_data.get("result")
        except Exception:
            # If any error occurs reading the cache, return None
            return None

    async def set(self, action: str, inputs: dict[str, Any], result: Any):
        """
        Cache action result.

        Args:
            action: Action name
            inputs: Action inputs
            result: Action result to cache
        """
        key = self._generate_key(action, inputs)
        cache_path = self._get_cache_path(key)

        cache_data = {
            "timestamp": time.time(),
            "action": action,
            "inputs": inputs,
            "result": result,
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
        except Exception:
            # Silently fail for cache errors
            pass
