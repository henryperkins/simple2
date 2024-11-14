# cache.py
import json
import time
from typing import Optional, Dict, Any
import redis
from logger import log_info, log_error, log_debug

class Cache:
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 86400,
        max_retries: int = 3
    ):
        """
        Initialize the cache with retry mechanism.

        Args:
            host (str): Redis host
            port (int): Redis port
            db (int): Redis database number
            password (Optional[str]): Redis password
            default_ttl (int): Default time-to-live for cache entries
            max_retries (int): Maximum number of connection retry attempts
        """
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        log_debug(f"Initializing Cache with host={host}, port={port}, db={db}, max_retries={max_retries}")

        for attempt in range(max_retries):
            try:
                log_debug(f"Attempting to connect to Redis (attempt {attempt + 1})")
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    retry_on_timeout=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()  # Test connection
                log_info("Connected to Redis cache successfully")
                break
            except redis.RedisError as e:
                log_error(f"Redis connection error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    log_error(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                    raise
                time.sleep(2 ** attempt)

    async def get_cached_docstring(
        self,
        function_id: str,
        return_metadata: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached docstring with retry mechanism and error handling.

        Args:
            function_id: Unique identifier for the function
            return_metadata: Whether to return metadata with the docstring

        Returns:
            Optional[Dict[str, Any]]: Cached docstring data or None if not found
        """
        cache_key = self._generate_cache_key(function_id)
        log_debug(f"Retrieving cache for function ID: {function_id} with key: {cache_key}")

        for attempt in range(self.max_retries):
            try:
                cached_value = self.redis_client.get(cache_key)
                if cached_value:
                    try:
                        docstring_data = json.loads(cached_value)
                        log_info(f"Cache hit for function ID: {function_id}")
                        return docstring_data if return_metadata else {'docstring': docstring_data['docstring']}
                    except json.JSONDecodeError as e:
                        log_error(f"Invalid JSON in cache for function ID: {function_id}: {e}")
                        self.redis_client.delete(cache_key)
                        return None

                log_info(f"Cache miss for function ID: {function_id}")
                return None

            except redis.RedisError as e:
                log_error(f"Redis error during get (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

    async def save_docstring(
        self,
        function_id: str,
        docstring_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Save docstring to cache with retry mechanism.

        Args:
            function_id: Unique identifier for the function
            docstring_data: Dictionary containing docstring and metadata
            ttl: Time-to-live in seconds (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(function_id)
        ttl = ttl or self.default_ttl
        log_debug(f"Saving docstring to cache for function ID: {function_id} with key: {cache_key} and TTL: {ttl}")

        for attempt in range(self.max_retries):
            try:
                docstring_data['cache_metadata'] = {
                    'timestamp': time.time(),
                    'ttl': ttl
                }
                serialized_data = json.dumps(docstring_data)
                self.redis_client.setex(name=cache_key, time=ttl, value=serialized_data)
                log_info(f"Cached docstring for function ID: {function_id}")
                return True

            except (redis.RedisError, json.JSONEncodeError) as e:
                log_error(f"Cache save error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return False

    async def delete_cached_docstring(self, function_id: str) -> bool:
        """
        Delete a cached docstring with retry mechanism.

        Args:
            function_id: Unique identifier for the function

        Returns:
            bool: True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(function_id)
        log_debug(f"Deleting cache for function ID: {function_id} with key: {cache_key}")

        for attempt in range(self.max_retries):
            try:
                result = self.redis_client.delete(cache_key)
                if result:
                    log_info(f"Deleted cached docstring for function ID: {function_id}")
                return bool(result)

            except redis.RedisError as e:
                log_error(f"Cache delete error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return False

    async def clear_cache(self) -> bool:
        """
        Clear all cached docstrings with retry mechanism.

        Returns:
            bool: True if successful, False otherwise
        """
        log_debug("Clearing entire cache")

        for attempt in range(self.max_retries):
            try:
                self.redis_client.flushdb()
                log_info("Cache cleared successfully")
                return True

            except redis.RedisError as e:
                log_error(f"Cache clear error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        log_debug("Retrieving cache statistics")

        try:
            info = self.redis_client.info()
            stats = {
                'used_memory': info['used_memory_human'],
                'connected_clients': info['connected_clients'],
                'total_keys': self.redis_client.dbsize(),
                'uptime_seconds': info['uptime_in_seconds']
            }
            log_info(f"Cache statistics retrieved: {stats}")
            return stats
        except redis.RedisError as e:
            log_error(f"Failed to get cache stats: {e}")
            return {}

    def _generate_cache_key(self, function_id: str) -> str:
        """Generate a unique cache key for a function."""
        log_debug(f"Generating cache key for function ID: {function_id}")
        return f"docstring:v1:{function_id}"

    def __enter__(self):
        """Context manager entry."""
        log_debug("Entering cache context manager")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        log_debug("Exiting cache context manager")
        try:
            self.redis_client.close()
            log_info("Redis connection closed successfully")
        except Exception as e:
            log_error(f"Error closing Redis connection: {e}")