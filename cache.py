"""
Enhanced Cache Management Module

This module provides advanced caching capabilities for docstrings and API responses,
with Redis-based distributed caching and in-memory fallback.

Version: 1.2.0
Author: Development Team
"""

import json
import time
from typing import Optional, Dict, Any, List
import redis
from dataclasses import dataclass
import asyncio
from logger import log_info, log_error
from monitoring import SystemMonitor

@dataclass
class CacheStats:
    """Statistics for cache operations."""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_requests: int = 0
    cache_size: int = 0
    avg_response_time: float = 0.0

class Cache:
    """Enhanced cache management with Redis and in-memory fallback."""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 86400,
        max_retries: int = 3,
        max_memory_items: int = 1000,
        monitor: Optional[SystemMonitor] = None
    ):
        """Initialize the enhanced cache system."""
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.monitor = monitor or SystemMonitor()
        self.stats = CacheStats()
        
        # Initialize Redis
        for attempt in range(max_retries):
            try:
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
                self.redis_client.ping()
                self.redis_available = True
                log_info("Redis cache initialized successfully")
                break
            except redis.RedisError as e:
                log_error(f"Redis connection error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    self.redis_available = False
                    log_error("Falling back to in-memory cache only")
                time.sleep(2 ** attempt)

        # Initialize in-memory cache
        self.memory_cache = LRUCache(max_size=max_memory_items)

    async def get_cached_docstring(
        self,
        function_id: str,
        return_metadata: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Enhanced docstring retrieval with fallback."""
        cache_key = self._generate_cache_key(function_id)
        start_time = time.time()

        try:
            # Try Redis first
            if self.redis_available:
                value = await self._get_from_redis(cache_key)
                if value:
                    self._update_stats(hit=True, response_time=time.time() - start_time)
                    return value if return_metadata else {'docstring': value['docstring']}

            # Try memory cache
            value = await self.memory_cache.get(cache_key)
            if value:
                self._update_stats(hit=True, response_time=time.time() - start_time)
                return value if return_metadata else {'docstring': value['docstring']}

            self._update_stats(hit=False, response_time=time.time() - start_time)
            return None

        except Exception as e:
            log_error(f"Cache retrieval error: {e}")
            self._update_stats(error=True, response_time=time.time() - start_time)
            return None

    async def save_docstring(
        self,
        function_id: str,
        docstring_data: Dict[str, Any],
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Enhanced docstring saving with tags support."""
        cache_key = self._generate_cache_key(function_id)
        ttl = ttl or self.default_ttl

        try:
            # Add metadata
            cache_entry = {
                **docstring_data,
                'cache_metadata': {
                    'timestamp': time.time(),
                    'ttl': ttl,
                    'tags': tags or []
                }
            }

            # Try Redis first
            if self.redis_available:
                success = await self._set_in_redis(cache_key, cache_entry, ttl)
                if success:
                    return True

            # Fallback to memory cache
            await self.memory_cache.set(cache_key, cache_entry, ttl)
            return True

        except Exception as e:
            log_error(f"Cache save error: {e}")
            return False

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        count = 0
        try:
            if self.redis_available:
                count += await self._invalidate_redis_by_tags(tags)
            count += await self.memory_cache.invalidate_by_tags(tags)
            return count
        except Exception as e:
            log_error(f"Tag-based invalidation error: {e}")
            return count

    def _update_stats(self, hit: bool = False, error: bool = False, response_time: float = 0.0):
        """Update cache statistics."""
        self.stats.total_requests += 1
        if hit:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        if error:
            self.stats.errors += 1
        
        self.stats.avg_response_time = (
            (self.stats.avg_response_time * (self.stats.total_requests - 1) + response_time)
            / self.stats.total_requests
        )

    def _generate_cache_key(self, function_id: str) -> str:
        """Generate a unique cache key for a function."""
        return f"docstring:v1:{function_id}"

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis with error handling."""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            log_error(f"Redis get error: {e}")
            return None

    async def _set_in_redis(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis with error handling."""
        try:
            serialized_value = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized_value)
            
            # Store tags for efficient tag-based invalidation
            if value.get('tags'):
                for tag in value['tags']:
                    tag_key = f"tag:{tag}"
                    self.redis_client.sadd(tag_key, key)
                    self.redis_client.expire(tag_key, ttl)
            
            return True
        except Exception as e:
            log_error(f"Redis set error: {e}")
            return False

    async def _invalidate_redis_by_tags(self, tags: List[str]) -> int:
        """Invalidate Redis entries by tags."""
        count = 0
        try:
            for tag in tags:
                tag_key = f"tag:{tag}"
                keys = self.redis_client.smembers(tag_key)
                if keys:
                    count += len(keys)
                    for key in keys:
                        await self._delete_from_redis(key)
                    self.redis_client.delete(tag_key)
            return count
        except Exception as e:
            log_error(f"Redis tag invalidation error: {e}")
            return count

    async def _delete_from_redis(self, key: str) -> bool:
        """Delete value from Redis with error handling."""
        try:
            # Get tags before deletion
            value = await self._get_from_redis(key)
            if value and value.get('tags'):
                for tag in value['tags']:
                    self.redis_client.srem(f"tag:{tag}", key)
            
            self.redis_client.delete(key)
            return True
        except Exception as e:
            log_error(f"Redis delete error: {e}")
            return False

class LRUCache:
    """Thread-safe LRU cache implementation for in-memory caching."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache and update access order."""
        async with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            if self._is_expired(entry):
                await self.delete(key)
                return None

            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry['value']

    async def set(self, key: str, value: Any, ttl: int):
        """Set value in cache with TTL."""
        async with self.lock:
            # Evict oldest item if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()

            entry = {
                'value': value,
                'expires_at': time.time() + ttl
            }
            
            self.cache[key] = entry
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    async def delete(self, key: str):
        """Delete value from cache."""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        count = 0
        async with self.lock:
            keys_to_delete = []
            for key, entry in self.cache.items():
                if entry['value'].get('tags'):
                    if any(tag in entry['value']['tags'] for tag in tags):
                        keys_to_delete.append(key)
            
            for key in keys_to_delete:
                await self.delete(key)
                count += 1
            
            return count

    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        return time.time() > entry['expires_at']

    async def _evict_oldest(self):
        """Evict oldest cache entry."""
        if self.access_order:
            oldest_key = self.access_order[0]
            await self.delete(oldest_key)