import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from logger import log_info, log_error, log_debug

@dataclass
class APIMetrics:
    """Metrics data structure for API operations."""
    timestamp: float
    operation: str
    response_time: float
    tokens_used: int
    status: str
    error: Optional[str] = None

@dataclass
class BatchMetrics:
    """Metrics data structure for batch operations."""
    total_functions: int
    successful: int
    failed: int
    total_tokens: int
    total_time: float
    average_time_per_function: float

class SystemMonitor:
    """
    Enhanced monitoring system implementing Azure OpenAI best practices.
    Tracks API usage, performance metrics, and system health.
    """
    def __init__(self):
        log_debug("Initializing SystemMonitor")
        self.api_metrics: List[APIMetrics] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.start_time: float = time.time()
        self.docstring_changes: Dict[str, List[Dict[str, str]]] = {
            'added': [],
            'updated': [],
            'retained': [],
            'failed': []
        }
        self.current_batch: Dict[str, int] = {
            'total_tokens': 0,
            'total_time': 0,
            'processed': 0,
            'failed': 0
        }
        
    def log_request(self, endpoint, tokens, response_time, status, error=None):
        """
        Log details of an API request.

        Args:
            endpoint (str): The API endpoint that was called.
            tokens (int): Number of tokens used in the request.
            response_time (float): Time taken for the request.
            status (str): Status of the request (e.g., 'success', 'error').
            error (str, optional): Error message if the request failed.
        """
        log_info(f"API Request to {endpoint}: {status}, Tokens: {tokens}, Time: {response_time}s")
        if error:
            log_error(f"Error: {error}")

    def log_api_request(
        self,
        endpoint: str,
        tokens: int,
        response_time: float,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """
        Log an API request with detailed metrics.

        Args:
            endpoint: The API endpoint called
            tokens: Number of tokens used
            response_time: Time taken for the request
            status: Status of the request
            error: Optional error message
        """
        log_debug(f"Logging API request to endpoint: {endpoint}")
        metric = APIMetrics(
            timestamp=time.time(),
            operation=endpoint,
            response_time=response_time,
            tokens_used=tokens,
            status=status,
            error=error
        )
        self.api_metrics.append(metric)
        log_info(f"API Request logged: {endpoint} - Status: {status}")

    def log_cache_hit(self, function_name: str) -> None:
        """Log a cache hit event."""
        self.cache_hits += 1
        log_info(f"Cache hit for function: {function_name}")

    def log_cache_miss(self, function_name: str) -> None:
        """Log a cache miss event."""
        self.cache_misses += 1
        log_info(f"Cache miss for function: {function_name}")

    def log_docstring_changes(self, action: str, function_name: str) -> None:
        """
        Log docstring changes with enhanced tracking.

        Args:
            action: Type of change (added/updated/retained/failed)
            function_name: Name of the function
        """
        log_debug(f"Logging docstring change: {action} for function: {function_name}")
        if action in self.docstring_changes:
            self.docstring_changes[action].append({
                'function': function_name,
                'timestamp': datetime.now().isoformat()
            })
            log_info(f"Docstring {action} for function: {function_name}")
        else:
            log_error(f"Unknown docstring action: {action}")

    def log_operation_complete(
        self,
        function_name: str,
        execution_time: float,
        tokens_used: int
    ) -> None:
        """
        Log completion of a function processing operation.

        Args:
            function_name: Name of the processed function
            execution_time: Time taken to process
            tokens_used: Number of tokens used
        """
        log_debug(f"Logging operation completion for function: {function_name}")
        self.current_batch['total_tokens'] += tokens_used
        self.current_batch['total_time'] += execution_time
        self.current_batch['processed'] += 1
        log_info(f"Operation complete for function: {function_name}")

    def log_batch_completion(self, total_functions: int) -> BatchMetrics:
        """
        Log completion of a batch processing operation.

        Args:
            total_functions: Total number of functions in the batch

        Returns:
            BatchMetrics: Metrics for the completed batch
        """
        log_debug("Logging batch completion")
        metrics = BatchMetrics(
            total_functions=total_functions,
            successful=self.current_batch['processed'],
            failed=self.current_batch['failed'],
            total_tokens=self.current_batch['total_tokens'],
            total_time=self.current_batch['total_time'],
            average_time_per_function=self.current_batch['total_time'] / 
                                    max(self.current_batch['processed'], 1)
        )
        
        # Reset batch metrics
        self.current_batch = {
            'total_tokens': 0,
            'total_time': 0,
            'processed': 0,
            'failed': 0
        }
        
        log_info(f"Batch processing completed: {metrics}")
        return metrics

    def get_metrics_summary(self) -> Dict:
        """
        Generate a comprehensive metrics summary.

        Returns:
            Dict: Complete metrics summary
        """
        log_debug("Generating metrics summary")
        runtime = time.time() - self.start_time
        total_requests = len(self.api_metrics)
        failed_requests = len([m for m in self.api_metrics if m.error])
        
        summary = {
            'runtime_seconds': runtime,
            'api_metrics': {
                'total_requests': total_requests,
                'failed_requests': failed_requests,
                'error_rate': failed_requests / max(total_requests, 1),
                'average_response_time': sum(m.response_time for m in self.api_metrics) / 
                                      max(total_requests, 1),
                'total_tokens_used': sum(m.tokens_used for m in self.api_metrics)
            },
            'cache_metrics': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            'docstring_changes': {
                action: len(changes) 
                for action, changes in self.docstring_changes.items()
            }
        }
        log_info(f"Metrics summary generated: {summary}")
        return summary

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            filepath: Path to save the metrics file
        """
        log_debug(f"Exporting metrics to file: {filepath}")
        try:
            metrics = self.get_metrics_summary()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            log_info(f"Metrics exported to: {filepath}")
        except Exception as e:
            log_error(f"Failed to export metrics: {str(e)}")