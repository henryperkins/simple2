"""
Token Management Module

Handles token counting, optimization, and management for Azure OpenAI API requests.
Provides efficient token estimation and prompt optimization with caching.

Version: 1.1.0
Author: Development Team
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import tiktoken

from logger import log_debug, log_error, log_info


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float

class TokenManager:
    """Token management for Azure OpenAI API requests."""

    MODEL_LIMITS = {
        "gpt-4": {"max_tokens": 8192, "cost_per_1k_prompt": 0.03, "cost_per_1k_completion": 0.06},
        "gpt-4-32k": {"max_tokens": 32768, "cost_per_1k_prompt": 0.06, "cost_per_1k_completion": 0.12},
        "gpt-3.5-turbo": {"max_tokens": 4096, "cost_per_1k_prompt": 0.0015, "cost_per_1k_completion": 0.002},
        "gpt-3.5-turbo-16k": {"max_tokens": 16384, "cost_per_1k_prompt": 0.003, "cost_per_1k_completion": 0.004}
    }

    def __init__(self, model: str = "gpt-4"):
        """Initialize TokenManager with model configuration."""
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.model_config = self.MODEL_LIMITS.get(model, self.MODEL_LIMITS["gpt-4"])
        log_debug(f"TokenManager initialized for model: {model}")

    @lru_cache(maxsize=128)
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text with caching.

        Args:
            text: Text to estimate tokens for

        Returns:
            int: Estimated token count
        """
        try:
            tokens = len(self.encoding.encode(text))
            log_debug(f"Estimated {tokens} tokens for text")
            return tokens
        except Exception as e:
            log_error(f"Error estimating tokens: {e}")
            return 0

    def optimize_prompt(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_sections: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """
        Optimize prompt to fit within token limits.

        Args:
            text: Text to optimize
            max_tokens: Maximum allowed tokens
            preserve_sections: Sections to preserve during optimization

        Returns:
            Tuple[str, TokenUsage]: Optimized text and token usage
        """
        max_tokens = max_tokens or (self.model_config["max_tokens"] // 2)
        current_tokens = self.estimate_tokens(text)

        if current_tokens <= max_tokens:
            log_info("Prompt is within token limits, no optimization needed.")
            return text, self._calculate_usage(current_tokens, 0)

        try:
            # Split into sections and preserve important parts
            sections = text.split('\n\n')
            preserved = []
            optional = []

            for section in sections:
                if preserve_sections and any(p in section for p in preserve_sections):
                    preserved.append(section)
                else:
                    optional.append(section)

            # Start with preserved content
            optimized = '\n\n'.join(preserved)
            remaining_tokens = max_tokens - self.estimate_tokens(optimized)

            # Add optional sections that fit
            for section in optional:
                section_tokens = self.estimate_tokens(section)
                if remaining_tokens >= section_tokens:
                    optimized = f"{optimized}\n\n{section}"
                    remaining_tokens -= section_tokens

            final_tokens = self.estimate_tokens(optimized)
            log_info(f"Prompt optimized from {current_tokens} to {final_tokens} tokens")
            return optimized, self._calculate_usage(final_tokens, 0)

        except Exception as e:
            log_error(f"Error optimizing prompt: {e}")
            return text, self._calculate_usage(current_tokens, 0)

    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Calculate token usage and cost."""
        total_tokens = prompt_tokens + completion_tokens
        prompt_cost = (prompt_tokens / 1000) * self.model_config["cost_per_1k_prompt"]
        completion_cost = (completion_tokens / 1000) * self.model_config["cost_per_1k_completion"]

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=prompt_cost + completion_cost
        )

    def validate_request(
        self,
        prompt: str,
        max_completion_tokens: Optional[int] = None
    ) -> Tuple[bool, Dict[str, int], str]:
        """
        Validate if request is within token limits.

        Args:
            prompt: The prompt to validate
            max_completion_tokens: Maximum allowed completion tokens

        Returns:
            Tuple[bool, Dict[str, int], str]: Validation result, metrics, and message
        """
        prompt_tokens = self.estimate_tokens(prompt)
        max_completion = max_completion_tokens or (self.model_config["max_tokens"] - prompt_tokens)
        total_tokens = prompt_tokens + max_completion

        metrics = {
            "prompt_tokens": prompt_tokens,
            "max_completion_tokens": max_completion,
            "total_tokens": total_tokens,
            "model_limit": self.model_config["max_tokens"]
        }

        if total_tokens > self.model_config["max_tokens"]:
            return False, metrics, f"Total tokens ({total_tokens}) exceeds model limit"

        return True, metrics, "Request validated successfully"

# Maintain backward compatibility with existing function calls
@lru_cache(maxsize=128)
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Legacy function for token estimation."""
    manager = TokenManager(model)
    return manager.estimate_tokens(text)

def optimize_prompt(text: str, max_tokens: int = 4000) -> str:
    """Legacy function for prompt optimization."""
    manager = TokenManager()
    optimized_text, _ = manager.optimize_prompt(text, max_tokens)
    return optimized_text