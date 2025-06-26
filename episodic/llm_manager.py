"""
Centralized LLM API call manager with thread-safe benchmarking.
Single entry point for all actual LLM API calls.
"""

import threading
import time
from typing import Dict, Any, Optional, Tuple, Union, Generator
from dataclasses import dataclass
import litellm
from episodic.config import config


@dataclass
class CallMetrics:
    """Thread-safe metrics for tracking API calls."""
    def __init__(self):
        self._lock = threading.Lock()
        self._calls_by_thread = {}
        self._total_calls = 0
        self._total_time = 0.0
        
    def record_call(self, duration: float):
        """Record an API call with thread safety."""
        thread_id = threading.get_ident()
        with self._lock:
            self._total_calls += 1
            self._total_time += duration
            if thread_id not in self._calls_by_thread:
                self._calls_by_thread[thread_id] = {'calls': 0, 'time': 0.0}
            self._calls_by_thread[thread_id]['calls'] += 1
            self._calls_by_thread[thread_id]['time'] += duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread-safe statistics."""
        with self._lock:
            return {
                'total_calls': self._total_calls,
                'total_time': self._total_time,
                'by_thread': dict(self._calls_by_thread)
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._calls_by_thread.clear()
            self._total_calls = 0
            self._total_time = 0.0


class LLMManager:
    """Centralized manager for all LLM API calls."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.metrics = CallMetrics()
        
    def make_api_call(
        self, 
        messages: list, 
        model: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[str, Dict], Tuple[Generator, None]]:
        """
        Single entry point for all LLM API calls.
        
        Args:
            messages: List of message dicts for the LLM
            model: Model name/identifier
            stream: Whether to stream the response
            **kwargs: Additional parameters for litellm
            
        Returns:
            For non-streaming: (response_text, cost_info)
            For streaming: (stream_generator, None)
        """
        start_time = time.time()
        thread_id = threading.get_ident()
        
        if config.get('debug', False):
            import traceback
            # Get the calling function name
            caller = traceback.extract_stack()[-3]
            print(f"[LLM API] Thread {thread_id}: Call #{self.metrics._total_calls + 1}")
            print(f"[LLM API] Model: {model} (stream={stream})")
            print(f"[LLM API] Called from: {caller.filename}:{caller.lineno} in {caller.name}()")
            print(f"[LLM API] Messages: {len(messages)} messages")
        
        try:
            # Make the actual API call
            response = litellm.completion(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            
            duration = time.time() - start_time
            self.metrics.record_call(duration)
            
            if stream:
                # For streaming, return the generator
                return response, None
            else:
                # For non-streaming, extract text and calculate cost
                response_text = response.choices[0].message.content
                
                # Calculate cost info using litellm's cost calculation
                try:
                    from litellm import completion_cost
                    cost = completion_cost(completion_response=response)
                except:
                    cost = 0.0
                
                cost_info = {
                    'input_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                    'output_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                    'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                    'cost_usd': cost
                }
                
                if config.get('debug', False):
                    print(f"[LLM API] Thread {thread_id}: Call #{self.metrics._total_calls} completed in {duration:.2f}s")
                    print(f"[LLM API] Response length: {len(response_text)} chars")
                    print("---")
                
                return response_text, cost_info
                
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_call(duration)
            
            if config.get('debug', False):
                print(f"[LLM API] Thread {thread_id}: Failed after {duration:.2f}s - {e}")
            
            raise
    
    def get_call_stats(self) -> Dict[str, Any]:
        """Get current API call statistics."""
        return self.metrics.get_stats()
    
    def reset_stats(self):
        """Reset API call statistics."""
        self.metrics.reset()


# Global singleton instance
llm_manager = LLMManager()