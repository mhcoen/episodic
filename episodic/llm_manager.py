"""
Centralized LLM API call manager with thread-safe benchmarking.
Single entry point for all actual LLM API calls.
"""

import threading
import time
import os
from typing import Dict, Any, Tuple, Union, Generator, Optional
from dataclasses import dataclass
from contextlib import redirect_stdout, redirect_stderr
import io
import litellm
from episodic.config import config

# Suppress LiteLLM debug messages
litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "ERROR"


@dataclass
class CallMetrics:
    """Thread-safe metrics for tracking API calls."""
    def __init__(self):
        self._lock = threading.Lock()
        self._calls_by_thread = {}
        self._total_calls = 0
        self._total_time = 0.0
        # Add cost tracking
        self._total_cost_usd = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens = 0
        
    def record_call(self, duration: float, cost_info: Optional[Dict[str, Any]] = None):
        """Record an API call with thread safety."""
        thread_id = threading.get_ident()
        with self._lock:
            self._total_calls += 1
            self._total_time += duration
            if thread_id not in self._calls_by_thread:
                self._calls_by_thread[thread_id] = {'calls': 0, 'time': 0.0}
            self._calls_by_thread[thread_id]['calls'] += 1
            self._calls_by_thread[thread_id]['time'] += duration
            
            # Track costs if provided
            if cost_info:
                self._total_cost_usd += cost_info.get('cost_usd', 0.0)
                self._total_input_tokens += cost_info.get('input_tokens', 0)
                self._total_output_tokens += cost_info.get('output_tokens', 0)
                self._total_tokens += cost_info.get('total_tokens', 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread-safe statistics."""
        with self._lock:
            return {
                'total_calls': self._total_calls,
                'total_time': self._total_time,
                'by_thread': dict(self._calls_by_thread),
                'total_cost_usd': self._total_cost_usd,
                'total_input_tokens': self._total_input_tokens,
                'total_output_tokens': self._total_output_tokens,
                'total_tokens': self._total_tokens
            }
    
    def get_cost_info(self) -> Dict[str, Any]:
        """Get just the cost information."""
        with self._lock:
            return {
                'total_cost_usd': self._total_cost_usd,
                'total_input_tokens': self._total_input_tokens,
                'total_output_tokens': self._total_output_tokens,
                'total_tokens': self._total_tokens
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._calls_by_thread.clear()
            self._total_calls = 0
            self._total_time = 0.0
            self._total_cost_usd = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_tokens = 0


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
            # Make the actual API call with output suppression
            # Suppress the annoying "Provider List" messages
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    stream=stream,
                    **kwargs
                )
            
            duration = time.time() - start_time
            
            if stream:
                # For streaming, wrap the generator to capture usage after completion
                def streaming_wrapper():
                    """Wrapper that yields chunks and tracks usage after stream completes."""
                    total_content = []
                    last_chunk = None
                    
                    try:
                        for chunk in response:
                            if hasattr(chunk, 'choices') and chunk.choices:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    total_content.append(delta.content)
                            last_chunk = chunk
                            yield chunk
                    finally:
                        # After streaming completes, calculate cost if we have usage info
                        if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                            try:
                                from litellm import stream_cost_calculator
                                # Try to use litellm's stream cost calculator if available
                                cost = stream_cost_calculator(model=model, usage=last_chunk.usage)
                            except:
                                # Fallback: estimate cost based on usage
                                try:
                                    from litellm import completion_cost
                                    # Create a mock response with usage for cost calculation
                                    mock_response = type('MockResponse', (), {
                                        'usage': last_chunk.usage,
                                        'model': model
                                    })()
                                    cost = completion_cost(completion_response=mock_response)
                                except:
                                    cost = 0.0
                            
                            cost_info = {
                                'input_tokens': last_chunk.usage.prompt_tokens if hasattr(last_chunk.usage, 'prompt_tokens') else 0,
                                'output_tokens': last_chunk.usage.completion_tokens if hasattr(last_chunk.usage, 'completion_tokens') else 0,
                                'total_tokens': last_chunk.usage.total_tokens if hasattr(last_chunk.usage, 'total_tokens') else 0,
                                'cost_usd': cost
                            }
                            
                            # Update metrics with the cost info
                            self.metrics.record_call(0, cost_info)  # 0 duration since we already recorded time
                            
                            if config.get('debug', False):
                                print(f"[LLM API] Stream completed. Cost: ${cost:.6f}")
                        else:
                            # No usage info available, just record the call
                            self.metrics.record_call(duration)
                
                # Return the wrapper generator
                return streaming_wrapper(), None
            else:
                # For non-streaming, extract text and calculate cost
                response_text = response.choices[0].message.content
                
                # Calculate cost info using litellm's cost calculation
                try:
                    from litellm import completion_cost
                    cost = completion_cost(completion_response=response)
                    if config.get('debug', False):
                        print(f"[LLM API] Cost calculated: ${cost:.6f}")
                except Exception as e:
                    cost = 0.0
                    if config.get('debug', False):
                        print(f"[LLM API] Cost calculation failed: {e}")
                        print(f"[LLM API] Model: {model}")
                        print(f"[LLM API] Response has usage: {hasattr(response, 'usage')}")
                        if hasattr(response, 'usage'):
                            print(f"[LLM API] Usage: {response.usage}")
                
                cost_info = {
                    'input_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                    'output_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                    'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                    'cost_usd': cost
                }
                
                # Record the call with cost info
                self.metrics.record_call(duration, cost_info)
                
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
    
    def get_session_costs(self) -> Dict[str, Any]:
        """Get current session cost information."""
        return self.metrics.get_cost_info()


# Global singleton instance
llm_manager = LLMManager()