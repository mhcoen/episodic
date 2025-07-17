"""
Add this to your script or episodic startup to suppress the LiteLLM async warning.
"""

import warnings
import asyncio

# Suppress the specific RuntimeWarning from LiteLLM
warnings.filterwarnings("ignore", message="coroutine 'close_litellm_async_clients' was never awaited")

# Alternative: Set event loop policy to suppress all asyncio warnings
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # Windows
# asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())  # Unix/Linux/Mac

# Or add to your environment:
# export PYTHONWARNINGS="ignore::RuntimeWarning:litellm.llms.custom_httpx.async_client_cleanup"