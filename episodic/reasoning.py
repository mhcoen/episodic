"""
Reasoning control for models with configurable thinking modes.

Handles three mechanisms:
- api_param: Pass parameters to API call (GPT-5.2, Ollama)
- system_prompt_tag: Inject tags into system prompt (Qwen3 via HF, Nemotron)
- inherent: Model always reasons, no control available (DeepSeek-R1)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from episodic.config import config


@dataclass
class ReasoningConfig:
    """Session-level reasoning configuration."""
    enabled: bool = True
    effort: Optional[str] = None      # For GPT-5.2: minimal/low/medium/high
    verbosity: Optional[str] = None   # For GPT-5.2: low/medium/high


class ReasoningController:
    """Applies reasoning configuration to API calls and prompts."""

    def __init__(self, model_config: dict):
        """
        Args:
            model_config: The specific model entry from models_template.json
        """
        self.model_config = model_config
        self.reasoning_spec = model_config.get("reasoning")

    @property
    def mechanism(self) -> Optional[str]:
        """Return the reasoning control mechanism, if any."""
        if not self.reasoning_spec:
            return None
        return self.reasoning_spec.get("mechanism")

    @property
    def is_controllable(self) -> bool:
        """Return True if reasoning can be toggled on/off."""
        return self.mechanism in ("api_param", "system_prompt_tag")

    @property
    def is_always_on(self) -> bool:
        """Return True if model always reasons (inherent mechanism)."""
        if self.mechanism == "inherent":
            return self.reasoning_spec.get("always_on", True)
        return False

    @property
    def has_reasoning(self) -> bool:
        """Return True if model supports reasoning at all."""
        return self.reasoning_spec is not None

    @property
    def default_enabled(self) -> bool:
        """Return the default state for reasoning."""
        if not self.reasoning_spec:
            return False

        # For inherent, always on
        if self.mechanism == "inherent":
            return True

        # For system_prompt_tag, check default field
        if self.mechanism == "system_prompt_tag":
            return self.reasoning_spec.get("default", "on") == "on"

        # For api_param, check if default is truthy
        if self.mechanism == "api_param":
            params = self.reasoning_spec.get("params", {})
            # Check think param (Ollama style)
            if "think" in params:
                return params["think"].get("default", True)
            # Check enable_thinking param (DeepSeek style)
            if "enable_thinking" in params:
                return params["enable_thinking"].get("default", False)
            # If has reasoning_effort, reasoning is implicitly available
            if "reasoning_effort" in params:
                return True

        return False

    def get_api_params(self, reasoning_config: ReasoningConfig) -> Dict[str, Any]:
        """
        Return additional API parameters for reasoning control.

        Args:
            reasoning_config: Current session reasoning configuration

        Returns:
            Dict of params to merge into API call
        """
        if self.mechanism != "api_param":
            return {}

        params = {}
        param_specs = self.reasoning_spec.get("params", {})

        # Handle boolean think param (Ollama style)
        if "think" in param_specs:
            params["think"] = reasoning_config.enabled

        # Handle enable_thinking param (DeepSeek V3.1 style)
        if "enable_thinking" in param_specs:
            params["enable_thinking"] = reasoning_config.enabled

        # Handle effort param (GPT-5.2 style)
        if "reasoning_effort" in param_specs:
            if reasoning_config.effort:
                params["reasoning_effort"] = reasoning_config.effort
            elif reasoning_config.enabled:
                params["reasoning_effort"] = param_specs["reasoning_effort"].get("default", "medium")

        # Handle verbosity param (GPT-5.2 style)
        if "verbosity" in param_specs:
            if reasoning_config.verbosity:
                params["verbosity"] = reasoning_config.verbosity

        return params

    def modify_system_prompt(self, system_prompt: str, reasoning_config: ReasoningConfig) -> str:
        """
        Inject reasoning tags into system prompt if needed.

        Args:
            system_prompt: Original system prompt
            reasoning_config: Current session reasoning configuration

        Returns:
            Modified system prompt with reasoning tags if applicable
        """
        if self.mechanism != "system_prompt_tag":
            return system_prompt

        tags = self.reasoning_spec.get("tags", {})
        tag = tags.get("on") if reasoning_config.enabled else tags.get("off")

        if not tag:
            return system_prompt

        placement = self.reasoning_spec.get("placement", "end_of_system")

        if placement == "end_of_system":
            return f"{system_prompt}\n\n{tag}"
        elif placement == "start_of_system":
            return f"{tag}\n\n{system_prompt}"

        return system_prompt

    def get_available_options(self) -> Dict[str, List]:
        """
        Return available options for this model's reasoning control.

        Returns:
            Dict mapping param names to their options, e.g.:
            {"reasoning_effort": ["minimal", "low", "medium", "high"]}
        """
        if self.mechanism != "api_param":
            return {}

        options = {}
        param_specs = self.reasoning_spec.get("params", {})

        for param_name, spec in param_specs.items():
            if "options" in spec:
                options[param_name] = spec["options"]

        return options


def get_type_indicator_with_reasoning(model_config: dict, base_indicator: str) -> str:
    """
    Return the type indicator string for a model, including reasoning suffix.

    Width is fixed at 5 characters inside brackets.

    Examples:
        [CI]   - Chat+Instruct, no reasoning
        [CIR]  - Chat+Instruct with reasoning
        [I R]  - Instruct with reasoning
        [C R]  - Chat with reasoning
        [B]    - Base, no reasoning

    Args:
        model_config: Model configuration dict from models.json
        base_indicator: Base indicator like '[CI]', '[I]', '[C]', '[B]'

    Returns:
        Padded indicator string like '[CIR] ' or '[I R] '
    """
    has_reasoning = "reasoning" in model_config

    # Extract base letters from indicator (e.g., 'CI' from '[CI]')
    base = base_indicator.strip('[]').strip()

    if has_reasoning:
        # Append R
        indicator = f"{base}R"
    else:
        indicator = base

    # Pad to 3 chars for consistent width
    indicator = indicator.ljust(3)

    return f"[{indicator}]"


def get_session_reasoning_config() -> ReasoningConfig:
    """
    Get the current session's reasoning configuration from config.

    Returns:
        ReasoningConfig populated from config settings
    """
    return ReasoningConfig(
        enabled=config.get("reasoning_enabled", True),
        effort=config.get("reasoning_effort"),
        verbosity=config.get("reasoning_verbosity")
    )


def set_session_reasoning_config(reasoning_config: ReasoningConfig):
    """
    Save reasoning configuration to config.

    Args:
        reasoning_config: Configuration to save
    """
    config.set("reasoning_enabled", reasoning_config.enabled)
    if reasoning_config.effort:
        config.set("reasoning_effort", reasoning_config.effort)
    if reasoning_config.verbosity:
        config.set("reasoning_verbosity", reasoning_config.verbosity)


def get_reasoning_controller_for_model(model_name: str) -> Optional[ReasoningController]:
    """
    Get a ReasoningController for a specific model by name.

    Args:
        model_name: Model name (e.g., 'gpt-5.2', 'nemotron-3-nano')

    Returns:
        ReasoningController if model found, None otherwise
    """
    from episodic.model_config import get_model_config
    from episodic.llm_config import find_provider_for_model

    mc = get_model_config()
    provider = find_provider_for_model(model_name)

    if not provider:
        # Try to extract provider from model name
        if "/" in model_name:
            provider = model_name.split("/")[0]
        else:
            return None

    model_info = mc.get_model_info(provider, model_name)
    if model_info:
        return ReasoningController(model_info)

    return None


def model_supports_reasoning(model_name: str) -> bool:
    """
    Check if a model supports reasoning control.

    Args:
        model_name: Model name to check

    Returns:
        True if model has reasoning configuration
    """
    controller = get_reasoning_controller_for_model(model_name)
    return controller is not None and controller.has_reasoning
