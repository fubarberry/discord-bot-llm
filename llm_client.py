# llm_client.py
import os
import json
from typing import List, Dict, Callable, Awaitable, Optional, Any
from llm_client_lmstudio import get_llm_response as get_lmstudio_response
from llm_client_gemini import get_llm_response as get_gemini_response
from search_agent import run_search_augmented_generation


def _get_default_provider() -> str:
    """Reads llm_provider from settings.json or falls back to env."""
    try:
        with open("settings.json", "r") as f:
            data = json.load(f)
            if "llm_provider" in data and data["llm_provider"]:
                return data["llm_provider"].upper()
    except Exception:
        pass
    return os.getenv("LLM_PROVIDER", "LMSTUDIO").upper()


def _get_default_model(provider: str) -> Optional[str]:
    """Reads default model from settings.json or falls back to env."""
    try:
        with open("settings.json", "r") as f:
            data = json.load(f)
            if provider == "GEMINI" and "gemini_model" in data and data["gemini_model"]:
                return data["gemini_model"]
    except Exception:
        pass
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash") if provider == "GEMINI" else None


async def query_llm(
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    thinking_enabled: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Sends a request to the configured LLM provider (Gemini or LM Studio).
    """
    if history is None:
        history = []

    active_provider = (provider or _get_default_provider()).upper()
    active_model = model or _get_default_model(active_provider)

    if active_provider == "GEMINI":
        print(f"Using Gemini API as the LLM provider (Model: {active_model}).")
        return await get_gemini_response(prompt, system_prompt, history, model_name=active_model, images=images)
    elif active_provider == "LMSTUDIO":
        print(f"Using LM Studio as the LLM provider (Model: {active_model or 'local-model'}).")
        return await get_lmstudio_response(prompt, system_prompt, thinking_enabled, history, model_name=active_model, images=images)
    else:
        print(f"Error: Unknown LLM_PROVIDER '{active_provider}'. Defaulting to LMSTUDIO.")
        return await get_lmstudio_response(prompt, system_prompt, thinking_enabled, history, model_name=active_model, images=images)


async def get_llm_response(
    prompt: str,
    system_prompt: str,
    thinking_enabled: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
    grounding: bool = False,
    status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Gets a response from the configured LLM provider, optionally using web search grounding.

    Args:
        prompt (str): The user's input prompt.
        system_prompt (str): The system prompt/personality.
        thinking_enabled (bool): Whether to show thinking tags (LM Studio).
        history (List[Dict[str, str]]): Conversation history.
        grounding (bool): Whether autonomous web search grounding is enabled.
        status_callback: Optional async callback to send progress updates (e.g. to Discord).
        provider (str): Optional override for LLM provider ('GEMINI' or 'LMSTUDIO').
        model (str): Optional override for model name.
        images (Optional[List[Dict[str, Any]]]): Optional images to include with the prompt.

    Returns:
        Optional[str]: Generated response string or error message.
    """
    if history is None:
        history = []

    if grounding:
        async def _query_helper(
            p: str,
            s: str,
            h: Optional[List[Dict[str, str]]],
            imgs: Optional[List[Dict[str, Any]]] = None
        ) -> Optional[str]:
            return await query_llm(
                p, s, h if h is not None else [],
                thinking_enabled=thinking_enabled,
                provider=provider,
                model=model,
                images=imgs
            )

        return await run_search_augmented_generation(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            query_llm_fn=_query_helper,
            status_callback=status_callback,
            images=images
        )

    return await query_llm(
        prompt,
        system_prompt,
        history=history,
        thinking_enabled=thinking_enabled,
        provider=provider,
        model=model,
        images=images
    )
