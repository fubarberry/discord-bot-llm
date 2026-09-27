# llm_client_gemini.py
import os
import json
import asyncio
from typing import List, Dict, Optional, Any, Callable, Awaitable

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def _get_default_model() -> str:
    """Reads gemini_model from settings.json or falls back to env/default."""
    try:
        with open("settings.json", "r") as f:
            data = json.load(f)
            if "gemini_model" in data and data["gemini_model"]:
                return data["gemini_model"]
    except Exception:
        pass
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def _get_default_fallback_model() -> str:
    """Reads gemini_fallback_model from settings.json or falls back to env/default."""
    try:
        with open("settings.json", "r") as f:
            data = json.load(f)
            if "gemini_fallback_model" in data and data["gemini_fallback_model"]:
                return data["gemini_fallback_model"]
            if "fallback_model" in data and data["fallback_model"]:
                return data["fallback_model"]
    except Exception:
        pass
    return os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")


def _get_default_timeout() -> float:
    """Reads gemini_timeout from settings.json or falls back to env/default (in seconds)."""
    try:
        with open("settings.json", "r") as f:
            data = json.load(f)
            if "gemini_timeout" in data and data["gemini_timeout"] is not None:
                return float(data["gemini_timeout"])
            if "timeout" in data and data["timeout"] is not None:
                return float(data["timeout"])
    except Exception:
        pass
    try:
        return float(os.getenv("GEMINI_TIMEOUT", 30.0))
    except ValueError:
        return 30.0


def _is_transient_or_overload_error(error: Exception) -> bool:
    """Checks if an error represents an overload, rate limit, timeout, or transient server issue."""
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return True

    error_msg = str(error).lower()
    transient_indicators = [
        "503",
        "unavailable",
        "overloaded",
        "high demand",
        "resource_exhausted",
        "429",
        "rate limit",
        "quota",
        "504",
        "deadline_exceeded",
        "502",
        "bad gateway",
        "timed out",
        "timeout",
    ]
    return any(indicator in error_msg for indicator in transient_indicators)


def _format_gemini_error(error: Exception, model_name: str, timeout: Optional[float] = None) -> str:
    """Formats a user-friendly error message for Gemini exceptions."""
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        to_str = f" after {timeout}s" if timeout else ""
        return f"⚠️ **Gemini API Error (Timeout)**: The request to model `{model_name}` timed out{to_str}. The service may be experiencing high demand."

    error_msg = str(error)
    if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
        return f"⚠️ **Gemini API Error (Rate Limit/Quota)**: Quota exceeded or rate limit hit on `{model_name}`. Details: {error_msg}"
    elif "503" in error_msg or "overloaded" in error_msg.lower():
        return f"⚠️ **Gemini API Error (Overloaded)**: The Gemini service is currently overloaded. Please try again in a few moments. Details: {error_msg}"
    elif "NOT_FOUND" in error_msg or "404" in error_msg:
        return f"⚠️ **Gemini API Error (Model Not Found)**: Model '{model_name}' was not found. Use `/model` to set a valid model. Details: {error_msg}"
    elif "API_KEY_INVALID" in error_msg or ("400" in error_msg and "API key" in error_msg):
        return f"⚠️ **Gemini API Error (Invalid Key)**: The provided `GEMINI_API_KEY` is invalid. Details: {error_msg}"
    else:
        return f"⚠️ **Gemini API Error**: {error_msg}"


def _format_fallback_failure(
    primary_model: str,
    primary_err: Exception,
    fallback_model: str,
    fallback_err: Exception,
    timeout: Optional[float] = None
) -> str:
    """Formats an error message when both primary and fallback models fail."""
    primary_is_to = isinstance(primary_err, (asyncio.TimeoutError, TimeoutError))
    fallback_is_to = isinstance(fallback_err, (asyncio.TimeoutError, TimeoutError))

    to_str = f" after {timeout}s" if timeout else ""
    primary_desc = f"timed out{to_str}" if primary_is_to else "overloaded / unavailable"
    fallback_desc = f"timed out{to_str}" if fallback_is_to else f"failed: {fallback_err}"

    if primary_is_to and fallback_is_to:
        return f"⚠️ **Gemini API Error (Timeout)**: Primary model `{primary_model}` and fallback model `{fallback_model}` both timed out{to_str}."

    return (
        f"⚠️ **Gemini API Error (Overloaded)**: Primary model `{primary_model}` was {primary_desc}, "
        f"and fallback model `{fallback_model}` also {fallback_desc}."
    )


async def _execute_gemini_request(
    client: Any,
    model_name: str,
    prompt: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    images: Optional[List[Dict[str, Any]]],
    temperature: Optional[float],
    timeout: Optional[float]
) -> str:
    """Executes a single Gemini chat request with timeout handling."""
    if not hasattr(client, 'aio'):
        raise RuntimeError("Async client is not available in the installed google-genai library.")

    # Convert history to Gemini's format
    gemini_history = []
    for message in history:
        role = 'user' if message.get('role') == 'user' else 'model'
        gemini_history.append({'role': role, 'parts': [{'text': message.get('content', '')}]})

    image_count_str = f" with {len(images)} image(s)" if images else ""
    print(f"Sending request to Gemini API (Model: {model_name}){image_count_str}...")

    temp = 0.7 if temperature is None else temperature
    chat = client.aio.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temp
        ),
        history=gemini_history
    )

    if images:
        message_parts = []
        for img in images:
            message_parts.append(
                types.Part.from_bytes(
                    data=img["data"],
                    mime_type=img.get("mime_type", "image/png")
                )
            )
        message_parts.append(types.Part.from_text(text=prompt))
        send_coro = chat.send_message(message_parts)
    else:
        send_coro = chat.send_message(prompt)

    if timeout and timeout > 0:
        response = await asyncio.wait_for(send_coro, timeout=timeout)
    else:
        response = await send_coro

    print(f"Successfully received response from Gemini API ({model_name}).")
    if response.text:
        return response.text.strip()

    # Fallback: if response.text is None, iteratively extract any text from the parts
    if hasattr(response, "candidates") and response.candidates:
        c = response.candidates[0]
        if hasattr(c, "content") and c.content and hasattr(c.content, "parts") and c.content.parts:
            extracted_text = ""
            for p in c.content.parts:
                if hasattr(p, "text") and p.text:
                    extracted_text += p.text

            if extracted_text.strip():
                return extracted_text.strip()

        reason = getattr(c, "finish_reason", "UNKNOWN")
        part_details = []
        if hasattr(c, "content") and hasattr(c.content, "parts"):
            for p in c.content.parts:
                if hasattr(p, "function_call") and p.function_call:
                    part_details.append(f"FunctionCall({p.function_call.name})")
                elif hasattr(p, "executable_code") and p.executable_code:
                    part_details.append("ExecutableCode")
                else:
                    part_details.append(f"Part(model_dump={p.model_dump()})")
        dump_info = ", ".join(part_details)
        print(f"Warning: response.text is None for model {model_name}. Finish reason: {reason}. Info: {dump_info}")
        return f"⚠️ **Gemini Error**: Model returned no text. Finish reason: {reason}."

    return "⚠️ **Gemini Error**: No candidate responses received from the model."


async def get_llm_response(
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    grounding: bool = False,
    model_name: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    fallback_model: Optional[str] = None,
    timeout: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_callback: Optional[Callable[[str], Awaitable[None]]] = None
) -> Optional[str]:
    """
    Sends a prompt to the Google Gemini API and gets a response.
    Supports automatic fallback model switching on overload/timeout and configurable request timeout.

    Args:
        prompt (str): The user's prompt to send to the language model.
        system_prompt (str): The system prompt to set the context for the model.
        history (List[Dict[str, str]]): The conversation history.
        grounding (bool): Kept for backwards compatibility.
        model_name (str): The Gemini model to use (e.g. 'gemini-2.5-flash').
        images (Optional[List[Dict[str, Any]]]): Optional list of image dicts with 'data' and 'mime_type'.
        temperature (Optional[float]): Model temperature (defaults to 0.7).
        fallback_model (Optional[str]): Gemini model to fall back to if primary model is overloaded/times out.
        timeout (Optional[float]): Timeout in seconds for the request.
        metadata (Optional[Dict[str, Any]]): Dict to record execution details (model used, fallback_used, etc.).
        status_callback (Optional[Callable]): Async callback for progress notifications.

    Returns:
        Optional[str]: The text response from the model, or an error description string.
    """
    if history is None:
        history = []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return "⚠️ **Gemini Configuration Error**: The `GEMINI_API_KEY` is missing. Please ask the bot administrator to configure it."

    active_model = model_name or _get_default_model()
    active_fallback = fallback_model or _get_default_fallback_model()
    active_timeout = timeout if timeout is not None else _get_default_timeout()

    if metadata is not None:
        metadata["provider"] = "GEMINI"
        metadata["model"] = active_model
        metadata["primary_model"] = active_model
        metadata["fallback_model"] = active_fallback
        metadata["fallback_used"] = False
        if temperature is not None:
            metadata["temperature"] = temperature

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return f"⚠️ **Gemini Client Error**: {e}"

    try:
        res = await _execute_gemini_request(
            client=client,
            model_name=active_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            images=images,
            temperature=temperature,
            timeout=active_timeout
        )
        return res
    except Exception as primary_exc:
        error_msg = str(primary_exc)
        print(f"Gemini API Exception ({active_model}): {error_msg}")

        can_fallback = (
            bool(active_fallback)
            and active_fallback.strip().lower() != active_model.strip().lower()
            and _is_transient_or_overload_error(primary_exc)
        )

        if can_fallback:
            is_timeout = isinstance(primary_exc, (asyncio.TimeoutError, TimeoutError))
            reason = f"timed out after {active_timeout}s" if is_timeout else "overloaded / unavailable"
            print(f"Primary model '{active_model}' was {reason}. Attempting fallback to '{active_fallback}'...")

            if status_callback:
                try:
                    await status_callback(f"⚠️ Primary model `{active_model}` was {reason}. Switching to fallback model `{active_fallback}`...")
                except Exception as cb_err:
                    print(f"Error calling status_callback: {cb_err}")

            try:
                fallback_res = await _execute_gemini_request(
                    client=client,
                    model_name=active_fallback,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=history,
                    images=images,
                    temperature=temperature,
                    timeout=active_timeout
                )

                if metadata is not None:
                    metadata["model"] = active_fallback
                    metadata["fallback_used"] = True
                    metadata["fallback_model"] = active_fallback

                print(f"Successfully received response from fallback Gemini model '{active_fallback}'.")
                return fallback_res

            except Exception as fallback_exc:
                print(f"Gemini fallback model '{active_fallback}' also failed: {fallback_exc}")
                return _format_fallback_failure(
                    primary_model=active_model,
                    primary_err=primary_exc,
                    fallback_model=active_fallback,
                    fallback_err=fallback_exc,
                    timeout=active_timeout
                )

        return _format_gemini_error(primary_exc, active_model, active_timeout)

