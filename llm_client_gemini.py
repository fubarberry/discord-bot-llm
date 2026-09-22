# llm_client_gemini.py
import os
import json
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Any


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


async def get_llm_response(
    prompt: str,
    system_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    grounding: bool = False,
    model_name: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None
) -> Optional[str]:
    """
    Sends a prompt to the Google Gemini API and gets a response.

    Args:
        prompt (str): The user's prompt to send to the language model.
        system_prompt (str): The system prompt to set the context for the model.
        history (List[Dict[str, str]]): The conversation history.
        grounding (bool): Kept for backwards compatibility.
        model_name (str): The Gemini model to use (e.g. 'gemini-2.5-flash').
        images (Optional[List[Dict[str, Any]]]): Optional list of image dicts with 'data' (bytes) and 'mime_type' (str).
        temperature (Optional[float]): Model temperature (defaults to 0.7).

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

    try:
        client = genai.Client(api_key=api_key)
        
        # Convert history to Gemini's format
        gemini_history = []
        for message in history:
            role = 'user' if message.get('role') == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [{'text': message.get('content', '')}]})

        image_count_str = f" with {len(images)} image(s)" if images else ""
        print(f"Sending request to Gemini API (Model: {active_model}){image_count_str}...")
        
        if hasattr(client, 'aio'):
            temp = 0.7 if temperature is None else temperature
            chat = client.aio.chats.create(
                model=active_model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temp
                ),
                history=gemini_history
            )
            
            # Prepare message content (text + optional image parts)
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
                response = await chat.send_message(message_parts)
            else:
                response = await chat.send_message(prompt)
            
            print("Successfully received response from Gemini API.")
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
                print(f"Warning: response.text is None. Finish reason: {reason}. Info: {dump_info}")
                return f"⚠️ **Gemini Error**: Model returned no text. Finish reason: {reason}."
            
            return "⚠️ **Gemini Error**: No candidate responses received from the model."
        else:
            print("Error: Async client not available.")
            return "⚠️ **Gemini Error**: Async client is not available in the installed google-genai library."

    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API Exception: {error_msg}")
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            return f"⚠️ **Gemini API Error (Rate Limit/Quota)**: Quota exceeded or rate limit hit. Details: {error_msg}"
        elif "503" in error_msg or "overloaded" in error_msg.lower():
            return f"⚠️ **Gemini API Error (Overloaded)**: The Gemini service is currently overloaded. Please try again in a few moments. Details: {error_msg}"
        elif "NOT_FOUND" in error_msg or "404" in error_msg:
            return f"⚠️ **Gemini API Error (Model Not Found)**: Model '{active_model}' was not found. Use `/model` to set a valid model. Details: {error_msg}"
        elif "API_KEY_INVALID" in error_msg or "400" in error_msg and "API key" in error_msg:
            return f"⚠️ **Gemini API Error (Invalid Key)**: The provided `GEMINI_API_KEY` is invalid. Details: {error_msg}"
        else:
            return f"⚠️ **Gemini API Error**: {error_msg}"
