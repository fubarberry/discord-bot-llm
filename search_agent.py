# search_agent.py
import re
from typing import List, Dict, Callable, Awaitable, Optional, Any
from web_search import search_duckduckgo, read_url_jina


async def _safe_update_status(status_callback: Optional[Callable[[str], Awaitable[None]]], text: str):
    """Safely invokes the status callback without failing if rate-limited or errored."""
    if status_callback:
        try:
            await status_callback(text)
        except Exception as e:
            print(f"Status update callback error: {e}")


def _parse_search_query(text: str) -> Optional[str]:
    """Extracts search query from model output."""
    match = re.search(r"SEARCH:\s*(.+)", text, re.IGNORECASE)
    if match:
        query = match.group(1).strip().strip('"\'')
        return query if query else None
    return None


def _parse_read_url(text: str) -> Optional[str]:
    """Extracts URL to read from model output."""
    match = re.search(r"READ:\s*(?:<|['\"])?(https?://[^\s>'\"\n]+)(?:>|['\"])?", text, re.IGNORECASE)
    if match:
        url = match.group(1).strip().rstrip('.,;')
        return url if url else None
    return None


async def run_search_augmented_generation(
    prompt: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    query_llm_fn: Callable[..., Awaitable[Optional[str]]],
    status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    max_search_steps: int = 3,
    images: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Orchestrates the decision, web search, web page reading, and response generation.

    Args:
        prompt (str): User's prompt.
        system_prompt (str): Active system prompt/personality for the assistant.
        history (List[Dict[str, str]]): Conversation history.
        query_llm_fn: Async callable that takes (prompt, system_prompt, history) and returns str | None.
        status_callback: Async callable to update Discord status message.
        max_search_steps (int): Maximum iterative search/read steps.

    Returns:
        Optional[str]: Final generated response.
    """
    async def _invoke_query_llm(p: str, s: str, h: Optional[List[Dict[str, str]]], imgs: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        if imgs:
            try:
                return await query_llm_fn(p, s, h, imgs)
            except TypeError:
                try:
                    return await query_llm_fn(p, s, h, images=imgs)
                except TypeError:
                    return await query_llm_fn(p, s, h)
        return await query_llm_fn(p, s, h)

    # Contextualize decision with recent history if available
    recent_history_context = ""
    if history:
        last_messages = history[-4:]
        formatted_history = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in last_messages])
        recent_history_context = f"Recent conversation context:\n{formatted_history}\n\n"

    decision_input = f"{recent_history_context}User prompt: {prompt}"

    if images:
        img_desc = "image" if len(images) == 1 else f"{len(images)} images"
        await _safe_update_status(status_callback, f"🖼️ Inspecting {img_desc} content...")

        decision_system = (
            "You are an AI assistant analyzing an image provided by the user.\n"
            "Examine the image carefully (including any visible text, headlines, claims, people, or data).\n"
            "Determine whether answering the user's prompt requires verifying claims or events with an external web search:\n"
            "- If NO web search is needed (e.g. describing the image, reading its text, identifying objects/animals/scenes, "
            "math/logic problems, general conversation, or the image itself provides all needed information), respond strictly with:\n"
            "NO_SEARCH\n"
            "- If web search IS needed (e.g. verifying whether a news headline, social media post, rumor, or claim shown in the image is true, "
            "or looking up recent real-world events referenced in the image), respond strictly with:\n"
            "SEARCH: <specific search query based on the claims/text extracted from the image>\n"
            "Do not provide any extra explanation, only NO_SEARCH or SEARCH: <query>."
        )
        decision_result = await _invoke_query_llm(decision_input, decision_system, [], images)
    else:
        await _safe_update_status(status_callback, "🔍 Checking if web search is needed...")

        decision_system = (
            "You are an AI decision router. Determine if answering the user's prompt requires "
            "real-time information, current events, recent data, or external web search.\n"
            "- If NO web search is needed (e.g. general knowledge, greetings, coding, math, "
            "creative writing, conversational replies), respond strictly with:\n"
            "NO_SEARCH\n"
            "- If web search IS needed (e.g. current events, recent news, live scores, weather, "
            "specifications of recently released products, or explicit user request to search), respond strictly with:\n"
            "SEARCH: <concise search query>\n"
            "Do not provide any explanation, only NO_SEARCH or SEARCH: <query>."
        )
        decision_result = await _invoke_query_llm(decision_input, decision_system, [])

    if decision_result and decision_result.startswith("⚠️"):
        return decision_result

    initial_query = None
    if decision_result:
        clean_decision = decision_result.strip()
        if "NO_SEARCH" in clean_decision.upper() and not clean_decision.upper().startswith("SEARCH:"):
            initial_query = None
        else:
            initial_query = _parse_search_query(clean_decision)
            if not initial_query and "SEARCH" in clean_decision.upper():
                # For text prompts, fallback to prompt if SEARCH keyword was outputted without format
                initial_query = prompt if not images else None

    # If no search is needed, generate standard response immediately
    if not initial_query:
        print("Decision: No web search needed. Proceeding to standard generation.")
        await _safe_update_status(status_callback, "⏳ Generating response...")
        return await _invoke_query_llm(prompt, system_prompt, history, images)

    print(f"Decision: Web search needed with initial query: '{initial_query}'")

    # Step 2: Iterative Search & Read Loop
    gathered_context: List[str] = []
    searched_queries = set()
    read_urls = set()
    current_action = f"SEARCH: {initial_query}"

    for step in range(max_search_steps):
        print(f"Web search loop step {step + 1}/{max_search_steps}: {current_action}")

        if current_action.upper().startswith("SEARCH:"):
            query = _parse_search_query(current_action)
            if not query or query in searched_queries:
                break
            searched_queries.add(query)

            await _safe_update_status(status_callback, f"🔍 Searching the web for: *{query}*...")
            results = await search_duckduckgo(query, max_results=5)

            if results:
                formatted_results = [f"Search results for \"{query}\":"]
                for idx, item in enumerate(results, 1):
                    formatted_results.append(
                        f"[{idx}] Title: {item.get('title')}\n"
                        f"    URL: {item.get('href')}\n"
                        f"    Snippet: {item.get('body')}"
                    )
                gathered_context.append("\n".join(formatted_results))
            else:
                gathered_context.append(f"Search for \"{query}\" returned no results.")

        elif current_action.upper().startswith("READ:"):
            url = _parse_read_url(current_action)
            if not url or url in read_urls:
                break
            read_urls.add(url)

            # Display truncated URL or domain in status
            display_url = url.split("://")[-1][:40]
            await _safe_update_status(status_callback, f"📖 Reading webpage: {display_url}...")
            content = await read_url_jina(url, max_chars=4000)
            gathered_context.append(f"Webpage content for <{url}>:\n{content}")

        # If this was the last allowed step, don't ask the model for another action
        if step == max_search_steps - 1:
            break

        # Check with model if further investigation (search / read) is needed or if ready
        agent_reasoning_system = (
            "You are a research assistant gathering information to answer a user prompt.\n"
            "Based on the research gathered so far, decide your NEXT action:\n"
            "1. To read full details from one of the retrieved URLs, respond with:\n"
            "READ: <url>\n"
            "2. To perform an additional search with a different query for missing facts, respond with:\n"
            "SEARCH: <different search query>\n"
            "3. If you have enough information to answer the user's prompt, respond with:\n"
            "READY\n\n"
            "Respond strictly with either READ: <url>, SEARCH: <query>, or READY."
        )

        research_summary = "\n\n---\n\n".join(gathered_context)
        reasoning_prompt = (
            f"User Prompt: {prompt}\n\n"
            f"Gathered Research:\n{research_summary}\n\n"
            "What is your next action? (READ: <url>, SEARCH: <query>, or READY)"
        )

        next_action_result = await query_llm_fn(reasoning_prompt, agent_reasoning_system, [])
        if not next_action_result:
            break
        if next_action_result.startswith("⚠️"):
            return next_action_result

        next_action_clean = next_action_result.strip()
        print(f"Model decided next action: {next_action_clean}")

        if "READY" in next_action_clean.upper():
            break
        elif next_action_clean.upper().startswith("READ:"):
            url = _parse_read_url(next_action_clean)
            if url and url not in read_urls:
                current_action = f"READ: {url}"
            else:
                break
        elif next_action_clean.upper().startswith("SEARCH:"):
            query = _parse_search_query(next_action_clean)
            if query and query not in searched_queries:
                current_action = f"SEARCH: {query}"
            else:
                break
        else:
            # If the model starts answering directly or returns unknown format, stop researching
            break

    # Step 3: Synthesis Phase
    await _safe_update_status(status_callback, "✍️ Generating final response...")

    research_summary = "\n\n---\n\n".join(gathered_context)
    augmented_system_prompt = (
        f"{system_prompt}\n\n"
        "You have access to the following real-time web search research to answer the user's prompt. "
        "Use this research to provide an accurate, up-to-date, and helpful response. "
        "Cite relevant sources or URLs if helpful, and adhere to your personality.\n\n"
        f"=== WEB SEARCH RESEARCH ===\n{research_summary}\n=== END RESEARCH ==="
    )

    final_response = await _invoke_query_llm(prompt, augmented_system_prompt, history, images)
    return final_response
