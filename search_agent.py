import re
import urllib.parse
from typing import List, Dict, Callable, Awaitable, Optional, Any
from web_search import search_duckduckgo, read_url_jina


KNOWN_DOMAINS = {
    "wikipedia.org": "Wikipedia",
    "github.com": "GitHub",
    "youtube.com": "YouTube",
    "reddit.com": "Reddit",
    "twitter.com": "X (Twitter)",
    "x.com": "X (Twitter)",
    "theverge.com": "The Verge",
    "arstechnica.com": "Ars Technica",
    "techcrunch.com": "TechCrunch",
    "wired.com": "Wired",
    "forbes.com": "Forbes",
    "bloomberg.com": "Bloomberg",
    "reuters.com": "Reuters",
    "nytimes.com": "The New York Times",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "cnn.com": "CNN",
    "theguardian.com": "The Guardian",
    "medium.com": "Medium",
    "stackoverflow.com": "Stack Overflow",
    "fandom.com": "Fandom",
    "ign.com": "IGN",
    "gamespot.com": "GameSpot",
    "androidcentral.com": "Android Central",
    "androidauthority.com": "Android Authority",
    "9to5mac.com": "9to5Mac",
    "9to5google.com": "9to5Google",
    "macrumors.com": "MacRumors",
    "tomshardware.com": "Tom's Hardware",
    "pcgamer.com": "PC Gamer",
    "polygon.com": "Polygon",
    "kotaku.com": "Kotaku",
    "imdb.com": "IMDb",
    "wsj.com": "The Wall Street Journal",
    "washingtonpost.com": "The Washington Post",
    "apnews.com": "AP News",
    "npr.org": "NPR",
    "espn.com": "ESPN",
    "cnet.com": "CNET",
    "zdnet.com": "ZDNet",
    "support.google.com": "Google Support",
    "google.com": "Google",
    "microsoft.com": "Microsoft",
    "apple.com": "Apple",
    "developer.mozilla.org": "MDN Web Docs",
    "python.org": "Python.org",
    "docs.python.org": "Python Docs",
    "w3schools.com": "W3Schools",
    "geeksforgeeks.org": "GeeksforGeeks",
}


def extract_website_name(url: str, title: str = "") -> str:
    """Extracts a human-readable website name from a URL and optional page title."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        for domain_key, name in KNOWN_DOMAINS.items():
            if netloc == domain_key or netloc.endswith("." + domain_key):
                return name

        if title and (" - " in title or " | " in title):
            sep = " - " if " - " in title else " | "
            parts = title.split(sep)
            possible_site = parts[-1].strip()
            if 0 < len(possible_site) < 30 and not possible_site.startswith("http"):
                return possible_site

        parts = netloc.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return netloc.capitalize()
    except Exception:
        return "Web"


def extract_article_title(raw_title: str, website_name: str = "") -> str:
    """Extracts and cleans article title, removing duplicate website names or generic boilerplates."""
    if not raw_title:
        return "Article"
    title = raw_title.strip()

    if website_name:
        pattern = rf"\s*[-|–]\s*{re.escape(website_name)}.*$"
        title = re.sub(pattern, "", title, flags=re.IGNORECASE).strip()
        pattern_prefix = rf"^{re.escape(website_name)}\s*[-|–:]\s*"
        title = re.sub(pattern_prefix, "", title, flags=re.IGNORECASE).strip()

    title = re.sub(r"\s*[-|–]\s*Wikipedia, the free encyclopedia.*$", "", title, flags=re.IGNORECASE).strip()
    title = title.strip(" -|–:")
    return title or raw_title.strip()


def format_source_citation(website: str, title: str, url: str) -> str:
    """Formats a source citation as '-# Source: [Website - Title](<URL>)' for Discord (smaller subtext)."""
    clean_url = url.strip().strip("<>").rstrip(".,;")
    return f"-# Source: [{website} - {title}](<{clean_url}>)"


def format_response_sources(
    response: str,
    sources_dict: Dict[str, Dict[str, str]],
    read_sources: List[Dict[str, str]],
    web_used: bool
) -> str:
    """
    Ensures that any cited website source in the response follows the format:
    -# Source: [<Website> - <Article Title>](<<URL>>)
    and appends a formatted citation if the bot gathered web info but omitted the citation.
    """
    if not web_used or not response:
        return response

    # 1. Clean any existing markdown links that might be missing <> or have double <<>>
    # Example: [Text](https://...) -> [Text](<https://...>)
    response = re.sub(r"\[([^\]]+)\]\(<{0,2}(https?://[^>\s)]+?)>{0,2}\)", r"[\1](<\2>)", response)

    # 2. Reformat plain text "Source: Website - Title http(s)://..." or "Source: Website - Title (<http...>)"
    def _reformat_plain_source(match):
        body = match.group(1).strip()
        url = match.group(2).strip().rstrip(".,;>")

        if not body or body.startswith("http") or "<website>" in body.lower():
            info = sources_dict.get(url)
            if info:
                link_text = f"{info['website']} - {info['title']}"
            else:
                site = extract_website_name(url)
                art = extract_article_title("", site)
                link_text = f"{site} - {art}"
        else:
            link_text = body.strip(" -:–[]()<>\"'")

        return f"-# Source: [{link_text}](<{url}>)"

    plain_source_regex = re.compile(
        r"^(?:(?:-#\s*)?Source:\s*)(?!\[)(.*?)(?:[:\-–]\s*|\s+)\(?<{0,2}(https?://[^\s>)]+?)>{0,2}\)?\s*$",
        re.MULTILINE | re.IGNORECASE
    )
    response = plain_source_regex.sub(_reformat_plain_source, response)

    # 3. Handle lines with only "Source: https://..." or "Source: <https://...>"
    only_url_regex = re.compile(r"^(?:(?:-#\s*)?Source:\s*)<{0,2}(https?://[^\s>)]+?)>{0,2}\s*$", re.MULTILINE | re.IGNORECASE)
    def _reformat_only_url(match):
        url = match.group(1).strip().rstrip(".,;>")
        info = sources_dict.get(url)
        link_text = f"{info['website']} - {info['title']}" if info else f"{extract_website_name(url)} - {extract_article_title('', extract_website_name(url))}"
        return f"-# Source: [{link_text}](<{url}>)"
    response = only_url_regex.sub(_reformat_only_url, response)

    # 4. Ensure any line starting with "Source: [" has "-# " prefix for small subtext
    response = re.sub(
        r"^(?!\s*-#\s*)Source:\s*(\[.+?\]\(<https?://.+?>\))",
        r"-# Source: \1",
        response,
        flags=re.MULTILINE | re.IGNORECASE
    )

    # 5. Check if any formatted Source: line is present in the response
    has_source = bool(re.search(r"^\s*(?:-#\s*)?Source:\s*\[.+?\]\(<https?://.+?>\)", response, re.MULTILINE | re.IGNORECASE))

    if not has_source:
        # Web was used, but the model omitted the source line - append the primary source(s)
        primary_sources = read_sources if read_sources else list(sources_dict.values())[:1]
        if primary_sources:
            lines_to_add = []
            seen_urls = set()
            for s in primary_sources:
                u = s.get("url")
                if u and u not in seen_urls:
                    seen_urls.add(u)
                    site = s.get("website") or extract_website_name(u)
                    art = s.get("title") or extract_article_title("", site)
                    lines_to_add.append(format_source_citation(site, art, u))
            if lines_to_add:
                response = response.rstrip() + "\n\n" + "\n".join(lines_to_add)

    return response


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
    images: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
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
        if metadata is not None:
            metadata["grounding_used"] = False
        await _safe_update_status(status_callback, "⏳ Generating response...")
        return await _invoke_query_llm(prompt, system_prompt, history, images)

    print(f"Decision: Web search needed with initial query: '{initial_query}'")

    # Step 2: Iterative Search & Read Loop
    gathered_context: List[str] = []
    searched_queries = set()
    read_urls = set()
    sources_dict: Dict[str, Dict[str, str]] = {}
    read_sources: List[Dict[str, str]] = []
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
                    href = item.get("href", "")
                    raw_title = item.get("title", "")
                    site_name = extract_website_name(href, raw_title)
                    art_title = extract_article_title(raw_title, site_name)
                    if href:
                        sources_dict[href] = {
                            "website": site_name,
                            "title": art_title,
                            "url": href
                        }
                    formatted_results.append(
                        f"[{idx}] Website: {site_name}\n"
                        f"    Title: {art_title}\n"
                        f"    URL: {href}\n"
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

            # Extract title if provided in Jina markdown
            jina_title = ""
            title_match = re.search(r"^Title:\s*(.+)$", content, re.MULTILINE)
            if title_match:
                jina_title = title_match.group(1).strip()

            existing = sources_dict.get(url, {})
            fallback_title = existing.get("title", "")
            raw_title = jina_title or fallback_title or ""
            site_name = extract_website_name(url, raw_title)
            art_title = extract_article_title(raw_title, site_name)
            source_info = {
                "website": site_name,
                "title": art_title,
                "url": url
            }
            sources_dict[url] = source_info
            read_sources.append(source_info)

            gathered_context.append(
                f"Webpage content for Website: {site_name} | Title: {art_title} | URL: <{url}>:\n{content}"
            )

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
        "Adhere to your personality.\n\n"
        "FORMATTING REQUIREMENT FOR SOURCES:\n"
        "When you use information from a website, you MUST include a formatted source attribution at the very end of your response in this exact format:\n"
        "-# Source: [Website - Article Title](<URL>)\n\n"
        "Example:\n"
        "-# Source: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)\n\n"
        "Rules:\n"
        "1. The line MUST begin with '-# Source: ' so Discord displays it as smaller subtext.\n"
        "2. Clicking the 'Website - Article Title' part should link to the website.\n"
        "3. The URL inside the parentheses MUST be enclosed in angle brackets <> (e.g. (<https://...>)) so Discord does not create an embed preview.\n"
        "4. If multiple websites were used for information, include each source on a separate line starting with '-# Source: '.\n"
        "5. Only cite URLs from the provided research.\n\n"
        f"=== WEB SEARCH RESEARCH ===\n{research_summary}\n=== END RESEARCH ==="
    )

    final_response = await _invoke_query_llm(prompt, augmented_system_prompt, history, images)
    if not final_response or final_response.startswith("⚠️"):
        return final_response

    if metadata is not None:
        metadata["grounding_used"] = bool(gathered_context)

    # Post-process to ensure formatting of "Source: [Website - Article Title](<URL>)"
    final_response = format_response_sources(
        final_response,
        sources_dict=sources_dict,
        read_sources=read_sources,
        web_used=bool(gathered_context)
    )
    return final_response
