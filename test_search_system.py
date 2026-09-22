# test_search_system.py
import asyncio
import unittest
import json
from web_search import search_duckduckgo, read_url_jina
from search_agent import (
    run_search_augmented_generation,
    _parse_search_query,
    _parse_read_url,
    extract_website_name,
    extract_article_title,
    format_source_citation,
    format_response_sources,
)
import llm_client
from bot import LLMBot, setup, POPULAR_GEMINI_MODELS


class TestSearchUtilities(unittest.IsolatedAsyncioTestCase):

    async def test_duckduckgo_search(self):
        results = await search_duckduckgo("python programming", max_results=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        first = results[0]
        self.assertIn("title", first)
        self.assertIn("href", first)
        self.assertIn("body", first)
        self.assertTrue(first["href"].startswith("http"))

    async def test_jina_reader(self):
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        content = await read_url_jina(url, max_chars=500)
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)
        self.assertIn("Python", content)

    def test_parsers(self):
        self.assertEqual(_parse_search_query('SEARCH: "latest AI news"'), "latest AI news")
        self.assertEqual(_parse_search_query("search: quantum computing breakthroughs"), "quantum computing breakthroughs")
        self.assertIsNone(_parse_search_query("NO_SEARCH"))

        self.assertEqual(_parse_read_url("READ: https://example.com/article"), "https://example.com/article")
        self.assertEqual(_parse_read_url('READ: <https://example.com/test>'), "https://example.com/test")

    def test_source_extraction_and_formatting(self):
        # Known domains
        self.assertEqual(extract_website_name("https://en.wikipedia.org/wiki/Sideloading"), "Wikipedia")
        self.assertEqual(extract_website_name("https://www.theverge.com/news/123"), "The Verge")
        self.assertEqual(extract_website_name("https://github.com/torvalds/linux"), "GitHub")

        # Fallback to domain or title
        self.assertEqual(extract_website_name("https://example.org/path", "Some Article - MySite"), "MySite")
        self.assertEqual(extract_website_name("https://techblog.com/path"), "Techblog")

        # Article title extraction and cleaning
        self.assertEqual(extract_article_title("Sideloading - Wikipedia", "Wikipedia"), "Sideloading")
        self.assertEqual(extract_article_title("New Pixel Features | The Verge", "The Verge"), "New Pixel Features")
        self.assertEqual(extract_article_title("Python (programming language) - Wikipedia, the free encyclopedia", "Wikipedia"), "Python (programming language)")

        # Formatting citation (small subtext with -#)
        citation = format_source_citation("Wikipedia", "Sideloading", "https://en.wikipedia.org/wiki/Sideloading")
        self.assertEqual(citation, "-# Source: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)")

    def test_format_response_sources(self):
        sources_dict = {
            "https://en.wikipedia.org/wiki/Sideloading": {
                "website": "Wikipedia",
                "title": "Sideloading",
                "url": "https://en.wikipedia.org/wiki/Sideloading"
            }
        }
        read_sources = [sources_dict["https://en.wikipedia.org/wiki/Sideloading"]]

        # 1. Already formatted correctly with -#
        inp1 = "Sideloading is...\n\n-# Source: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)"
        self.assertEqual(format_response_sources(inp1, sources_dict, read_sources, True), inp1)

        # 2. Source line without -# should have -# added
        inp2_raw = "Sideloading is...\n\nSource: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)"
        self.assertEqual(format_response_sources(inp2_raw, sources_dict, read_sources, True), inp1)

        # 3. Markdown link without <> should have <> and -# added
        inp2 = "Sideloading is...\n\nSource: [Wikipedia - Sideloading](https://en.wikipedia.org/wiki/Sideloading)"
        exp2 = "Sideloading is...\n\n-# Source: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)"
        self.assertEqual(format_response_sources(inp2, sources_dict, read_sources, True), exp2)

        # 4. Plain text source formatted into markdown link with <> and -#
        inp3 = "Sideloading is...\n\nSource: Wikipedia - Sideloading: https://en.wikipedia.org/wiki/Sideloading"
        exp3 = "Sideloading is...\n\n-# Source: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)"
        self.assertEqual(format_response_sources(inp3, sources_dict, read_sources, True), exp3)

        # 5. Only URL on Source: line
        inp4 = "Sideloading is...\n\nSource: https://en.wikipedia.org/wiki/Sideloading"
        exp4 = "Sideloading is...\n\n-# Source: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)"
        self.assertEqual(format_response_sources(inp4, sources_dict, read_sources, True), exp4)

        # 6. Missing source line when web was used -> auto-appends with -#
        inp5 = "Sideloading is transferring files."
        res5 = format_response_sources(inp5, sources_dict, read_sources, True)
        self.assertTrue(res5.endswith("-# Source: [Wikipedia - Sideloading](<https://en.wikipedia.org/wiki/Sideloading>)"))

        # 7. Web was NOT used -> no source added
        inp6 = "Tell me a joke: Why did the chicken cross the road?"
        self.assertEqual(format_response_sources(inp6, sources_dict, read_sources, False), inp6)

    def test_prompt_name_resolution(self):
        bot = LLMBot()
        self.assertEqual(bot.get_prompt_name(bot.prompts["default"]), "default")
        self.assertEqual(bot.get_prompt_name(bot.prompts["pirate"]), "pirate")
        self.assertEqual(bot.get_prompt_name("You are a unique custom assistant."), "custom")


class TestSearchAgent(unittest.IsolatedAsyncioTestCase):

    async def test_agent_no_search_needed(self):
        status_updates = []

        async def mock_status(text: str):
            status_updates.append(text)

        async def mock_llm(prompt: str, system: str, history=None):
            if "AI decision router" in system:
                return "NO_SEARCH"
            return "General answer without search."

        response = await run_search_augmented_generation(
            prompt="Tell me a joke",
            system_prompt="You are a funny assistant.",
            history=[],
            query_llm_fn=mock_llm,
            status_callback=mock_status
        )

        self.assertEqual(response, "General answer without search.")
        self.assertTrue(any("Checking if web search is needed" in s for s in status_updates))
        self.assertTrue(any("Generating response" in s for s in status_updates))
        self.assertFalse(any("Searching the web" in s for s in status_updates))

    async def test_agent_with_search_and_read(self):
        status_updates = []

        async def mock_status(text: str):
            status_updates.append(text)

        call_count = {"count": 0}

        async def mock_llm(prompt: str, system: str, history=None):
            call_count["count"] += 1
            if "AI decision router" in system:
                return "SEARCH: latest python release features"
            elif "research assistant gathering information" in system:
                if call_count["count"] == 2:
                    return "READ: https://en.wikipedia.org/wiki/Python_(programming_language)"
                else:
                    return "READY"
            else:
                self.assertIn("WEB SEARCH RESEARCH", system)
                return "Python latest release information synthesized successfully."

        response = await run_search_augmented_generation(
            prompt="What is new in the latest Python release?",
            system_prompt="You are a helpful assistant.",
            history=[],
            query_llm_fn=mock_llm,
            status_callback=mock_status,
            max_search_steps=3
        )

        self.assertTrue(response.startswith("Python latest release information synthesized successfully."))
        self.assertIn("-# Source: [Wikipedia - Python (programming language)](<https://en.wikipedia.org/wiki/Python_(programming_language)>)", response)
        self.assertTrue(any("Searching the web for" in s for s in status_updates))
        self.assertTrue(any("Reading webpage" in s for s in status_updates))
        self.assertTrue(any("Generating final response" in s for s in status_updates))

    async def test_agent_error_propagation(self):
        async def mock_error_llm(prompt: str, system: str, history=None):
            return "⚠️ **Gemini API Error (Rate Limit/Quota)**: 429 ResourceExhausted quota exceeded."

        response = await run_search_augmented_generation(
            prompt="What happened today?",
            system_prompt="You are a helpful assistant.",
            history=[],
            query_llm_fn=mock_error_llm
        )

        self.assertTrue(response.startswith("⚠️ **Gemini API Error"))


class TestBotSetupAndSettings(unittest.IsolatedAsyncioTestCase):

    async def test_bot_initialization_and_settings(self):
        bot = LLMBot()
        self.assertIsNotNone(bot)
        self.assertTrue(hasattr(bot, "grounding_enabled"))
        self.assertTrue(hasattr(bot, "llm_provider"))
        self.assertTrue(hasattr(bot, "gemini_model"))
        self.assertTrue(hasattr(bot, "run_in_background"))
        self.assertTrue(hasattr(bot, "max_history"))
        self.assertTrue(hasattr(bot, "last_response_info"))

        # Verify settings types
        self.assertIn(bot.llm_provider, ["GEMINI", "LMSTUDIO"])
        self.assertIsInstance(bot.max_history, int)
        self.assertIsInstance(bot.run_in_background, bool)

        await setup(bot)
        commands = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        self.assertIn("grounding", commands)
        self.assertIn("websearch", commands)
        self.assertIn("provider", commands)
        self.assertIn("source", commands)
        self.assertIn("model", commands)
        self.assertIn("help", commands)
        self.assertIn("info", commands)
        self.assertNotIn("prompt", commands)
        self.assertIn("listprompts", commands)
        self.assertNotIn("showprompts", commands)
        self.assertIn("temp", commands)

        # Check default_permissions on admin commands
        self.assertTrue(commands["provider"].default_permissions.administrator)
        self.assertTrue(commands["source"].default_permissions.administrator)
        self.assertTrue(commands["model"].default_permissions.administrator)
        self.assertTrue(commands["temp"].default_permissions.administrator)

    async def test_info_command_output(self):
        from bot import info_command
        from unittest.mock import AsyncMock, MagicMock

        bot = LLMBot()
        bot.last_response_info = {
            "provider": "GEMINI",
            "model": "gemini-2.0-flash",
            "grounding_enabled": True,
            "grounding_used": True,
            "system_prompt": "You are a helpful pirate assistant.",
            "random_mode": False,
            "image_processing_used": True
        }

        interaction = MagicMock()
        interaction.client = bot
        interaction.response.send_message = AsyncMock()

        await info_command.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args[1]
        embed = call_kwargs["embed"]
        self.assertIsNotNone(embed)
        self.assertEqual(embed.title, "ℹ️ Response & Bot Information")

        fields = {f.name: f.value for f in embed.fields}
        self.assertIn("🤖 LLM Provider", fields)
        self.assertIn("Google Gemini", fields["🤖 LLM Provider"])
        self.assertIn("`GEMINI`", fields["🤖 LLM Provider"])
        self.assertEqual(fields["🧠 Model"], "`gemini-2.0-flash`")
        self.assertEqual(fields["🌡️ Temperature"], "`0.7`")
        self.assertEqual(fields["🌍 Grounding Setting"], "**Enabled**")
        self.assertEqual(fields["🔍 Grounding Used"], "**Yes**")
        self.assertEqual(fields["🎲 Random Mode"], "**Disabled**")
        self.assertEqual(fields["🖼️ Image Processing Used"], "**Yes**")
        prompt_field_val = next(v for k, v in fields.items() if "System Prompt" in k)
        self.assertIn("You are a helpful pirate assistant.", prompt_field_val)

    async def test_info_command_fallback_when_no_responses(self):
        from bot import info_command
        from unittest.mock import AsyncMock, MagicMock

        bot = LLMBot()
        bot.last_response_info = None

        interaction = MagicMock()
        interaction.client = bot
        interaction.response.send_message = AsyncMock()

        await info_command.callback(interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args[1]
        embed = call_kwargs["embed"]
        self.assertIn("No responses have been generated yet", embed.description)
        fields = {f.name: f.value for f in embed.fields}
        self.assertEqual(fields["🌡️ Temperature"], f"`{bot.temperature}`")
        self.assertEqual(fields["🔍 Grounding Used"], "*N/A (No responses yet)*")
        self.assertEqual(fields["🖼️ Image Processing Used"], "*N/A (No responses yet)*")

    async def test_altered_prompt_notice(self):
        from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

        bot = LLMBot()
        bot.random_mode = False
        bot.system_prompt = bot.prompts["pirate"]
        bot.prompt_name = "pirate"

        bot_user = MagicMock()
        bot_user.id = 12345
        mock_msg = MagicMock()
        mock_msg.author.name = "User"
        mock_msg.content = f"<@{bot_user.id}> Hello!"
        mock_msg.attachments = []
        mock_msg.reference = None
        mock_msg.channel.id = 999
        status_msg = AsyncMock()
        mock_msg.reply = AsyncMock(return_value=status_msg)

        with patch.object(LLMBot, "user", new_callable=PropertyMock, return_value=bot_user):
            with patch("bot.get_llm_response", new=AsyncMock(return_value="Ahoy matey!")):
                await bot.on_message(mock_msg)

            status_msg.edit.assert_called()
            last_edit_content = status_msg.edit.call_args[1]["content"]
            self.assertIn("Ahoy matey!", last_edit_content)
            self.assertIn("-# System prompt: pirate", last_edit_content)

            # Now test default prompt - no notice when random mode is False
            bot.system_prompt = bot.prompts["default"]
            bot.prompt_name = "default"
            with patch("bot.get_llm_response", new=AsyncMock(return_value="Hello, how can I help?")):
                await bot.on_message(mock_msg)

            last_edit_content2 = status_msg.edit.call_args[1]["content"]
            self.assertNotIn("System prompt:", last_edit_content2)

            # Test random mode enabled: should show '(random)' in notice even for default or pirate
            bot.random_mode = True
            with patch("bot.get_llm_response", new=AsyncMock(return_value="Random reply!")):
                await bot.on_message(mock_msg)

            last_edit_content3 = status_msg.edit.call_args[1]["content"]
            self.assertIn("-# System prompt:", last_edit_content3)
            self.assertIn("(random)", last_edit_content3)

    async def test_temp_command(self):
        from bot import temp_command
        from unittest.mock import AsyncMock, MagicMock, patch

        bot = LLMBot()
        interaction = MagicMock()
        interaction.client = bot
        interaction.guild = MagicMock()
        interaction.user.guild_permissions.administrator = True
        interaction.response.send_message = AsyncMock()

        # Test valid temperature
        with patch.object(bot, "save_settings") as mock_save:
            await temp_command.callback(interaction, 1.2)
            self.assertEqual(bot.temperature, 1.2)
            mock_save.assert_called_once()
            interaction.response.send_message.assert_called_with("✅ Temperature updated to **1.2**.", ephemeral=True)

        # Test out of range temperature (too high)
        interaction.response.send_message.reset_mock()
        await temp_command.callback(interaction, 2.5)
        interaction.response.send_message.assert_called_with("❌ Temperature must be between 0.0 and 2.0.", ephemeral=True)

        # Test out of range temperature (negative)
        interaction.response.send_message.reset_mock()
        await temp_command.callback(interaction, -0.1)
        interaction.response.send_message.assert_called_with("❌ Temperature must be between 0.0 and 2.0.", ephemeral=True)

        # Test non-admin user
        interaction.user.guild_permissions.administrator = False
        interaction.response.send_message.reset_mock()
        await temp_command.callback(interaction, 0.8)
        interaction.response.send_message.assert_called_with("❌ You do not have permission to use this command (Administrator required).", ephemeral=True)

    async def test_listprompts_command(self):
        from bot import list_prompts
        from unittest.mock import AsyncMock, MagicMock

        bot = LLMBot()
        interaction = MagicMock()
        interaction.client = bot
        interaction.response.send_message = AsyncMock()

        await list_prompts.callback(interaction)
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args[1]
        embed = call_kwargs["embed"]
        self.assertEqual(embed.title, "Available System Prompts")
        self.assertGreater(len(embed.fields), 0)

    def test_message_chunking(self):
        long_response = "A" * 4500
        parts = [long_response[i:i+2000] for i in range(0, len(long_response), 2000)]
        self.assertEqual(len(parts), 3)
        self.assertEqual(len(parts[0]), 2000)
        self.assertEqual(len(parts[1]), 2000)
        self.assertEqual(len(parts[2]), 500)


if __name__ == "__main__":
    unittest.main()
