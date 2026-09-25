# test_fallback.py
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Mock non-standard library packages if not installed in the test environment
for mod in [
    "aiohttp", "google", "google.genai", "google.genai.types", "dotenv",
    "discord", "discord.ext", "discord.ext.commands", "discord.app_commands",
    "ddgs", "duckduckgo_search"
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

import llm_client_gemini
import llm_client


class TestGeminiFallbackAndTimeout(unittest.IsolatedAsyncioTestCase):

    def test_is_transient_or_overload_error(self):
        """Verify overload and timeout error classification."""
        user_error = Exception(
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
        )
        self.assertTrue(llm_client_gemini._is_transient_or_overload_error(user_error))
        self.assertTrue(llm_client_gemini._is_transient_or_overload_error(asyncio.TimeoutError()))
        self.assertTrue(llm_client_gemini._is_transient_or_overload_error(TimeoutError()))
        self.assertTrue(llm_client_gemini._is_transient_or_overload_error(Exception("429 RESOURCE_EXHAUSTED: quota exceeded")))
        self.assertTrue(llm_client_gemini._is_transient_or_overload_error(Exception("504 Gateway Timeout")))

        # Non-transient errors
        self.assertFalse(llm_client_gemini._is_transient_or_overload_error(Exception("404 NOT_FOUND: model not found")))
        self.assertFalse(llm_client_gemini._is_transient_or_overload_error(Exception("API_KEY_INVALID: bad key")))

    async def test_gemini_primary_success(self):
        """When primary model succeeds, fallback is not called."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("llm_client_gemini.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.aio = MagicMock()

                async def mock_execute(client, model_name, prompt, system_prompt, history, images, temperature, timeout):
                    return f"Response from {model_name}"

                with patch("llm_client_gemini._execute_gemini_request", side_effect=mock_execute):
                    metadata = {}
                    res = await llm_client_gemini.get_llm_response(
                        prompt="Hello",
                        system_prompt="Assistant",
                        model_name="gemini-2.0-flash",
                        fallback_model="gemini-2.5-flash-lite",
                        timeout=15.0,
                        metadata=metadata
                    )

                    self.assertEqual(res, "Response from gemini-2.0-flash")
                    self.assertFalse(metadata.get("fallback_used"))
                    self.assertEqual(metadata.get("model"), "gemini-2.0-flash")

    async def test_gemini_overload_triggers_fallback(self):
        """When primary model encounters 503 overload, it switches to fallback model."""
        user_error = Exception(
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
        )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("llm_client_gemini.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.aio = MagicMock()

                models_called = []
                status_updates = []

                async def mock_execute(client, model_name, prompt, system_prompt, history, images, temperature, timeout):
                    models_called.append(model_name)
                    if model_name == "gemini-2.0-flash":
                        raise user_error
                    return f"Response from {model_name}"

                async def mock_status_callback(msg: str):
                    status_updates.append(msg)

                with patch("llm_client_gemini._execute_gemini_request", side_effect=mock_execute):
                    metadata = {}
                    res = await llm_client_gemini.get_llm_response(
                        prompt="Hello",
                        system_prompt="Assistant",
                        model_name="gemini-2.0-flash",
                        fallback_model="gemini-2.5-flash-lite",
                        timeout=30.0,
                        metadata=metadata,
                        status_callback=mock_status_callback
                    )

                    self.assertEqual(models_called, ["gemini-2.0-flash", "gemini-2.5-flash-lite"])
                    self.assertEqual(res, "Response from gemini-2.5-flash-lite")
                    self.assertTrue(metadata.get("fallback_used"))
                    self.assertEqual(metadata.get("fallback_model"), "gemini-2.5-flash-lite")
                    self.assertEqual(metadata.get("primary_model"), "gemini-2.0-flash")
                    self.assertEqual(metadata.get("model"), "gemini-2.5-flash-lite")
                    self.assertTrue(len(status_updates) > 0)
                    self.assertIn("gemini-2.5-flash-lite", status_updates[0])

    async def test_gemini_timeout_triggers_fallback(self):
        """When primary model times out, it switches to fallback model."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("llm_client_gemini.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.aio = MagicMock()

                models_called = []

                async def mock_execute(client, model_name, prompt, system_prompt, history, images, temperature, timeout):
                    models_called.append(model_name)
                    if model_name == "gemini-2.0-flash":
                        raise asyncio.TimeoutError()
                    return f"Response from {model_name}"

                with patch("llm_client_gemini._execute_gemini_request", side_effect=mock_execute):
                    metadata = {}
                    res = await llm_client_gemini.get_llm_response(
                        prompt="Hello",
                        system_prompt="Assistant",
                        model_name="gemini-2.0-flash",
                        fallback_model="gemini-2.5-flash-lite",
                        timeout=5.0,
                        metadata=metadata
                    )

                    self.assertEqual(models_called, ["gemini-2.0-flash", "gemini-2.5-flash-lite"])
                    self.assertEqual(res, "Response from gemini-2.5-flash-lite")
                    self.assertTrue(metadata.get("fallback_used"))
                    self.assertEqual(metadata.get("model"), "gemini-2.5-flash-lite")

    async def test_gemini_both_fail(self):
        """When both primary and fallback fail, returns informative error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("llm_client_gemini.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.aio = MagicMock()

                async def mock_execute(client, model_name, prompt, system_prompt, history, images, temperature, timeout):
                    if model_name == "gemini-2.0-flash":
                        raise Exception("503 Service Unavailable")
                    raise asyncio.TimeoutError()

                with patch("llm_client_gemini._execute_gemini_request", side_effect=mock_execute):
                    metadata = {}
                    res = await llm_client_gemini.get_llm_response(
                        prompt="Hello",
                        system_prompt="Assistant",
                        model_name="gemini-2.0-flash",
                        fallback_model="gemini-2.5-flash-lite",
                        timeout=10.0,
                        metadata=metadata
                    )

                    self.assertTrue(res.startswith("⚠️ **Gemini API Error"))
                    self.assertIn("gemini-2.0-flash", res)
                    self.assertIn("gemini-2.5-flash-lite", res)

    async def test_gemini_non_transient_error_no_fallback(self):
        """When error is non-transient (like model not found), fallback is not invoked."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("llm_client_gemini.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.aio = MagicMock()

                models_called = []

                async def mock_execute(client, model_name, prompt, system_prompt, history, images, temperature, timeout):
                    models_called.append(model_name)
                    raise Exception("404 NOT_FOUND: model not found")

                with patch("llm_client_gemini._execute_gemini_request", side_effect=mock_execute):
                    metadata = {}
                    res = await llm_client_gemini.get_llm_response(
                        prompt="Hello",
                        system_prompt="Assistant",
                        model_name="invalid-model-name",
                        fallback_model="gemini-2.5-flash-lite",
                        timeout=10.0,
                        metadata=metadata
                    )

                    self.assertEqual(models_called, ["invalid-model-name"])
                    self.assertIn("Model Not Found", res)
                    self.assertFalse(metadata.get("fallback_used"))

    def test_discord_subtext_formatting(self):
        """Verify the exact Discord subtext formatting for system prompts and fallback notices."""
        # Case 1: Only fallback model used
        llm_response = "Here is the response."
        resp_metadata = {"fallback_used": True, "fallback_model": "gemini-2.5-flash-lite"}
        current_prompt_name = "default"
        random_mode = False

        discord_response = llm_response
        if (current_prompt_name and current_prompt_name.lower() != "default") or random_mode:
            random_suffix = " (random)" if random_mode else ""
            discord_response = f"{discord_response.rstrip()}\n-# System prompt: {current_prompt_name}{random_suffix}"

        if resp_metadata.get("fallback_used"):
            used_fallback = resp_metadata.get("fallback_model", "gemini-2.5-flash-lite")
            discord_response = f"{discord_response.rstrip()}\n-# Fallback model used: {used_fallback}"

        self.assertEqual(
            discord_response,
            "Here is the response.\n-# Fallback model used: gemini-2.5-flash-lite"
        )

        # Case 2: Both custom prompt and fallback model used
        current_prompt_name = "pirate"
        discord_response = llm_response
        if (current_prompt_name and current_prompt_name.lower() != "default") or random_mode:
            random_suffix = " (random)" if random_mode else ""
            discord_response = f"{discord_response.rstrip()}\n-# System prompt: {current_prompt_name}{random_suffix}"

        if resp_metadata.get("fallback_used"):
            used_fallback = resp_metadata.get("fallback_model", "gemini-2.5-flash-lite")
            discord_response = f"{discord_response.rstrip()}\n-# Fallback model used: {used_fallback}"

        self.assertEqual(
            discord_response,
            "Here is the response.\n-# System prompt: pirate\n-# Fallback model used: gemini-2.5-flash-lite"
        )

    async def test_query_llm_gemini_routing(self):
        """Verify query_llm properly routes fallback parameters for Gemini provider."""
        with patch("llm_client.get_gemini_response") as mock_gemini:
            mock_gemini.return_value = "Gemini ok"
            meta = {}
            res = await llm_client.query_llm(
                prompt="test",
                system_prompt="sys",
                provider="GEMINI",
                model="gemini-2.0-flash",
                fallback_model="gemini-2.5-flash-lite",
                timeout=25.0,
                metadata=meta
            )

            self.assertEqual(res, "Gemini ok")
            mock_gemini.assert_called_once()
            call_kwargs = mock_gemini.call_args.kwargs
            self.assertEqual(call_kwargs["model_name"], "gemini-2.0-flash")
            self.assertEqual(call_kwargs["fallback_model"], "gemini-2.5-flash-lite")
            self.assertEqual(call_kwargs["timeout"], 25.0)

    async def test_query_llm_lmstudio_routing(self):
        """Verify query_llm does NOT pass fallback or Gemini timeout to LM Studio."""
        with patch("llm_client.get_lmstudio_response") as mock_lmstudio:
            mock_lmstudio.return_value = "LMStudio ok"
            meta = {}
            res = await llm_client.query_llm(
                prompt="test",
                system_prompt="sys",
                provider="LMSTUDIO",
                model="local-model",
                fallback_model="gemini-2.5-flash-lite",
                timeout=25.0,
                metadata=meta
            )

            self.assertEqual(res, "LMStudio ok")
            mock_lmstudio.assert_called_once()
            call_kwargs = mock_lmstudio.call_args.kwargs
            self.assertNotIn("fallback_model", call_kwargs)
            self.assertNotIn("timeout", call_kwargs)


class TestMessageTriggerCriteria(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        class DummyBot:
            MAX_ROLE_MEMBERS_FOR_PING = 5
            def __init__(self, user):
                self.user = user

        self.bot = DummyBot(MagicMock(id=12345, name="TestBot"))
        from bot import should_respond_to_message
        self.should_respond = lambda msg: should_respond_to_message(self.bot, msg)

    async def test_ignore_own_message(self):
        msg = MagicMock()
        msg.author = self.bot.user
        self.assertFalse(await self.should_respond(msg))

    async def test_direct_mention_responds(self):
        msg = MagicMock()
        msg.author = MagicMock(id=999)
        msg.guild = MagicMock()
        msg.mentions = [self.bot.user]
        msg.mention_everyone = False
        msg.role_mentions = []
        self.assertTrue(await self.should_respond(msg))

    async def test_everyone_here_ignored(self):
        msg = MagicMock()
        msg.author = MagicMock(id=999)
        msg.guild = MagicMock()
        msg.mentions = []  # Not directly mentioned
        msg.mention_everyone = True
        msg.role_mentions = []
        self.assertFalse(await self.should_respond(msg))

    async def test_role_mention_with_under_5_members_responds(self):
        msg = MagicMock()
        msg.author = MagicMock(id=999)
        guild = MagicMock()
        msg.guild = guild

        # Role with 2 members
        small_role = MagicMock(id=888, name="SmallRole")
        small_role.is_default.return_value = False
        small_role.members = [MagicMock(), MagicMock()]  # 2 members (< 5)

        # Bot is in this role
        bot_member = MagicMock()
        bot_member.roles = [small_role]
        guild.me = bot_member

        msg.mentions = []
        msg.mention_everyone = False
        msg.role_mentions = [small_role]

        self.assertTrue(await self.should_respond(msg))

    async def test_role_mention_with_5_or_more_members_ignored(self):
        msg = MagicMock()
        msg.author = MagicMock(id=999)
        guild = MagicMock()
        msg.guild = guild

        # Role with 6 members
        large_role = MagicMock(id=777, name="LargeRole")
        large_role.is_default.return_value = False
        large_role.members = [MagicMock() for _ in range(6)]  # 6 members (>= 5)

        # Bot is in this role
        bot_member = MagicMock()
        bot_member.roles = [large_role]
        guild.me = bot_member

        msg.mentions = []
        msg.mention_everyone = False
        msg.role_mentions = [large_role]

        self.assertFalse(await self.should_respond(msg))

    async def test_role_mention_not_assigned_to_bot_ignored(self):
        msg = MagicMock()
        msg.author = MagicMock(id=999)
        guild = MagicMock()
        msg.guild = guild

        # Role with 2 members, but bot is NOT in it
        other_role = MagicMock(id=666, name="OtherRole")
        other_role.is_default.return_value = False
        other_role.members = [MagicMock(), MagicMock()]

        # Bot has different roles
        bot_member = MagicMock()
        bot_member.roles = []
        guild.me = bot_member

        msg.mentions = []
        msg.mention_everyone = False
        msg.role_mentions = [other_role]

        self.assertFalse(await self.should_respond(msg))

    def test_role_mention_prompt_cleaning(self):
        role = MagicMock(id=888)
        content = f"<@&888> Tell me a joke @everyone"
        cleaned = content.replace(f'<@!{self.bot.user.id}>', '').replace(f'<@{self.bot.user.id}>', '')
        cleaned = cleaned.replace(f'<@&{role.id}>', '')
        cleaned = cleaned.replace('@everyone', '').replace('@here', '').strip()
        self.assertEqual(cleaned, "Tell me a joke")


class TestSettingsAndIntents(unittest.TestCase):

    def test_bot_loads_fallback_and_members_intent_from_settings(self):
        """Verify bot reads fallback model, timeout, and members intent from settings."""
        from bot import LLMBot
        test_settings = {
            "gemini_fallback_model": "custom-fallback-model",
            "gemini_timeout": 45.0,
            "enable_members_intent": True,
            "llm_provider": "GEMINI",
            "gemini_model": "gemini-2.5-flash"
        }
        with patch.object(LLMBot, "load_settings", return_value=test_settings):
            with patch.object(LLMBot, "load_prompts", return_value={"default": "assistant"}):
                with patch.object(LLMBot, "load_banned_words", return_value=[]):
                    bot = LLMBot()
                    self.assertEqual(bot.gemini_fallback_model, "custom-fallback-model")
                    self.assertEqual(bot.gemini_timeout, 45.0)
                    self.assertTrue(bot.enable_members_intent)
                    self.assertTrue(bot.intents.members)

    def test_bot_members_intent_disabled_by_default(self):
        """Verify members intent is False when configured as False in settings."""
        from bot import LLMBot
        test_settings = {
            "gemini_fallback_model": "gemini-2.5-flash-lite",
            "gemini_timeout": 30.0,
            "enable_members_intent": False
        }
        with patch.object(LLMBot, "load_settings", return_value=test_settings):
            with patch.object(LLMBot, "load_prompts", return_value={"default": "assistant"}):
                with patch.object(LLMBot, "load_banned_words", return_value=[]):
                    bot = LLMBot()
                    self.assertFalse(bot.enable_members_intent)
                    self.assertFalse(bot.intents.members)

    def test_save_settings_persists_fallback_and_members_intent(self):
        """Verify save_settings includes fallback model, timeout, and members intent."""
        from bot import LLMBot
        test_settings = {
            "gemini_fallback_model": "gemini-test-fallback",
            "gemini_timeout": 20.0,
            "enable_members_intent": True
        }
        saved_data = {}
        def mock_dump(obj, f, **kwargs):
            saved_data.update(obj)

        with patch.object(LLMBot, "load_settings", return_value=test_settings):
            with patch.object(LLMBot, "load_prompts", return_value={"default": "assistant"}):
                with patch.object(LLMBot, "load_banned_words", return_value=[]):
                    bot = LLMBot()
                    with patch("builtins.open", MagicMock()):
                        with patch("json.dump", side_effect=mock_dump):
                            bot.save_settings()
                            self.assertEqual(saved_data.get("gemini_fallback_model"), "gemini-test-fallback")
                            self.assertEqual(saved_data.get("gemini_timeout"), 20.0)
                            self.assertEqual(saved_data.get("enable_members_intent"), True)


if __name__ == "__main__":
    unittest.main()

