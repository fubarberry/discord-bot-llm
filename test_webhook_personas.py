import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import discord
from bot import LLMBot, reload_prompts_command, list_prompts


class TestWebhookPersonas(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Create a bot instance with minimal mock settings
        with patch.object(LLMBot, "load_settings", return_value={"prompt_name": "default"}), \
             patch.object(LLMBot, "load_banned_words", return_value=[]):
            self.bot = LLMBot()
            mock_user = MagicMock()
            mock_user.id = 12345
            mock_user.name = "AIBot"
            mock_user.display_name = "AI Assistant"
            mock_user.display_avatar = MagicMock()
            mock_user.display_avatar.url = "https://example.com/bot-default-avatar.png"
            self.bot._connection.user = mock_user

    def test_load_prompts_legacy_and_dict_formats(self):
        """Verify load_prompts handles both string format and dict format with avatar_url and username."""
        raw_mock = {
            "legacy": "You are a legacy assistant.",
            "pirate": {
                "prompt": "Arrr! You are a pirate.",
                "avatar_url": "https://example.com/pirate.png",
                "username": "Cap'n Jack"
            },
            "tsundere": {
                "text": "It's not like I like you or anything!",
                "pfp": "https://example.com/tsundere.png"
            },
            "empty_avatar": {
                "system_prompt": "Helpful prompt.",
                "avatar_url": "   "
            }
        }
        with patch.object(LLMBot, "load_prompts", return_value=raw_mock):
            self.bot.reload_prompts()

        # Check self.prompts
        self.assertEqual(self.bot.prompts["legacy"], "You are a legacy assistant.")
        self.assertEqual(self.bot.prompts["pirate"], "Arrr! You are a pirate.")
        self.assertEqual(self.bot.prompts["tsundere"], "It's not like I like you or anything!")
        self.assertEqual(self.bot.prompts["empty_avatar"], "Helpful prompt.")

        # Check self.prompt_profiles
        self.assertEqual(self.bot.get_prompt_profile("legacy"), {"avatar_url": None, "username": None})
        self.assertEqual(self.bot.get_prompt_profile("pirate"), {
            "avatar_url": "https://example.com/pirate.png",
            "username": "Cap'n Jack"
        })
        self.assertEqual(self.bot.get_prompt_profile("tsundere"), {
            "avatar_url": "https://example.com/tsundere.png",
            "username": None
        })
        self.assertEqual(self.bot.get_prompt_profile("empty_avatar"), {
            "avatar_url": None,
            "username": None
        })
        self.assertEqual(self.bot.get_prompt_profile("nonexistent"), {
            "avatar_url": None,
            "username": None
        })

    async def test_get_webhook_for_dm_channel(self):
        """Verify DMs return None because webhooks are unsupported in private channels."""
        dm_channel = MagicMock(spec=discord.DMChannel)
        result = await self.bot.get_webhook_for_channel(dm_channel)
        self.assertIsNone(result)

    async def test_get_webhook_for_guild_channel_without_permission(self):
        """Verify channels without 'Manage Webhooks' permission return None."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.guild = MagicMock()
        me = MagicMock()
        channel.guild.me = me
        perms = MagicMock()
        perms.manage_webhooks = False
        channel.permissions_for.return_value = perms

        result = await self.bot.get_webhook_for_channel(channel)
        self.assertIsNone(result)

    async def test_get_webhook_creates_new_webhook_if_none_exists(self):
        """Verify a new webhook is created when one does not exist and is cached."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 1001
        channel.name = "general"
        channel.guild = MagicMock()
        channel.guild.me = MagicMock()
        perms = MagicMock()
        perms.manage_webhooks = True
        channel.permissions_for.return_value = perms

        channel.webhooks = AsyncMock(return_value=[])
        new_webhook = MagicMock(spec=discord.Webhook)
        new_webhook.token = "token123"
        channel.create_webhook = AsyncMock(return_value=new_webhook)

        wh, thread = await self.bot.get_webhook_for_channel(channel)
        self.assertEqual(wh, new_webhook)
        self.assertIsNone(thread)
        channel.create_webhook.assert_called_once()
        self.assertIn(1001, self.bot._webhook_cache)

        # Second call should use cache without calling create_webhook again
        wh_cached, _ = await self.bot.get_webhook_for_channel(channel)
        self.assertEqual(wh_cached, new_webhook)
        self.assertEqual(channel.create_webhook.call_count, 1)

    async def test_get_webhook_reuses_existing_bot_webhook(self):
        """Verify an existing webhook created by the bot is reused."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 2002
        channel.name = "general"
        channel.guild = MagicMock()
        channel.guild.me = MagicMock()
        perms = MagicMock()
        perms.manage_webhooks = True
        channel.permissions_for.return_value = perms

        existing_wh = MagicMock(spec=discord.Webhook)
        existing_wh.token = "token456"
        existing_wh.user = self.bot.user
        channel.webhooks = AsyncMock(return_value=[existing_wh])
        channel.create_webhook = AsyncMock()

        wh, thread = await self.bot.get_webhook_for_channel(channel)
        self.assertEqual(wh, existing_wh)
        self.assertIsNone(thread)
        channel.create_webhook.assert_not_called()

    async def test_get_webhook_for_thread(self):
        """Verify webhooks in threads fetch the parent channel and return the thread."""
        parent_channel = MagicMock(spec=discord.TextChannel)
        parent_channel.id = 3003
        parent_channel.name = "parent-general"
        parent_channel.guild = MagicMock()
        parent_channel.guild.me = MagicMock()
        perms = MagicMock()
        perms.manage_webhooks = True
        parent_channel.permissions_for.return_value = perms

        existing_wh = MagicMock(spec=discord.Webhook)
        existing_wh.token = "token789"
        existing_wh.user = self.bot.user
        parent_channel.webhooks = AsyncMock(return_value=[existing_wh])

        thread = MagicMock(spec=discord.Thread)
        thread.id = 4004
        thread.name = "discussion-thread"
        thread.parent = parent_channel

        wh, ret_thread = await self.bot.get_webhook_for_channel(thread)
        self.assertEqual(wh, existing_wh)
        self.assertEqual(ret_thread, thread)

    async def test_send_via_webhook_success(self):
        """Verify send_via_webhook successfully sends message with custom username and avatar_url."""
        webhook = MagicMock(spec=discord.Webhook)
        webhook.send = AsyncMock()
        self.bot.get_webhook_for_channel = AsyncMock(return_value=(webhook, None))

        channel = MagicMock(spec=discord.TextChannel)
        success = await self.bot.send_via_webhook(
            channel=channel,
            content="Hello from pirate persona!",
            username="Cap'n Jack",
            avatar_url="https://example.com/pirate.png"
        )

        self.assertTrue(success)
        webhook.send.assert_called_once_with(
            content="Hello from pirate persona!",
            username="Cap'n Jack",
            avatar_url="https://example.com/pirate.png"
        )

    async def test_send_via_webhook_chunking(self):
        """Verify send_via_webhook splits content exceeding 2000 characters."""
        webhook = MagicMock(spec=discord.Webhook)
        webhook.send = AsyncMock()
        self.bot.get_webhook_for_channel = AsyncMock(return_value=(webhook, None))

        channel = MagicMock(spec=discord.TextChannel)
        long_content = "A" * 4500
        success = await self.bot.send_via_webhook(
            channel=channel,
            content=long_content,
            username="Bot",
            avatar_url="https://example.com/bot.png"
        )

        self.assertTrue(success)
        self.assertEqual(webhook.send.call_count, 3)
        calls = webhook.send.call_args_list
        self.assertEqual(len(calls[0].kwargs["content"]), 2000)
        self.assertEqual(len(calls[1].kwargs["content"]), 2000)
        self.assertEqual(len(calls[2].kwargs["content"]), 500)

    async def test_send_via_webhook_invalid_avatar_retry(self):
        """Verify send_via_webhook retries without avatar_url if Discord rejects the avatar."""
        webhook = MagicMock(spec=discord.Webhook)
        # First call raises HTTPException (e.g. invalid avatar), second call succeeds
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.reason = "Bad Request"
        http_exc = discord.HTTPException(response=mock_response, message="Invalid Form Body In avatar_url")

        webhook.send = AsyncMock(side_effect=[http_exc, None])
        self.bot.get_webhook_for_channel = AsyncMock(return_value=(webhook, None))

        channel = MagicMock(spec=discord.TextChannel)
        success = await self.bot.send_via_webhook(
            channel=channel,
            content="Response text",
            username="Bot",
            avatar_url="https://example.com/broken-link.png"
        )

        self.assertTrue(success)
        self.assertEqual(webhook.send.call_count, 2)
        # First call had avatar_url
        self.assertEqual(webhook.send.call_args_list[0].kwargs.get("avatar_url"), "https://example.com/broken-link.png")
        # Retry had no avatar_url
        self.assertNotIn("avatar_url", webhook.send.call_args_list[1].kwargs)

    async def test_on_message_sends_via_webhook_and_deletes_status(self):
        """Verify on_message sends via webhook using the active prompt's avatar and deletes status message."""
        self.bot.prompt_name = "pirate"
        self.bot.prompt_profiles["pirate"] = {
            "avatar_url": "https://example.com/pirate.png",
            "username": "Cap'n Jack"
        }
        self.bot.random_mode = False

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock()
        message.author.id = 99999
        message.author.name = "TestUser"
        message.content = "<@12345> Ahoy there!"
        message.role_mentions = []
        message.reference = None
        message.attachments = []
        message.embeds = []

        status_msg = MagicMock(spec=discord.Message)
        status_msg.edit = AsyncMock()
        status_msg.delete = AsyncMock()
        message.reply = AsyncMock(return_value=status_msg)

        self.bot.should_respond = AsyncMock(return_value=True)
        self.bot.send_via_webhook = AsyncMock(return_value=True)

        with patch("bot.get_llm_response", AsyncMock(return_value="Ahoy matey!")):
            await self.bot.on_message(message)

        # Verify send_via_webhook was called with persona details
        self.bot.send_via_webhook.assert_called_once()
        call_kwargs = self.bot.send_via_webhook.call_args[1]
        self.assertEqual(call_kwargs["username"], "Cap'n Jack")
        self.assertEqual(call_kwargs["avatar_url"], "https://example.com/pirate.png")
        self.assertIn("Ahoy matey!", call_kwargs["content"])

        # Verify status message was deleted
        status_msg.delete.assert_called_once()

    async def test_on_message_fallback_edits_status_when_webhook_fails(self):
        """Verify on_message edits status_msg if webhook sending fails or is unsupported."""
        self.bot.prompt_name = "pirate"
        self.bot.prompt_profiles["pirate"] = {
            "avatar_url": "https://example.com/pirate.png",
            "username": "Cap'n Jack"
        }
        self.bot.random_mode = False

        message = MagicMock(spec=discord.Message)
        message.author = MagicMock()
        message.author.id = 99999
        message.author.name = "TestUser"
        message.content = "<@12345> Hello bot!"
        message.role_mentions = []
        message.reference = None
        message.attachments = []
        message.embeds = []

        status_msg = MagicMock(spec=discord.Message)
        status_msg.edit = AsyncMock()
        status_msg.delete = AsyncMock()
        message.reply = AsyncMock(return_value=status_msg)

        self.bot.should_respond = AsyncMock(return_value=True)
        # Webhook fails (e.g. DM channel or missing permissions)
        self.bot.send_via_webhook = AsyncMock(return_value=False)

        with patch("bot.get_llm_response", AsyncMock(return_value="Standard assistant response")):
            await self.bot.on_message(message)

        # Verify send_via_webhook was attempted
        self.bot.send_via_webhook.assert_called_once()

        # Status message was NOT deleted, but edited as fallback
        status_msg.delete.assert_not_called()
        status_msg.edit.assert_called()
        edited_content = status_msg.edit.call_args[1]["content"]
        self.assertIn("Standard assistant response", edited_content)

    async def test_send_via_webhook_with_local_image_file(self):
        """Verify send_via_webhook reads local image files and creates persona webhook with avatar bytes."""
        webhook = MagicMock(spec=discord.Webhook)
        webhook.avatar = MagicMock()
        webhook.avatar.url = "https://cdn.discordapp.com/avatars/123/pirate-avatar.png"
        webhook.send = AsyncMock()
        self.bot.get_webhook_for_channel = AsyncMock(return_value=(webhook, None))

        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 5005

        with patch("os.path.isfile", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data=b"imagebytes123")):
            success = await self.bot.send_via_webhook(
                channel=channel,
                content="Ahoy from local file!",
                username="Local Pirate",
                avatar_url="alternate profile pics/pirate.jpg",
                prompt_name="pirate"
            )

        self.assertTrue(success)
        self.bot.get_webhook_for_channel.assert_called_once_with(
            channel,
            prompt_name="pirate",
            avatar_bytes=b"imagebytes123"
        )
        self.assertEqual(
            self.bot._local_avatar_cdn_cache.get("alternate profile pics/pirate.jpg"),
            "https://cdn.discordapp.com/avatars/123/pirate-avatar.png"
        )

    async def test_reload_prompts_slash_command(self):
        """Verify /reloadprompts slash command reloads prompts and replies with count."""
        interaction = MagicMock(spec=discord.Interaction)
        interaction.client = self.bot
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        test_prompts = {
            "p1": {"prompt": "Prompt 1", "avatar_url": "https://example.com/1.png"},
            "p2": {"prompt": "Prompt 2"}
        }
        with patch.object(LLMBot, "load_prompts", return_value=test_prompts):
            await reload_prompts_command.callback(interaction)

        self.assertEqual(len(self.bot.prompts), 2)
        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        self.assertIn("Successfully reloaded 2 system prompts", msg)
        self.assertIn("1 with custom profile picture", msg)

    async def test_should_respond_to_direct_bot_reply(self):
        """Verify should_respond returns True when replying to a message authored directly by the bot."""
        message = MagicMock(spec=discord.Message)
        message.author = MagicMock()
        message.author.id = 99999
        message.guild = MagicMock()
        message.mentions = []
        message.role_mentions = []
        message.content = "What did you mean?"
        message.webhook_id = None

        # Referenced message by the bot
        ref_msg = MagicMock(spec=discord.Message)
        ref_msg.id = 88888
        ref_msg.author = self.bot.user
        ref_msg.webhook_id = None

        message.reference = MagicMock()
        message.reference.message_id = 88888
        message.reference.resolved = ref_msg

        from bot import should_respond_to_message
        should_respond = await should_respond_to_message(self.bot, message)
        self.assertTrue(should_respond)

    async def test_should_respond_to_bot_webhook_reply(self):
        """Verify should_respond returns True when replying to a message authored by a bot webhook."""
        message = MagicMock(spec=discord.Message)
        message.author = MagicMock()
        message.author.id = 99999
        message.guild = MagicMock()
        message.mentions = []
        message.role_mentions = []
        message.content = "Tell me more about that!"
        message.webhook_id = None

        # Add webhook id to bot's known webhooks
        self.bot._known_webhook_ids.add(77777)

        ref_msg = MagicMock(spec=discord.Message)
        ref_msg.id = 66666
        ref_msg.author = MagicMock() # Webhook author object, not bot.user
        ref_msg.author.id = 77777
        ref_msg.webhook_id = 77777

        message.reference = MagicMock()
        message.reference.message_id = 66666
        message.reference.resolved = ref_msg

        from bot import should_respond_to_message
        should_respond = await should_respond_to_message(self.bot, message)
        self.assertTrue(should_respond)

    async def test_should_respond_ignores_reply_to_other_users_and_webhooks(self):
        """Verify should_respond returns False when replying to other users or unrelated webhooks."""
        from bot import should_respond_to_message

        # Reply to regular other user
        message = MagicMock(spec=discord.Message)
        message.author = MagicMock()
        message.author.id = 99999
        message.guild = MagicMock()
        message.mentions = []
        message.role_mentions = []
        message.content = "Hey friend"
        message.webhook_id = None

        ref_msg = MagicMock(spec=discord.Message)
        ref_msg.id = 55555
        ref_msg.author = MagicMock()
        ref_msg.author.id = 44444
        ref_msg.webhook_id = None

        message.reference = MagicMock()
        message.reference.message_id = 55555
        message.reference.resolved = ref_msg

        self.assertFalse(await should_respond_to_message(self.bot, message))

        # Reply to unrelated third-party webhook
        ref_webhook_msg = MagicMock(spec=discord.Message)
        ref_webhook_msg.id = 55556
        ref_webhook_msg.author = MagicMock()
        ref_webhook_msg.webhook_id = 33333

        message.reference.resolved = ref_webhook_msg
        message.channel.webhooks = AsyncMock(return_value=[])

        self.assertFalse(await should_respond_to_message(self.bot, message))

    async def test_is_bot_webhook_id_discovery_and_caching(self):
        """Verify is_bot_webhook_id dynamically discovers webhooks in channel and caches them."""
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 1212
        channel.parent = None

        wh1 = MagicMock()
        wh1.id = 10101
        wh1.name = "AIBot Webhook"
        wh1.user = MagicMock() # not self.user, but name matches bot general webhook

        wh2 = MagicMock()
        wh2.id = 20202
        wh2.name = "Unrelated Webhook"
        wh2.user = MagicMock()

        channel.webhooks = AsyncMock(return_value=[wh1, wh2])

        # Initially not in memory cache
        self.assertNotIn(10101, self.bot._known_webhook_ids)

        is_ours = await self.bot.is_bot_webhook_id(10101, channel)
        self.assertTrue(is_ours)
        self.assertIn(10101, self.bot._known_webhook_ids)

        # Unrelated webhook is False
        is_unrelated = await self.bot.is_bot_webhook_id(20202, channel)
        self.assertFalse(is_unrelated)

    async def test_on_message_ignores_bot_own_webhook_messages(self):
        """Verify on_message immediately returns when incoming message was sent by a bot webhook."""
        message = MagicMock(spec=discord.Message)
        message.author = MagicMock()
        message.author.id = 10101
        message.webhook_id = 10101
        message.channel = MagicMock()

        self.bot._known_webhook_ids.add(10101)
        self.bot.should_respond = AsyncMock()

        await self.bot.on_message(message)
        self.bot.should_respond.assert_not_called()

    async def test_on_message_reply_context_strips_system_prompt_subtext(self):
        """Verify replying to a bot response cleans footer subtexts (-# System prompt: ...) from context."""
        message = MagicMock(spec=discord.Message)
        message.author = MagicMock()
        message.author.id = 99999
        message.author.name = "TestUser"
        message.guild = MagicMock()
        message.channel = MagicMock()
        message.content = "What did you say?"
        message.mentions = []
        message.role_mentions = []
        message.attachments = []
        message.embeds = []
        message.webhook_id = None

        ref_msg = MagicMock(spec=discord.Message)
        ref_msg.id = 44444
        ref_msg.author = MagicMock()
        ref_msg.author.name = "Pirate Bot"
        ref_msg.author.id = 77777
        ref_msg.webhook_id = 77777
        ref_msg.attachments = []
        ref_msg.embeds = []
        ref_msg.content = "Shiver me timbers!\n\n-# System prompt: pirate\n-# Fallback model used: gemini-2.5-flash-lite"

        message.reference = MagicMock()
        message.reference.message_id = 44444
        message.reference.resolved = ref_msg

        self.bot._known_webhook_ids.add(77777)

        # Mock LLM and status reply
        status_msg = AsyncMock()
        status_msg.delete = AsyncMock()
        message.reply = AsyncMock(return_value=status_msg)

        captured_prompt = None
        async def fake_get_response(prompt, system_prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Aye matey!"

        with patch("bot.get_llm_response", side_effect=fake_get_response), \
             patch.object(self.bot, "send_via_webhook", return_value=True):
            await self.bot.on_message(message)

        self.assertIsNotNone(captured_prompt)
        self.assertIn("Shiver me timbers!", captured_prompt)
        self.assertNotIn("-# System prompt:", captured_prompt)
        self.assertNotIn("-# Fallback model used:", captured_prompt)



if __name__ == "__main__":
    unittest.main()
