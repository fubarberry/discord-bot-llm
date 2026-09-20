# test_image_support.py
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from dotenv import load_dotenv

load_dotenv()

# Minimal valid 1x1 PNG bytes
TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00'
    b'\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0'
    b'\x00\x00\x00\x00IEND\xaeB`\x82'
)


class TestImageSupport(unittest.IsolatedAsyncioTestCase):

    def test_lmstudio_payload_format(self):
        """Verify that LM Studio client formats OpenAI vision messages properly."""
        import llm_client_lmstudio
        import json

        images = [{"data": TINY_PNG, "mime_type": "image/png"}]
        prompt = "What is this image?"
        system_prompt = "You are an assistant."

        captured_payload = None

        class MockResponse:
            status = 200
            async def json(self):
                return {"choices": [{"message": {"content": "This is a red pixel."}}]}

        class MockSession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def post(self, url, headers=None, data=None):
                nonlocal captured_payload
                captured_payload = json.loads(data)
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__.return_value = MockResponse()
                return mock_ctx

        with patch("aiohttp.ClientSession", return_value=MockSession()):
            resp = asyncio.run(
                llm_client_lmstudio.get_llm_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    images=images
                )
            )

        self.assertIsNotNone(captured_payload)
        user_msg = captured_payload["messages"][-1]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(user_msg["content"][0]["type"], "text")
        self.assertIn("What is this image?", user_msg["content"][0]["text"])
        self.assertEqual(user_msg["content"][1]["type"], "image_url")
        self.assertTrue(user_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_bot_attachment_extraction(self):
        """Verify attachment extraction in bot.py."""
        from bot import _extract_images_from_message

        # Mock image attachment
        img_att = MagicMock()
        img_att.content_type = "image/png"
        img_att.filename = "test.png"
        img_att.size = 100
        img_att.read = AsyncMock(return_value=TINY_PNG)

        # Mock non-image attachment
        doc_att = MagicMock()
        doc_att.content_type = "application/pdf"
        doc_att.filename = "document.pdf"
        doc_att.size = 500

        mock_msg = MagicMock()
        mock_msg.attachments = [img_att, doc_att]
        mock_msg.embeds = []

        images = asyncio.run(_extract_images_from_message(mock_msg))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["mime_type"], "image/png")
        self.assertEqual(images[0]["data"], TINY_PNG)

    async def test_bot_on_message_image_feedback(self):
        """Verify on_message sends instant image feedback message and edits status."""
        from bot import LLMBot
        from unittest.mock import PropertyMock

        bot = LLMBot()
        mock_user = MagicMock()
        mock_user.id = 12345
        mock_user.name = "TestBot"

        img_att = MagicMock()
        img_att.content_type = "image/png"
        img_att.filename = "test.png"
        img_att.size = 100
        img_att.read = AsyncMock(return_value=TINY_PNG)

        status_msg = AsyncMock()
        status_msg.edit = AsyncMock()

        mock_msg = MagicMock()
        mock_msg.author = MagicMock()
        mock_msg.author.name = "TestUser"
        mock_msg.content = "<@12345> what is in this picture?"
        mock_msg.attachments = [img_att]
        mock_msg.reference = None
        mock_msg.channel = MagicMock()
        mock_msg.channel.id = 999
        mock_msg.reply = AsyncMock(return_value=status_msg)

        with patch.object(LLMBot, 'user', new_callable=PropertyMock, return_value=mock_user):
            with patch("bot.get_llm_response", new=AsyncMock(return_value="A red square.")):
                await bot.on_message(mock_msg)

        # 1. Verify initial reply was sent immediately with image processing message
        mock_msg.reply.assert_called_once()
        initial_status_call = mock_msg.reply.call_args[0][0]
        self.assertIn("Processing image", initial_status_call)

        # 2. Verify status was edited to analyzing
        edit_calls = [call[1].get("content") or call[0][0] for call in status_msg.edit.call_args_list]
        self.assertTrue(any("Analyzing image" in str(c) for c in edit_calls))

        # 3. Verify final response was edited into status_msg
        self.assertEqual(edit_calls[-1], "A red square.")

    async def test_gemini_with_image_live(self):
        """Test Gemini API with live image input if GEMINI_API_KEY is present."""
        import llm_client_gemini

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY not configured, skipping live test.")

        images = [{"data": TINY_PNG, "mime_type": "image/png"}]
        response = await llm_client_gemini.get_llm_response(
            prompt="What color is this 1x1 image? Answer in one or two words.",
            system_prompt="You are a helpful assistant.",
            model_name="gemini-3.6-flash",
            images=images
        )
        print("\n[Live Gemini Image Response]:", response)
        self.assertIsNotNone(response)
        if "503" in response or "overloaded" in response.lower():
            self.skipTest("Gemini API overloaded during test.")
        self.assertFalse(response.startswith("⚠️"), f"Error returned: {response}")
        self.assertTrue(any(c in response.lower() for c in ["red", "color", "pixel", "image", "solid"]))

    async def test_llm_client_forwarding(self):
        """Test end-to-end get_llm_response in llm_client.py."""
        import llm_client
        if not os.getenv("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY not configured, skipping live test.")

        images = [{"data": TINY_PNG, "mime_type": "image/png"}]
        response = await llm_client.get_llm_response(
            prompt="Identify the single primary color in this image in one word.",
            system_prompt="You are a helpful assistant.",
            provider="GEMINI",
            model="gemini-3.6-flash",
            images=images
        )
        print("[Live llm_client.get_llm_response]:", response)
        self.assertIsNotNone(response)
        if "503" in response or "overloaded" in response.lower():
            self.skipTest("Gemini API overloaded during test.")
        self.assertIn("red", response.lower())


    async def test_search_agent_image_no_search(self):
        """Verify search agent passes images to decision step and skips search when image provides answer."""
        from search_agent import run_search_augmented_generation

        images = [{"data": TINY_PNG, "mime_type": "image/png"}]
        received_images_in_decision = None
        status_updates = []

        async def mock_query_llm(p, s, h, imgs=None):
            nonlocal received_images_in_decision
            if "analyzing an image provided by the user" in s.lower():
                received_images_in_decision = imgs
                return "NO_SEARCH"
            return "This image is a red pixel."

        async def status_cb(text):
            status_updates.append(text)

        response = await run_search_augmented_generation(
            prompt="what is this?",
            system_prompt="You are an assistant.",
            history=[],
            query_llm_fn=mock_query_llm,
            status_callback=status_cb,
            images=images
        )

        # 1. Vision model received images during decision step
        self.assertIsNotNone(received_images_in_decision)
        self.assertEqual(len(received_images_in_decision), 1)

        # 2. Status message showed image inspection
        self.assertTrue(any("Inspecting image content" in s for s in status_updates))

        # 3. Direct response without web search
        self.assertEqual(response, "This image is a red pixel.")

    async def test_search_agent_image_with_claim_search(self):
        """Verify search agent generates targeted search query from image claims."""
        from search_agent import run_search_augmented_generation

        images = [{"data": TINY_PNG, "mime_type": "image/png"}]
        searched_query = None

        async def mock_query_llm(p, s, h, imgs=None):
            if "analyzing an image provided by the user" in s.lower():
                return "SEARCH: NASA rover discovers liquid water on Mars"
            elif "real-time web search research" in s:
                return "Yes, NASA announced liquid water was confirmed."
            elif "next action" in p.lower():
                return "READY"
            return "ok"

        with patch("search_agent.search_duckduckgo", new=AsyncMock(return_value=[{"title": "NASA news", "href": "https://nasa.gov", "body": "Rover found water."}])) as mock_ddg:
            response = await run_search_augmented_generation(
                prompt="is this true?",
                system_prompt="You are an assistant.",
                history=[],
                query_llm_fn=mock_query_llm,
                images=images
            )
            mock_ddg.assert_called_once_with("NASA rover discovers liquid water on Mars", max_results=5)

        self.assertIn("liquid water", response)

    async def test_fetch_image_from_url_with_discord_ua(self):
        """Verify _fetch_image_from_url uses Discordbot UA and parses image content."""
        from bot import _fetch_image_from_url, DISCORD_BOT_USER_AGENT

        class MockResponse:
            status = 200
            headers = {"Content-Type": "image/jpeg"}
            async def read(self):
                return TINY_PNG

        captured_headers = None
        class MockSession:
            def __init__(self, headers=None):
                nonlocal captured_headers
                captured_headers = headers
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def get(self, url, timeout=None, allow_redirects=True):
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__.return_value = MockResponse()
                return mock_ctx

        with patch("aiohttp.ClientSession", side_effect=MockSession):
            result = await _fetch_image_from_url("https://www.kkinstagram.com/reel/12345/?stkn=abc")
            self.assertIsNotNone(result)
            self.assertEqual(result["mime_type"], "image/jpeg")
            self.assertEqual(result["data"], TINY_PNG)
            self.assertEqual(captured_headers.get("User-Agent"), DISCORD_BOT_USER_AGENT)

    async def test_extract_images_from_embed_proxy(self):
        """Verify _extract_images_from_message prioritizes proxy_url for embeds."""
        from bot import _extract_images_from_message
        import discord

        embed = discord.Embed.from_dict({
            "title": "Embed with image",
            "image": {
                "url": "https://origin.example.com/image.jpg",
                "proxy_url": "https://images-ext-1.discordapp.net/external/image.jpg"
            }
        })
        mock_msg = MagicMock()
        mock_msg.attachments = []
        mock_msg.embeds = [embed]
        mock_msg.content = ""

        with patch("bot._fetch_image_from_url", new=AsyncMock(return_value={"data": TINY_PNG, "mime_type": "image/jpeg", "filename": "embed_image"})) as mock_fetch:
            imgs = await _extract_images_from_message(mock_msg)
            self.assertEqual(len(imgs), 1)
            mock_fetch.assert_called_once_with("https://images-ext-1.discordapp.net/external/image.jpg")

    async def test_extract_images_from_content_urls(self):
        """Verify _extract_images_from_message detects image URLs directly in message content."""
        from bot import _extract_images_from_message

        mock_msg = MagicMock()
        mock_msg.attachments = []
        mock_msg.embeds = []
        mock_msg.content = "Look at this https://www.kkinstagram.com/reel/12345/?stkn=abc and also https://en.wikipedia.org/wiki/Test"

        async def fake_fetch(url):
            if "kkinstagram" in url:
                return {"data": TINY_PNG, "mime_type": "image/jpeg", "filename": "embed_image", "source_url": url}
            return None

        with patch("bot._fetch_image_from_url", side_effect=fake_fetch):
            imgs = await _extract_images_from_message(mock_msg)
            self.assertEqual(len(imgs), 1)
            self.assertEqual(imgs[0]["mime_type"], "image/jpeg")

    async def test_on_message_media_url_prompt_cleanup(self):
        """Verify on_message clears prompt when the user sends only a media link."""
        from bot import LLMBot
        from unittest.mock import PropertyMock

        bot = LLMBot()
        mock_user = MagicMock()
        mock_user.id = 12345
        mock_user.name = "TestBot"

        status_msg = AsyncMock()
        status_msg.edit = AsyncMock()

        test_url = "https://www.kkinstagram.com/reel/12345/?stkn=abc"
        mock_msg = MagicMock()
        mock_msg.id = 777
        mock_msg.author = MagicMock()
        mock_msg.author.name = "TestUser"
        mock_msg.content = f"<@12345> {test_url}"
        mock_msg.attachments = []
        mock_msg.embeds = []
        mock_msg.reference = None
        mock_msg.channel = MagicMock()
        mock_msg.channel.id = 999
        mock_msg.channel.fetch_message = AsyncMock(return_value=mock_msg)
        mock_msg.reply = AsyncMock(return_value=status_msg)

        img_data = {"data": TINY_PNG, "mime_type": "image/jpeg", "filename": "embed_image", "source_url": test_url}

        with patch.object(LLMBot, 'user', new_callable=PropertyMock, return_value=mock_user):
            with patch("bot._extract_images_from_message", new=AsyncMock(return_value=[img_data])):
                with patch("bot.get_llm_response", new=AsyncMock(return_value="Detailed visual description.")) as mock_llm:
                    await bot.on_message(mock_msg)

        mock_llm.assert_called_once()
        call_prompt = mock_llm.call_args[0][0]
        # Should have converted bare media URL into the default vision prompt
        self.assertIn("Describe this image in detail", call_prompt)
        self.assertNotIn(test_url, call_prompt)


if __name__ == "__main__":
    unittest.main()

