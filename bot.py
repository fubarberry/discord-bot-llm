# bot.py
import discord
from discord import app_commands
from discord.ext import commands
from llm_client import get_llm_response
import json
from typing import List, Dict, Optional, Any
import random
import re
import os
import aiohttp

# The maximum number of messages to keep in the history for each channel.
# This is set via the MAX_HISTORY environment variable.
MAX_HISTORY = int(os.getenv("MAX_HISTORY", 20))

SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.gif')
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB limit
MAX_IMAGES = 5


async def _fetch_image_from_attachment(attachment: discord.Attachment) -> Optional[Dict[str, Any]]:
    """Downloads and returns image data if attachment is a supported image."""
    content_type = attachment.content_type or ""
    is_image = content_type.startswith("image/") or attachment.filename.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
    if not is_image:
        return None
    if attachment.size > MAX_IMAGE_SIZE:
        print(f"Skipping attachment {attachment.filename}: size {attachment.size} exceeds 20MB limit.")
        return None
    try:
        data = await attachment.read()
        mime = content_type if content_type.startswith("image/") else "image/png"
        return {"data": data, "mime_type": mime, "filename": attachment.filename}
    except Exception as e:
        print(f"Error reading attachment {attachment.filename}: {e}")
        return None


async def _fetch_image_from_url(url: str) -> Optional[Dict[str, Any]]:
    """Downloads image bytes from a URL (e.g. from Discord embeds)."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
                    if ct.startswith("image/") or any(url.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
                        data = await resp.read()
                        if len(data) <= MAX_IMAGE_SIZE:
                            mime = ct if ct.startswith("image/") else "image/png"
                            return {"data": data, "mime_type": mime, "filename": "embed_image"}
    except Exception as e:
        print(f"Error fetching image from URL {url}: {e}")
    return None


async def _extract_images_from_message(msg: discord.Message) -> List[Dict[str, Any]]:
    """Extracts images from message attachments and embeds."""
    images = []
    # 1. Attachments
    if msg.attachments:
        for att in msg.attachments:
            img = await _fetch_image_from_attachment(att)
            if img:
                images.append(img)
                if len(images) >= MAX_IMAGES:
                    break

    # 2. Embeds (if no attachments or space remaining)
    if len(images) < MAX_IMAGES and msg.embeds:
        for embed in msg.embeds:
            url_to_fetch = None
            if embed.image and embed.image.url:
                url_to_fetch = embed.image.url
            elif embed.thumbnail and embed.thumbnail.url:
                url_to_fetch = embed.thumbnail.url
            
            if url_to_fetch:
                img = await _fetch_image_from_url(url_to_fetch)
                if img:
                    images.append(img)
                    if len(images) >= MAX_IMAGES:
                        break

    return images

class LLMBot(commands.Bot):
    """
    A Discord bot that uses slash commands to interact with a configured LLM.
    """
    def __init__(self):
        """
        Initializes the bot. The LLM provider is configured via environment variables.
        """
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        intents.guilds = True
        # The prefix is required but won't be used for slash commands.
        super().__init__(command_prefix="!", intents=intents)

        self.prompts = self.load_prompts()
        self.banned_words = self.load_banned_words() # Load banned words
        
        # Load persistent settings
        self.settings = self.load_settings()
        self.system_prompt = self.settings.get("system_prompt", self.prompts.get("default", "You are a helpful assistant."))
        self.random_mode = self.settings.get("random_mode", False)
        self.thinking_enabled = self.settings.get("thinking_enabled", False)
        self.grounding_enabled = self.settings.get("grounding_enabled", False)
        self.llm_provider = self.settings.get("llm_provider", os.getenv("LLM_PROVIDER", "LMSTUDIO")).upper()
        self.gemini_model = self.settings.get("gemini_model", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        self.run_in_background = self.settings.get("run_in_background", os.getenv("RUN_IN_BACKGROUND", "false").lower() == "true")
        self.max_history = int(self.settings.get("max_history", os.getenv("MAX_HISTORY", 15)))
        self.last_random_prompt = None # This doesn't need to be persisted
        
        self.message_history = {}
        print(f"Bot initialized (Provider: {self.llm_provider}, Model: {self.gemini_model}). Connecting to Discord...")

    def load_prompts(self) -> dict:
        """Loads system prompts from the prompts.json file."""
        try:
            with open("prompts.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("Error: prompts.json not found. Using default prompt only.")
            return {"default": "You are a helpful assistant."}
        except json.JSONDecodeError:
            print("Error: Could not decode prompts.json. Please check its format.")
            return {"default": "You are a helpful assistant."}

    def load_banned_words(self) -> List[str]:
        """Loads banned words from the banned_words.json file."""
        try:
            with open("banned_words.json", "r") as f:
                words = json.load(f)
                # Ensure all words are lowercase for case-insensitive matching
                return [word.lower() for word in words]
        except FileNotFoundError:
            print("Warning: banned_words.json not found. No words will be banned.")
            return []
        except json.JSONDecodeError:
            print("Error: Could not decode banned_words.json. No words will be banned.")
            return []

    def load_settings(self) -> dict:
        """Loads settings from settings.json (falls back to settings.json.example)."""
        try:
            with open("settings.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            try:
                with open("settings.json.example", "r") as f:
                    print("Notice: settings.json not found. Loaded defaults from settings.json.example.")
                    return json.load(f)
            except Exception:
                print("Warning: settings.json not found. Using defaults.")
                return {}
        except json.JSONDecodeError:
            print("Error: Could not decode settings.json. Using defaults.")
            return {}

    def save_settings(self):
        """Saves current settings to settings.json."""
        settings = {
            "system_prompt": self.system_prompt,
            "random_mode": self.random_mode,
            "thinking_enabled": self.thinking_enabled,
            "grounding_enabled": self.grounding_enabled,
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model,
            "run_in_background": self.run_in_background,
            "max_history": self.max_history
        }
        try:
            with open("settings.json", "w") as f:
                json.dump(settings, f, indent=4)
            print("Settings saved to settings.json")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def contains_banned_word(self, text: str) -> bool:
        """Checks if the given text contains any banned words (case-insensitive)."""
        if not self.banned_words:
            return False
        # Use regex to check for whole words to avoid partial matches (e.g., 'assist' in 'assistant')
        for word in self.banned_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text.lower()):
                return True
        return False

    async def setup_hook(self):
        """Syncs slash commands when the bot logs in."""
        await self.tree.sync()
        print("Slash commands have been synced.")

    async def on_ready(self):
        """Called when the bot has successfully connected to Discord."""
        print(f'Logged in as {self.user.name} (ID: {self.user.id})')
        print('Bot is ready to receive commands and messages.')
        print('------')

    async def on_message(self, message: discord.Message):
        """Handles direct mentions to the bot."""
        if message.author == self.user:
            return

        if self.user.mentioned_in(message):
            status_msg = None
            try:
                # Send early status reply so the user gets instant visual feedback
                has_possible_images = bool(message.attachments) or bool(message.reference)
                if has_possible_images:
                    status_msg = await message.reply("🖼️ **Processing image(s)...** Please wait, image analysis can take a moment.", mention_author=False)
                else:
                    status_msg = await message.reply("⏳ Generating response...", mention_author=False)

                prompt = message.content.replace(f'<@!{self.user.id}>', '').replace(f'<@{self.user.id}>', '').strip()
                images: List[Dict[str, Any]] = []

                # Check if the message is a reply
                if message.reference and message.reference.message_id:
                    try:
                        referenced_message = await message.channel.fetch_message(message.reference.message_id)
                        
                        # Extract images from the referenced message
                        ref_images = await _extract_images_from_message(referenced_message)
                        if ref_images:
                            images.extend(ref_images)
                            print(f"Extracted {len(ref_images)} image(s) from referenced message by {referenced_message.author.name}")

                        if referenced_message.content:
                            context_str = f"[Context: Replying to a message by {referenced_message.author.name}: \"{referenced_message.content}\"]\n\n"
                            prompt = context_str + prompt
                            print(f"Added reply context from {referenced_message.author.name}")
                        elif ref_images:
                            context_str = f"[Context: Replying to an image posted by {referenced_message.author.name}]\n\n"
                            prompt = context_str + prompt
                    except discord.NotFound:
                        print("Referenced message not found (it might have been deleted).")
                    except Exception as e:
                        print(f"Error fetching referenced message: {e}")

                # Extract images from current message
                current_images = await _extract_images_from_message(message)
                if current_images:
                    for img in current_images:
                        if len(images) < MAX_IMAGES:
                            images.append(img)
                    print(f"Extracted {len(current_images)} image(s) from current message by {message.author.name}")

                if not prompt and not images:
                    await status_msg.edit(content="You mentioned me, but didn't ask anything or provide an image! How can I help?")
                    return
                elif not prompt and images:
                    prompt = "Describe this image in detail and tell me what you observe."

                # Check user's prompt for banned words
                if self.contains_banned_word(prompt):
                    await status_msg.edit(content="I'm sorry, but your message contains inappropriate language and cannot be processed.")
                    print(f"Rejected prompt from {message.author.name} due to banned word.")
                    return

                # Update status with image count details if images were detected
                if images:
                    img_text = "image" if len(images) == 1 else f"{len(images)} images"
                    await status_msg.edit(content=f"🖼️ **Analyzing {img_text}...** Please hold on, vision models may take a bit longer to process.")
                elif has_possible_images:
                    await status_msg.edit(content="⏳ Generating response...")

                async def update_status(text: str):
                    if status_msg:
                        try:
                            await status_msg.edit(content=text)
                        except Exception as e:
                            print(f"Error editing status message: {e}")

                prompt_log = prompt if len(prompt) <= 100 else f"{prompt[:100]}..."
                image_log = f" [with {len(images)} image(s)]" if images else ""
                print(f"Received prompt from {message.author.name}: '{prompt_log}'{image_log}")

                channel_id = message.channel.id
                if channel_id not in self.message_history:
                    self.message_history[channel_id] = []

                history = self.message_history[channel_id]
                
                current_system_prompt = self.system_prompt
                if self.random_mode:
                    available_prompts = list(self.prompts.values())
                    if available_prompts:
                        chosen_prompt = random.choice(available_prompts)
                        self.last_random_prompt = chosen_prompt
                        current_system_prompt = chosen_prompt
                        print(f"Random mode ON. Using prompt: {chosen_prompt[:60]}...")
                    else:
                        print("Random mode ON, but no prompts are available. Using default.")

                llm_response = await get_llm_response(
                    prompt,
                    current_system_prompt,
                    thinking_enabled=self.thinking_enabled,
                    history=history,
                    grounding=self.grounding_enabled,
                    status_callback=update_status,
                    provider=self.llm_provider,
                    model=self.gemini_model if self.llm_provider == "GEMINI" else None,
                    images=images if images else None
                )

                if llm_response:
                    # If this is an error reported from the LLM provider, show it directly without saving to history
                    if llm_response.startswith("⚠️"):
                        print(f"Reporting LLM error to user: {llm_response}")
                        await status_msg.edit(content=llm_response)
                        return

                    # Check AI's response for banned words
                    if self.contains_banned_word(llm_response):
                        await status_msg.edit(content="I'm sorry, the generated response contained inappropriate content and has been blocked.")
                        print("Blocked an AI response due to a banned word.")
                        return
                    
                    # Add the user's prompt and the AI's response to the history
                    history_prompt = prompt
                    if images:
                        history_prompt = f"{prompt}\n[Attached {len(images)} image(s)]"

                    history.append({"role": "user", "content": history_prompt})
                    history.append({"role": "assistant", "content": llm_response})
                    
                    # Keep the history to a manageable size
                    if len(history) > self.max_history * 2: # Each interaction is 2 messages
                        self.message_history[channel_id] = history[-(self.max_history * 2):]

                    if len(llm_response) > 2000:
                        parts = [llm_response[i:i+2000] for i in range(0, len(llm_response), 2000)]
                        await status_msg.edit(content=parts[0])
                        for part in parts[1:]:
                            await message.channel.send(part)
                    else:
                        await status_msg.edit(content=llm_response)
                else:
                    await status_msg.edit(content="⚠️ No response was received from the model.")
            
            except Exception as e:
                print(f"An error occurred while processing a message: {e}")
                if status_msg:
                    try:
                        await status_msg.edit(content=f"⚠️ An unexpected error occurred: {e}")
                    except Exception:
                        pass
                else:
                    await message.channel.send(f"⚠️ An unexpected error occurred: {e}")

# --- Slash Commands ---
# (The rest of the slash commands remain the same)

@app_commands.command(name="setprompt", description="Sets the system prompt for the bot. This disables random mode.")
@app_commands.describe(name="Choose a preset or type your own custom prompt.")
async def setprompt(interaction: discord.Interaction, name: str):
    bot = interaction.client
    
    response_parts = []
    if bot.random_mode:
        bot.random_mode = False
        response_parts.append("🎲 Random prompt mode has been **disabled**.")

    if name in bot.prompts:
        bot.system_prompt = bot.prompts[name]
        response_parts.append(f"System prompt changed to **{name}**.")
    else:
        bot.system_prompt = name
        response_parts.append(f"Custom system prompt has been set.")
    
    bot.save_settings()
    
    await interaction.response.send_message("\n".join(response_parts), ephemeral=True)


@setprompt.autocomplete('name')
async def setprompt_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    bot = interaction.client
    choices = [
        app_commands.Choice(name=prompt_name, value=prompt_name)
        for prompt_name in bot.prompts if current.lower() in prompt_name.lower()
    ]
    return choices[:25]

@app_commands.command(name="prompt", description="Shows the current system prompt or random mode status.")
async def prompt(interaction: discord.Interaction):
    bot = interaction.client
    if bot.random_mode:
        response_message = "🎲 **Random prompt mode is ON.**"
        if bot.last_random_prompt:
            response_message += f"\n\n**Last Used Prompt:**\n```\n{bot.last_random_prompt}\n```"
        else:
            response_message += "\n\nA random prompt will be picked for the next message you send me."
        await interaction.response.send_message(response_message, ephemeral=True)
    else:
        await interaction.response.send_message(f"**Current System Prompt:**\n```\n{bot.system_prompt}\n```", ephemeral=True)

@app_commands.command(name="random", description="Toggle random system prompts for each reply.")
@app_commands.describe(enabled="Set to 'True' to enable, 'False' to disable.")
async def random_command(interaction: discord.Interaction, enabled: bool):
    bot = interaction.client
    bot.random_mode = enabled
    bot.save_settings()
    if enabled:
        bot.last_random_prompt = None
        await interaction.response.send_message("✅ Random prompt mode has been **enabled**.", ephemeral=True)
    else:
        await interaction.response.send_message("❌ Random prompt mode has been **disabled**.", ephemeral=True)

@app_commands.command(name="think", description="Toggle whether the bot shows its thought process (LM Studio only).")
@app_commands.describe(enabled="Set to 'True' to enable, 'False' to disable.")
async def think_command(interaction: discord.Interaction, enabled: bool):
    bot = interaction.client
    bot.thinking_enabled = enabled
    bot.save_settings()
    if enabled:
        await interaction.response.send_message("🤔 Thinking mode has been **enabled**. The bot will now show its thought process.", ephemeral=True)
    else:
        await interaction.response.send_message("✅ Thinking mode has been **disabled**.", ephemeral=True)

@app_commands.command(name="grounding", description="Toggle web search grounding (DuckDuckGo + Jina Reader).")
@app_commands.describe(enabled="Set to 'True' to enable, 'False' to disable.")
async def grounding_command(interaction: discord.Interaction, enabled: bool):
    bot = interaction.client
    bot.grounding_enabled = enabled
    bot.save_settings()
    if enabled:
        await interaction.response.send_message("🌍 Web search grounding has been **enabled**. The bot will search the web when needed.", ephemeral=True)
    else:
        await interaction.response.send_message("✅ Web search grounding has been **disabled**.", ephemeral=True)

@app_commands.command(name="websearch", description="Toggle web search grounding (DuckDuckGo + Jina Reader).")
@app_commands.describe(enabled="Set to 'True' to enable, 'False' to disable.")
async def websearch_command(interaction: discord.Interaction, enabled: bool):
    await grounding_command(interaction, enabled)

POPULAR_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

@app_commands.command(name="provider", description="Switch the LLM provider between LM Studio and Gemini (Admin only).")
@app_commands.describe(name="Select the LLM provider.")
@app_commands.choices(name=[
    app_commands.Choice(name="LM Studio (Local)", value="LMSTUDIO"),
    app_commands.Choice(name="Google Gemini (Cloud)", value="GEMINI")
])
@app_commands.default_permissions(administrator=True)
async def provider_command(interaction: discord.Interaction, name: app_commands.Choice[str]):
    if interaction.guild and not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ You do not have permission to use this command (Administrator required).", ephemeral=True)
        return

    bot = interaction.client
    bot.llm_provider = name.value
    bot.save_settings()
    
    if name.value == "GEMINI":
        await interaction.response.send_message(f"✅ LLM provider switched to **Google Gemini** (Model: `{bot.gemini_model}`).", ephemeral=True)
    else:
        await interaction.response.send_message("✅ LLM provider switched to **LM Studio** (Local Server).", ephemeral=True)

@app_commands.command(name="source", description="Alias for /provider (Admin only).")
@app_commands.describe(name="Select the LLM provider.")
@app_commands.choices(name=[
    app_commands.Choice(name="LM Studio (Local)", value="LMSTUDIO"),
    app_commands.Choice(name="Google Gemini (Cloud)", value="GEMINI")
])
@app_commands.default_permissions(administrator=True)
async def source_command(interaction: discord.Interaction, name: app_commands.Choice[str]):
    await provider_command(interaction, name)

@app_commands.command(name="model", description="Change the active LLM model (Admin only).")
@app_commands.describe(name="Choose a preset model or enter a custom model name.")
@app_commands.default_permissions(administrator=True)
async def model_command(interaction: discord.Interaction, name: str):
    if interaction.guild and not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ You do not have permission to use this command (Administrator required).", ephemeral=True)
        return

    bot = interaction.client
    model_name = name.strip()
    if not model_name:
        await interaction.response.send_message(f"Current Gemini model: `{bot.gemini_model}` (Active Provider: `{bot.llm_provider}`).", ephemeral=True)
        return

    bot.gemini_model = model_name
    bot.save_settings()
    await interaction.response.send_message(f"✅ Model updated to **{model_name}** (Active provider: `{bot.llm_provider}`).", ephemeral=True)

@model_command.autocomplete('name')
async def model_command_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    choices = [
        app_commands.Choice(name=m, value=m)
        for m in POPULAR_GEMINI_MODELS
        if current.lower() in m.lower()
    ]
    if current and not any(c.value == current for c in choices):
        choices.insert(0, app_commands.Choice(name=f"Custom: {current}", value=current))
    return choices[:25]

@app_commands.command(name="showprompts", description="Lists all available preset prompts.")
async def list_prompts(interaction: discord.Interaction):
    bot = interaction.client
    if not bot.prompts:
        await interaction.response.send_message("No predefined prompts found.", ephemeral=True)
        return
    embed = discord.Embed(title="Available System Prompts", color=discord.Color.blue())
    for name, content in bot.prompts.items():
        embed.add_field(name=name, value=f"```{content[:100]}...```", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@app_commands.command(name="help", description="Shows the list of available commands.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="Bot Commands", description="Here are the available slash commands:", color=discord.Color.green())
    embed.add_field(name="/provider [LM Studio|Gemini]", value="Switches the active LLM provider (Admin only). Alias: `/source`.", inline=False)
    embed.add_field(name="/model [name]", value="Changes the active model name (Admin only).", inline=False)
    embed.add_field(name="/setprompt [name|custom]", value="Sets the bot's system prompt. This disables random mode.", inline=False)
    embed.add_field(name="/prompt", value="Displays the current system prompt or random mode status.", inline=False)
    embed.add_field(name="/random [True|False]", value="Toggles using a random prompt for each reply.", inline=False)
    embed.add_field(name="/think [True|False]", value="Toggles whether the bot shows its thought process (LM Studio only).", inline=False)
    embed.add_field(name="/grounding [True|False]", value="Toggles whether the bot uses web search grounding (DuckDuckGo + Jina Reader).", inline=False)
    embed.add_field(name="/showprompts", value="Lists all available preset prompts.", inline=False)
    embed.add_field(name="/clearhistory", value="Clears the conversation history for this channel.", inline=False)
    embed.add_field(name="/help", value="Shows this help message.", inline=False)
    embed.add_field(name="Mention the bot (@BotName)", value="Ask the bot a question directly by mentioning it.", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@app_commands.command(name="clearhistory", description="Clears the conversation history for this channel.")
async def clear_history(interaction: discord.Interaction):
    bot = interaction.client
    channel_id = interaction.channel_id
    if channel_id in bot.message_history:
        bot.message_history[channel_id] = []
        await interaction.response.send_message("Conversation history for this channel has been cleared.", ephemeral=True)
    else:
        await interaction.response.send_message("There is no history to clear for this channel.", ephemeral=True)

async def setup(bot: commands.Bot):
    """Adds the slash commands to the bot's command tree."""
    bot.tree.add_command(provider_command)
    bot.tree.add_command(source_command)
    bot.tree.add_command(model_command)
    bot.tree.add_command(setprompt)
    bot.tree.add_command(prompt)
    bot.tree.add_command(list_prompts)
    bot.tree.add_command(help_command)
    bot.tree.add_command(random_command)
    bot.tree.add_command(think_command)
    bot.tree.add_command(grounding_command)
    bot.tree.add_command(websearch_command)
    bot.tree.add_command(clear_history)


