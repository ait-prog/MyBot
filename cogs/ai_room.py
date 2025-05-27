import discord
from discord.ext import commands
from typing import Optional, Dict, List
import asyncio
import json
import os

from database import Database
from utils import create_embed, format_currency
from model import ChatModel


class AIRoom(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()
        self.model = ChatModel()
        self.active_chats: Dict[int, List[dict]] = {}
        self.chat_history_file = "chat_history.json"
        self.load_chat_history()

    def load_chat_history(self):
        """Load chat history from file"""
        if os.path.exists(self.chat_history_file):
            try:
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    self.active_chats = json.load(f)
            except json.JSONDecodeError:
                self.active_chats = {}
        else:
            self.active_chats = {}

    def save_chat_history(self):
        """Save chat history to file"""
        with open(self.chat_history_file, 'w', encoding='utf-8') as f:
            json.dump(self.active_chats, f, ensure_ascii=False, indent=2)

    @commands.command(name="createairoom", aliases=["airoom"])
    async def create_ai_room(self, ctx):
        """Create a private room with AI companion"""
        balance = await self.db.get_balance(ctx.author.id)
        if balance["coins"] < 5000:
            await ctx.send("❌ You don't have enough coins! You need 5000 coins.")
            return

        try:
            # Create role for the room
            role = await ctx.guild.create_role(
                name=f"AI-Room-{ctx.author.name}",
                color=discord.Color.purple(),
                reason=f"Created for AI room by {ctx.author}"
            )

            # Create text channel
            text_channel = await ctx.guild.create_text_channel(
                name=f"ai-chat-{ctx.author.name}",
                overwrites={
                    ctx.guild.default_role: discord.PermissionOverwrite(read_messages=False),
                    role: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                    ctx.author: discord.PermissionOverwrite(manage_channels=True)
                }
            )

            # Add role to user
            await ctx.author.add_roles(role)

            # Save information in database
            await self.db.add_private_room(
                text_channel.id,
                text_channel.id,
                role.id,
                ctx.guild.id,
                ctx.author.id,
                "AI Chat Room"
            )

            # Deduct coins
            await self.db.add_currency(ctx.author.id, "coins", -5000)

            # Initialize chat history
            self.active_chats[text_channel.id] = []
            self.save_chat_history()

            embed = create_embed(
                title="AI Room Created",
                description=f"You created an AI chat room {text_channel.mention}!",
                fields=[
                    {
                        "name": "Role",
                        "value": role.mention,
                        "inline": True
                    },
                    {
                        "name": "Cost",
                        "value": format_currency(5000),
                        "inline": True
                    }
                ],
                color=discord.Color.purple()
            )

            await ctx.send(embed=embed)

            # Send welcome message
            welcome_embed = create_embed(
                title="Welcome to AI Chat Room",
                description=(
                    "I'm your AI companion! Feel free to chat with me.\n\n"
                    "Available commands:\n"
                    "• `!clear` - Clear chat history\n"
                    "• `!personality` - Change AI personality\n"
                    "• `!memory` - Show chat memory status\n"
                    "• `!help` - Show this help message"
                ),
                color=discord.Color.purple()
            )
            await text_channel.send(embed=welcome_embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permission to create channels and roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while creating the room!")

    @commands.command(name="personality")
    async def change_personality(self, ctx, *, personality: str):
        """Change AI personality"""
        if ctx.channel.id not in self.active_chats:
            await ctx.send("❌ This command can only be used in AI chat rooms!")
            return

        # Save personality change in chat history
        self.active_chats[ctx.channel.id].append({
            "type": "system",
            "content": f"Personality changed to: {personality}"
        })
        self.save_chat_history()

        embed = create_embed(
            title="Personality Changed",
            description=f"AI personality has been changed to: {personality}",
            color=discord.Color.purple()
        )
        await ctx.send(embed=embed)

    @commands.command(name="memory")
    async def show_memory(self, ctx):
        """Show chat memory status"""
        if ctx.channel.id not in self.active_chats:
            await ctx.send("❌ This command can only be used in AI chat rooms!")
            return

        messages = self.active_chats[ctx.channel.id]
        memory_size = len(messages)

        embed = create_embed(
            title="Chat Memory Status",
            description=f"Current memory size: {memory_size} messages",
            fields=[
                {
                    "name": "Memory Limit",
                    "value": "50 messages",
                    "inline": True
                },
                {
                    "name": "Memory Usage",
                    "value": f"{memory_size}/50",
                    "inline": True
                }
            ],
            color=discord.Color.purple()
        )
        await ctx.send(embed=embed)

    @commands.command(name="clear")
    async def clear_chat(self, ctx):
        """Clear chat history with AI"""
        if ctx.channel.id not in self.active_chats:
            await ctx.send("❌ This command can only be used in AI chat rooms!")
            return

        self.active_chats[ctx.channel.id] = []
        self.save_chat_history()

        embed = create_embed(
            title="Chat History Cleared",
            description="All chat history has been cleared!",
            color=discord.Color.purple()
        )
        await ctx.send(embed=embed)

    @commands.Cog.listener()
    async def on_message(self, message):
        """Handle messages in AI rooms"""
        if message.author.bot:
            return

        # Check if message is in an AI room
        if message.channel.id in self.active_chats:
            # Add user message to history
            self.active_chats[message.channel.id].append({
                "user": message.author.name,
                "content": message.content
            })

            # Keep only last 50 messages
            if len(self.active_chats[message.channel.id]) > 50:
                self.active_chats[message.channel.id] = self.active_chats[message.channel.id][-50:]

            # Save chat history
            self.save_chat_history()

            # Get AI response
            response = await self.model.generate_response(
                message.author.id,
                message.content,
                history=self.active_chats[message.channel.id]
            )

            # Add AI response to history
            self.active_chats[message.channel.id].append({
                "user": "AI",
                "content": response
            })

            # Send response
            await message.channel.send(response)

    @commands.Cog.listener()
    async def on_guild_channel_delete(self, channel):
        """Clean up when AI room is deleted"""
        if channel.id in self.active_chats:
            del self.active_chats[channel.id]
            self.save_chat_history()


def setup(bot):
    bot.add_cog(AIRoom(bot))
