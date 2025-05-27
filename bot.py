import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from model import ChatModel
from datetime import datetime, timedelta
from typing import Optional


load_dotenv()
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
chat_model = ChatModel()


@bot.event
async def on_ready():
    print(f'{bot.user} work!')
    print(f'Device: {chat_model.device}')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if isinstance(message.channel, discord.DMChannel):
        try:
            response = chat_model.generate_response(message.author.id, message.content)
            await message.channel.send(response)

        except Exception as e:
            await message.channel.send(f"Произошла ошибка: {str(e)}")

    await bot.process_commands(message)


@bot.command(name='clear')
async def clear_history(ctx):
    """Delete history with user"""
    if isinstance(ctx.channel, discord.DMChannel):
        chat_model.clear_history(ctx.author.id)
        await ctx.send("Complete!")


@commands.command(name="serverinfo")
async def server_info(self, ctx):
    """Show server information"""
    guild = ctx.guild
    embed = create_embed(
        title=f"Server Information: {guild.name}",
        fields=[
            {
                "name": "Owner",
                "value": str(guild.owner),
                "inline": True
            },
            {
                "name": "Created by",
                "value": guild.created_at.strftime("%d.%m.%Y"),
                "inline": True
            },
            {
                "name": "Members",
                "value": f"Total: {guild.member_count}\nHumans: {len([m for m in guild.members if not m.bot])}\nBots: {len([m for m in guild.members if m.bot])}",
                "inline": True
            },
            {
                "name": "Channels",
                "value": f"Text: {len(guild.text_channels)}\nVoice: {len(guild.voice_channels)}\nCategories: {len(guild.categories)}",
                "inline": True
            },
            {
                "name": "Roles",
                "value": str(len(guild.roles)),
                "inline": True
            },
            {
                "name": "Boost Level",
                "value": f"Level {guild.premium_tier}",
                "inline": True
            }
        ],
        thumbnail=guild.icon.url if guild.icon else None
    )
    await ctx.send(embed=embed)


@commands.command(name="userinfo")
async def user_info(self, ctx, member: Optional[discord.Member] = None):
    """Show user information"""
    member = member or ctx.author
    user_info = format_user_info(member)

    embed = create_embed(
        title=f"User Information: {user_info['name']}",
        fields=[
            {
                "name": "Joined Server",
                "value": user_info["joined_at"],
                "inline": True
            },
            {
                "name": "Account Created",
                "value": member.created_at.strftime("%d.%m.%Y"),
                "inline": True
            },
            {
                "name": "Roles",
                "value": ", ".join(user_info["roles"]) if user_info["roles"] else "None",
                "inline": False
            }
        ],
        thumbnail=user_info["avatar"]
    )
    await ctx.send(embed=embed)


@commands.command(name="poll")
async def create_poll(self, ctx, question: str, *options):
    """Create a poll with up to 10 options"""
    if len(options) < 2:
        await ctx.send("❌ You need at least 2 options for a poll!")
        return

    if len(options) > 10:
        await ctx.send("❌ You can't have more than 10 options!")
        return

    # Create poll embed
    embed = create_embed(
        title="📊 Poll",
        description=question,
        fields=[
            {
                "name": f"{chr(0x1F1E6 + i)} {option}",
                "value": "0 votes",
                "inline": False
            } for i, option in enumerate(options)
        ],
        footer=f"Poll created by {ctx.author.name}"
    )


    poll_message = await ctx.send(embed=embed)
    for i in range(len(options)):
        await poll_message.add_reaction(chr(0x1F1E6 + i))


@commands.command(name="remind")
async def set_reminder(self, ctx, time: str, *, reminder: str):
    """Set a reminder (e.g. !remind 1h Buy groceries)"""
    seconds = parse_time(time)
    if not seconds:
        await ctx.send("❌ Invalid time format! Use: 1h, 30m, 1d")
        return


    reminder_time = datetime.utcnow() + timedelta(seconds=seconds)


    await self.db.add_reminder(ctx.author.id, reminder, reminder_time)

    embed = create_embed(
        title="Reminder Set",
        description=f"I'll remind you to: {reminder}",
        fields=[
            {
                "name": "Time",
                "value": get_time_until(reminder_time),
                "inline": True
            }
        ],
        color=discord.Color.green()
    )
    await ctx.send(embed=embed)


@commands.command(name="reminders")
async def list_reminders(self, ctx):
    """List your active reminders"""
    reminders = await self.db.get_reminders(ctx.author.id)

    if not reminders:
        await ctx.send("You don't have any active reminders!")
        return

    embed = create_embed(
        title="Your Reminders",
        description="\n".join(
            f"• {reminder['text']} ({get_time_until(reminder['time'])})"
            for reminder in reminders
        ),
        color=discord.Color.blue()
    )
    await ctx.send(embed=embed)


@commands.command(name="cancelreminder")
async def cancel_reminder(self, ctx, reminder_id: int):
    """Cancel a reminder by its ID"""
    success = await self.db.remove_reminder(ctx.author.id, reminder_id)

    if success:
        await ctx.send("✅ Reminder cancelled!")
    else:
        await ctx.send("❌ Reminder not found!")


@commands.command(name="weather")
async def get_weather(self, ctx, *, city: str):
    """Get weather information for a city"""

    embed = create_embed(
        title=f"Weather in {city}",
        description="Weather information will be displayed here",
        fields=[
            {
                "name": "Temperature",
                "value": "25°C",
                "inline": True
            },
            {
                "name": "Condition",
                "value": "Sunny",
                "inline": True
            },
            {
                "name": "Humidity",
                "value": "65%",
                "inline": True
            }
        ],
        color=discord.Color.blue()
    )
    await ctx.send(embed=embed)


@commands.command(name="translate")
async def translate_text(self, ctx, target_lang: str, *, text: str):
    """Translate text to another language"""

    embed = create_embed(
        title="Translation",
        fields=[
            {
                "name": "Original",
                "value": text,
                "inline": False
            },
            {
                "name": f"Translated to {target_lang}",
                "value": "Translated text will appear here",
                "inline": False
            }
        ],
        color=discord.Color.blue()
    )
    await ctx.send(embed=embed)


@commands.command(name="avatar")
async def show_avatar(self, ctx, member: Optional[discord.Member] = None):
    """Show user's avatar"""
    member = member or ctx.author

    embed = create_embed(
        title=f"{member.name}'s Avatar",
        color=discord.Color.blue()
    )
    embed.set_image(url=member.avatar.url if member.avatar else member.default_avatar.url)

    await ctx.send(embed=embed)


@commands.command(name="banner")
async def show_banner(self, ctx, member: Optional[discord.Member] = None):
    """Show user's banner"""
    member = member or ctx.author

    if not member.banner:
        await ctx.send("❌ This user doesn't have a banner!")
        return

    embed = create_embed(
        title=f"{member.name}'s Banner",
        color=discord.Color.blue()
    )
    embed.set_image(url=member.banner.url)

    await ctx.send(embed=embed)


@commands.command(name="servericon")
async def show_server_icon(self, ctx):
    """Show server icon"""
    if not ctx.guild.icon:
        await ctx.send("❌ This server doesn't have an icon!")
        return

    embed = create_embed(
        title=f"{ctx.guild.name}'s Icon",
        color=discord.Color.blue()
    )
    embed.set_image(url=ctx.guild.icon.url)

    await ctx.send(embed=embed)


@commands.command(name="serverbanner")
async def show_server_banner(self, ctx):
    """Show server banner"""
    if not ctx.guild.banner:
        await ctx.send("❌ This server doesn't have a banner!")
        return

    embed = create_embed(
        title=f"{ctx.guild.name}'s Banner",
        color=discord.Color.blue()
    )
    embed.set_image(url=ctx.guild.banner.url)

    await ctx.send(embed=embed)


bot.run(os.getenv('DISCORD_TOKEN'))
