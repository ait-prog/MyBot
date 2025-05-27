import discord
import random
from datetime import datetime, timedelta
from typing import Optional, Union, List
import asyncio


def format_time(seconds: int) -> str:
    """Format time into readable string"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def format_currency(amount: int, currency_type: str = "coins") -> str:
    """Format currency with emoji"""
    if currency_type == "coins":
        return f"{amount} 🪙"
    return f"{amount} 💎"


def create_embed(
        title: str,
        description: str = "",
        color: Union[discord.Color, int] = discord.Color.blue(),
        fields: List[dict] = None,
        thumbnail: str = None,
        footer: str = None
) -> discord.Embed:
    """Create a beautiful embed"""
    embed = discord.Embed(
        title=title,
        description=description,
        color=color
    )

    if fields:
        for field in fields:
            embed.add_field(
                name=field["name"],
                value=field["value"],
                inline=field.get("inline", True)
            )

    if thumbnail:
        embed.set_thumbnail(url=thumbnail)

    if footer:
        embed.set_footer(text=footer)

    return embed


def get_random_reward() -> tuple:
    """Generate random reward"""
    reward_type = random.choices(
        ["coins", "diamonds"],
        weights=[0.8, 0.2]
    )[0]

    if reward_type == "coins":
        amount = random.randint(10, 100)
    else:
        amount = random.randint(1, 5)

    return reward_type, amount


def calculate_voice_reward(seconds: int) -> int:
    """Calculate reward for voice channel time"""
    return (seconds // 60) * 2  # 2 coins per minute


def is_valid_room_name(name: str) -> bool:
    """Validate room name"""
    return 3 <= len(name) <= 32 and all(c.isalnum() or c in " -_" for c in name)


def get_role_color(role: discord.Role) -> discord.Color:
    """Get role color or return default"""
    return role.color if role.color != discord.Color.default() else discord.Color.blue()


def format_user_info(user: discord.Member) -> dict:
    """Format user information"""
    return {
        "name": str(user),
        "id": user.id,
        "avatar": user.avatar.url if user.avatar else None,
        "joined_at": user.joined_at.strftime("%d.%m.%Y") if user.joined_at else "Unknown",
        "roles": [role.name for role in user.roles[1:]]  # Skip @everyone
    }


def get_time_until(target_time: datetime) -> str:
    """Return time until specified moment"""
    now = datetime.utcnow()
    delta = target_time - now

    if delta.total_seconds() <= 0:
        return "Now"

    days = delta.days
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def parse_time(time_str: str) -> Optional[int]:
    """Parse time string into seconds"""
    try:
        if time_str.endswith("m"):
            return int(time_str[:-1]) * 60
        elif time_str.endswith("h"):
            return int(time_str[:-1]) * 3600
        elif time_str.endswith("d"):
            return int(time_str[:-1]) * 86400
        else:
            return int(time_str)
    except ValueError:
        return None


def get_cooldown_text(seconds: int) -> str:
    """Return cooldown time text"""
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minutes"
    else:
        hours = seconds // 3600
        return f"{hours} hours"


def format_leaderboard(entries: List[dict], title: str) -> str:
    """Format leaderboard display"""
    if not entries:
        return "No data"

    result = [f"**{title}**\n"]
    for i, entry in enumerate(entries, 1):
        result.append(f"{i}. {entry['name']}: {entry['value']}")

    return "\n".join(result)


def get_random_duel_message(winner: str, loser: str) -> str:
    """Return random duel message"""
    messages = [
        f"{winner} defeated {loser} in an epic battle!",
        f"{winner} destroyed {loser} with one hit!",
        f"{winner} showed {loser} who's the boss!",
        f"{winner} won the duel against {loser}!",
        f"{loser} couldn't withstand {winner}'s power!"
    ]
    return random.choice(messages)


def get_random_marriage_message(user1: str, user2: str) -> str:
    """Return random marriage message"""
    messages = [
        f"💕 {user1} and {user2} are now husband and wife!",
        f"💑 {user1} proposed to {user2} and got accepted!",
        f"💝 {user1} and {user2} are now bound by marriage!",
        f"💖 {user1} and {user2} are now one family!",
        f"💗 {user1} and {user2} vowed eternal love!"
    ]
    return random.choice(messages)


def get_random_divorce_message(user1: str, user2: str) -> str:
    """Return random divorce message"""
    messages = [
        f"💔 {user1} and {user2} decided to divorce...",
        f"😢 {user1} and {user2} are no longer together...",
        f"😭 {user1} and {user2} broke their relationship...",
        f"😞 {user1} and {user2} filed for divorce...",
        f"😔 {user1} and {user2} decided to go separate ways..."
    ]
    return random.choice(messages)


def get_random_room_name() -> str:
    """Generate random room name"""
    adjectives = ["Cozy", "Warm", "Bright", "Quiet", "Peaceful", "Calm"]
    nouns = ["nest", "haven", "place", "space", "corner", "home"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


def format_voice_stats(seconds: int) -> str:
    """Format voice channel statistics"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60

    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def get_random_welcome_message(user: str) -> str:
    """Return random welcome message"""
    messages = [
        f"Welcome, {user}! 🎉",
        f"Greetings, {user}! ✨",
        f"Glad to see you, {user}! 🌟",
        f"Welcome back, {user}! 🎊",
        f"Hello, {user}! Welcome! 🎈"
    ]
    return random.choice(messages)


def get_random_farewell_message(user: str) -> str:
    """Return random farewell message"""
    messages = [
        f"Goodbye, {user}! 👋",
        f"See you again, {user}! 💫",
        f"Take care, {user}! ✨",
        f"Come back soon, {user}! 🌟",
        f"Bye, {user}! Take care! 🎈"
    ]
    return random.choice(messages)
