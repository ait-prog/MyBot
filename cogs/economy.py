import discord
from discord.ext import commands
import random
from datetime import datetime, timedelta
from typing import Optional
import asyncio

from database import Database
from utils import (
    format_currency,
    create_embed,
    get_random_reward,
    calculate_voice_reward,
    get_cooldown_text
)


class Economy(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()
        self.daily_cooldowns = {}
        self.work_cooldowns = {}
        self.rob_cooldowns = {}
        self.duel_cooldowns = {}

    @commands.command(name="balance", aliases=["bal"])
    async def balance(self, ctx, member: Optional[discord.Member] = None):
        """Shows user's balance"""
        member = member or ctx.author
        balance = await self.db.get_balance(member.id)

        embed = create_embed(
            title=f"Balance of {member.name}",
            fields=[
                {
                    "name": "Coins",
                    "value": format_currency(balance["coins"]),
                    "inline": True
                },
                {
                    "name": "Diamonds",
                    "value": format_currency(balance["diamonds"], "diamonds"),
                    "inline": True
                }
            ],
            thumbnail=member.avatar.url if member.avatar else None
        )

        await ctx.send(embed=embed)

    @commands.command(name="daily")
    @commands.cooldown(1, 86400, commands.BucketType.user)  # 24 hours
    async def daily(self, ctx):
        """Get daily reward"""
        reward_type, amount = get_random_reward()
        new_balance = await self.db.add_currency(ctx.author.id, reward_type, amount)

        embed = create_embed(
            title="Daily Reward",
            description=f"You received {format_currency(amount, reward_type)}!",
            fields=[
                {
                    "name": "Your balance",
                    "value": format_currency(new_balance[reward_type], reward_type)
                }
            ]
        )

        await ctx.send(embed=embed)

    @commands.command(name="work")
    @commands.cooldown(1, 3600, commands.BucketType.user)  # 1 hour
    async def work(self, ctx):
        """Earn coins"""
        amount = random.randint(50, 200)
        new_balance = await self.db.add_currency(ctx.author.id, "coins", amount)

        embed = create_embed(
            title="Work",
            description=f"You earned {format_currency(amount)}!",
            fields=[
                {
                    "name": "Your balance",
                    "value": format_currency(new_balance["coins"])
                }
            ]
        )

        await ctx.send(embed=embed)

    @commands.command(name="give", aliases=["pay"])
    async def give(self, ctx, member: discord.Member, amount: int):
        """Transfer coins to another user"""
        if amount <= 0:
            await ctx.send("❌ Amount must be positive!")
            return

        sender_balance = await self.db.get_balance(ctx.author.id)
        if sender_balance["coins"] < amount:
            await ctx.send("❌ You don't have enough coins!")
            return

        # Deduct coins from sender
        await self.db.add_currency(ctx.author.id, "coins", -amount)
        # Add coins to receiver
        receiver_balance = await self.db.add_currency(member.id, "coins", amount)

        embed = create_embed(
            title="Coin Transfer",
            description=f"You transferred {format_currency(amount)} to {member.name}!",
            fields=[
                {
                    "name": f"Balance of {member.name}",
                    "value": format_currency(receiver_balance["coins"])
                }
            ]
        )

        await ctx.send(embed=embed)

    @commands.command(name="rob")
    @commands.cooldown(1, 7200, commands.BucketType.user)  # 2 hours
    async def rob(self, ctx, member: discord.Member):
        """Attempt to rob another user"""
        if member.id == ctx.author.id:
            await ctx.send("❌ You can't rob yourself!")
            return

        target_balance = await self.db.get_balance(member.id)
        if target_balance["coins"] < 100:
            await ctx.send("❌ This user has too few coins to rob!")
            return

        # 50% success chance
        if random.random() < 0.5:
            amount = min(target_balance["coins"], random.randint(100, 500))
            # Deduct coins from victim
            await self.db.add_currency(member.id, "coins", -amount)
            # Add coins to robber
            robber_balance = await self.db.add_currency(ctx.author.id, "coins", amount)

            embed = create_embed(
                title="Successful Robbery",
                description=f"You successfully robbed {member.name} and got {format_currency(amount)}!",
                fields=[
                    {
                        "name": "Your balance",
                        "value": format_currency(robber_balance["coins"])
                    }
                ],
                color=discord.Color.green()
            )
        else:
            fine = random.randint(100, 300)
            robber_balance = await self.db.add_currency(ctx.author.id, "coins", -fine)

            embed = create_embed(
                title="Failed Robbery",
                description=f"You were caught trying to rob {member.name}! You paid a fine of {format_currency(fine)}.",
                fields=[
                    {
                        "name": "Your balance",
                        "value": format_currency(robber_balance["coins"])
                    }
                ],
                color=discord.Color.red()
            )

        await ctx.send(embed=embed)

    @commands.command(name="duel")
    @commands.cooldown(1, 3600, commands.BucketType.user)  # 1 hour
    async def duel(self, ctx, member: discord.Member, amount: int):
        """Challenge another user to a duel"""
        if member.id == ctx.author.id:
            await ctx.send("❌ You can't duel yourself!")
            return

        if amount <= 0:
            await ctx.send("❌ Amount must be positive!")
            return

        # Check balances
        author_balance = await self.db.get_balance(ctx.author.id)
        member_balance = await self.db.get_balance(member.id)

        if author_balance["coins"] < amount:
            await ctx.send("❌ You don't have enough coins!")
            return

        if member_balance["coins"] < amount:
            await ctx.send("❌ Your opponent doesn't have enough coins!")
            return

        # Request confirmation
        embed = create_embed(
            title="Duel Challenge",
            description=f"{ctx.author.name} challenges {member.name} to a duel!\nStake: {format_currency(amount)}\n\n{member.name}, do you accept the challenge?",
            color=discord.Color.gold()
        )

        message = await ctx.send(embed=embed)
        await message.add_reaction("✅")
        await message.add_reaction("❌")

        def check(reaction, user):
            return user == member and str(reaction.emoji) in ["✅", "❌"]

        try:
            reaction, user = await self.bot.wait_for("reaction_add", timeout=30.0, check=check)

            if str(reaction.emoji) == "✅":
                # Determine winner
                winner = random.choice([ctx.author, member])
                loser = member if winner == ctx.author else ctx.author

                # Transfer coins
                await self.db.add_currency(winner.id, "coins", amount)
                await self.db.add_currency(loser.id, "coins", -amount)

                embed = create_embed(
                    title="Duel Result",
                    description=f"{winner.name} defeated {loser.name} and got {format_currency(amount)}!",
                    color=discord.Color.green()
                )
            else:
                embed = create_embed(
                    title="Duel Canceled",
                    description=f"{member.name} declined the duel!",
                    color=discord.Color.red()
                )

            await message.edit(embed=embed)

        except asyncio.TimeoutError:
            embed = create_embed(
                title="Duel Canceled",
                description=f"{member.name} didn't respond to the challenge!",
                color=discord.Color.red()
            )
            await message.edit(embed=embed)

    @commands.command(name="shop")
    async def shop(self, ctx):
        """Show the shop"""
        embed = create_embed(
            title="Shop",
            description="Available items:",
            fields=[
                {
                    "name": "🎨 Colored Role",
                    "value": "1000 coins",
                    "inline": True
                },
                {
                    "name": "🎭 Special Role",
                    "value": "500 coins",
                    "inline": True
                },
                {
                    "name": "🎪 Private Room",
                    "value": "2000 coins",
                    "inline": True
                }
            ]
        )

        await ctx.send(embed=embed)

    @commands.command(name="buy")
    async def buy(self, ctx, item: str):
        """Buy an item from the shop"""
        # TODO: Implement item purchasing
        await ctx.send("❌ This feature is under development!")

    @commands.command(name="leaderboard", aliases=["lb"])
    async def leaderboard(self, ctx):
        """Show the leaderboard"""
        # TODO: Implement leaderboard
        await ctx.send("❌ This feature is under development!")

    @daily.error
    @work.error
    @rob.error
    @duel.error
    async def cooldown_error(self, ctx, error):
        """Handle command cooldown errors"""
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"❌ Please wait {get_cooldown_text(int(error.retry_after))}!")


def setup(bot):
    bot.add_cog(Economy(bot))
