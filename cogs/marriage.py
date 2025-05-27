import discord
from discord.ext import commands
from typing import Optional
import asyncio

from database import Database
from utils import (
    create_embed,
    format_currency,
    get_random_marriage_message,
    get_random_divorce_message
)


class Marriage(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()
        self.marriage_cooldowns = {}

    @commands.command(name="propose")
    async def propose(self, ctx, member: discord.Member):
        """Propose marriage to another user"""
        if member.id == ctx.author.id:
            await ctx.send("❌ You cannot propose to yourself!")
            return

        # Check if either user is already married
        author_marriage = await self.db.get_marriage(ctx.author.id)
        member_marriage = await self.db.get_marriage(member.id)

        if author_marriage:
            await ctx.send("❌ You are already married!")
            return

        if member_marriage:
            await ctx.send("❌ This user is already married!")
            return

        # Request confirmation
        embed = create_embed(
            title="Marriage Proposal",
            description=f"{ctx.author.name} wants to marry you!\n\n{member.name}, do you accept?",
            color=discord.Color.pink()
        )

        message = await ctx.send(embed=embed)
        await message.add_reaction("✅")
        await message.add_reaction("❌")

        def check(reaction, user):
            return user == member and str(reaction.emoji) in ["✅", "❌"]

        try:
            reaction, user = await self.bot.wait_for("reaction_add", timeout=30.0, check=check)

            if str(reaction.emoji) == "✅":
                # Create marriage
                await self.db.add_marriage(ctx.author.id, member.id)

                embed = create_embed(
                    title="Marriage Accepted",
                    description=get_random_marriage_message(ctx.author.name, member.name),
                    color=discord.Color.pink()
                )
            else:
                embed = create_embed(
                    title="Proposal Declined",
                    description=f"{member.name} declined your proposal...",
                    color=discord.Color.red()
                )

            await message.edit(embed=embed)

        except asyncio.TimeoutError:
            embed = create_embed(
                title="Proposal Cancelled",
                description=f"{member.name} didn't respond to your proposal...",
                color=discord.Color.red()
            )
            await message.edit(embed=embed)

    @commands.command(name="divorce")
    async def divorce(self, ctx):
        """Divorce your spouse"""
        marriage = await self.db.get_marriage(ctx.author.id)
        if not marriage:
            await ctx.send("❌ You are not married!")
            return

        # Determine spouse
        spouse_id = marriage[1] if marriage[0] == ctx.author.id else marriage[0]
        spouse = ctx.guild.get_member(spouse_id)

        if not spouse:
            await ctx.send("❌ Your spouse is not found on the server!")
            return

        # Request confirmation
        embed = create_embed(
            title="Divorce",
            description=f"Are you sure you want to divorce {spouse.name}?",
            color=discord.Color.red()
        )

        message = await ctx.send(embed=embed)
        await message.add_reaction("✅")
        await message.add_reaction("❌")

        def check(reaction, user):
            return user == ctx.author and str(reaction.emoji) in ["✅", "❌"]

        try:
            reaction, user = await self.bot.wait_for("reaction_add", timeout=30.0, check=check)

            if str(reaction.emoji) == "✅":
                # End marriage
                await self.db.remove_marriage(ctx.author.id, spouse_id)

                embed = create_embed(
                    title="Marriage Ended",
                    description=get_random_divorce_message(ctx.author.name, spouse.name),
                    color=discord.Color.red()
                )
            else:
                embed = create_embed(
                    title="Divorce Cancelled",
                    description="You decided to stay married!",
                    color=discord.Color.green()
                )

            await message.edit(embed=embed)

        except asyncio.TimeoutError:
            embed = create_embed(
                title="Divorce Cancelled",
                description="Time for confirmation has expired!",
                color=discord.Color.red()
            )
            await message.edit(embed=embed)

    @commands.command(name="marriage")
    async def marriage_info(self, ctx, member: Optional[discord.Member] = None):
        """Show marriage information"""
        member = member or ctx.author
        marriage = await self.db.get_marriage(member.id)

        if not marriage:
            await ctx.send(f"❌ {member.name} is not married!")
            return

        # Determine spouse
        spouse_id = marriage[1] if marriage[0] == member.id else marriage[0]
        spouse = ctx.guild.get_member(spouse_id)

        if not spouse:
            await ctx.send("❌ Spouse not found on the server!")
            return

        embed = create_embed(
            title="Marriage Information",
            description=f"💑 {member.name} and {spouse.name} are married!",
            color=discord.Color.pink()
        )

        await ctx.send(embed=embed)

    @commands.command(name="kiss")
    async def kiss(self, ctx, member: discord.Member):
        """Kiss your spouse"""
        marriage = await self.db.get_marriage(ctx.author.id)
        if not marriage:
            await ctx.send("❌ You are not married!")
            return

        # Check if user is spouse
        spouse_id = marriage[1] if marriage[0] == ctx.author.id else marriage[0]
        if member.id != spouse_id:
            await ctx.send("❌ You can only kiss your spouse!")
            return

        embed = create_embed(
            title="Kiss",
            description=f"💋 {ctx.author.name} kissed {member.name}!",
            color=discord.Color.pink()
        )

        await ctx.send(embed=embed)

    @commands.command(name="hug")
    async def hug(self, ctx, member: discord.Member):
        """Hug your spouse"""
        marriage = await self.db.get_marriage(ctx.author.id)
        if not marriage:
            await ctx.send("❌ You are not married!")
            return

        # Check if user is spouse
        spouse_id = marriage[1] if marriage[0] == ctx.author.id else marriage[0]
        if member.id != spouse_id:
            await ctx.send("❌ You can only hug your spouse!")
            return

        embed = create_embed(
            title="Hug",
            description=f"🤗 {ctx.author.name} hugged {member.name}!",
            color=discord.Color.pink()
        )

        await ctx.send(embed=embed)


def setup(bot):
    bot.add_cog(Marriage(bot)) 
