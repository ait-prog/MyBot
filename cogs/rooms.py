import discord
from discord.ext import commands
from typing import Optional
import asyncio

from database import Database
from utils import (
    create_embed,
    format_currency,
    is_valid_room_name,
    get_random_room_name
)


class Rooms(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()

    @commands.command(name="createroom")
    async def create_room(self, ctx, name: Optional[str] = None):
        """Create a private room"""
        balance = await self.db.get_balance(ctx.author.id)
        if balance["coins"] < 2000:
            await ctx.send("❌ You don't have enough coins! You need 2000 coins.")
            return

        if name and not is_valid_room_name(name):
            await ctx.send("❌ Invalid room name! Use only letters, numbers, spaces and -_ symbols")
            return

        try:
            # Create role for the room
            role = await ctx.guild.create_role(
                name=f"Room-{ctx.author.name}",
                color=discord.Color.blue(),
                reason=f"Created for private room by user {ctx.author}"
            )

            # Create voice channel
            voice_channel = await ctx.guild.create_voice_channel(
                name=name or get_random_room_name(),
                overwrites={
                    ctx.guild.default_role: discord.PermissionOverwrite(connect=False),
                    role: discord.PermissionOverwrite(connect=True),
                    ctx.author: discord.PermissionOverwrite(manage_channels=True)
                }
            )

            # Add role to user
            await ctx.author.add_roles(role)

            # Save information to database
            await self.db.add_private_room(
                voice_channel.id,
                voice_channel.id,
                role.id,
                ctx.guild.id,
                ctx.author.id,
                voice_channel.name
            )

            # Deduct coins
            await self.db.add_currency(ctx.author.id, "coins", -2000)

            embed = create_embed(
                title="Room created",
                description=f"You've created private room {voice_channel.mention}!",
                fields=[
                    {
                        "name": "Role",
                        "value": role.mention,
                        "inline": True
                    },
                    {
                        "name": "Cost",
                        "value": format_currency(2000),
                        "inline": True
                    }
                ]
            )

            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to create channels and roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while creating the room!")

    @commands.command(name="deleteroom")
    async def delete_room(self, ctx):
        """Delete your private room"""
        rooms = await self.db.get_user_rooms(ctx.author.id, ctx.guild.id)
        if not rooms:
            await ctx.send("❌ You don't have any private rooms!")
            return

        room = rooms[0]  # Take the first room
        room_data = await self.db.get_room_data(room["room_id"])

        if not room_data or room_data["owner_id"] != ctx.author.id:
            await ctx.send("❌ You are not the owner of this room!")
            return

        try:
            # Delete channel and role
            voice_channel = ctx.guild.get_channel(room["voice_id"])
            role = ctx.guild.get_role(room["role_id"])

            if voice_channel:
                await voice_channel.delete()
            if role:
                await role.delete()

            embed = create_embed(
                title="Room deleted",
                description="Your private room has been successfully deleted!",
                color=discord.Color.red()
            )

            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to delete channels and roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while deleting the room!")

    @commands.command(name="invite")
    async def invite_to_room(self, ctx, member: discord.Member):
        """Invite user to your room"""
        rooms = await self.db.get_user_rooms(ctx.author.id, ctx.guild.id)
        if not rooms:
            await ctx.send("❌ You don't have any private rooms!")
            return

        room = rooms[0]  # Take the first room
        room_data = await self.db.get_room_data(room["room_id"])

        if not room_data or room_data["owner_id"] != ctx.author.id:
            await ctx.send("❌ You are not the owner of this room!")
            return

        try:
            # Add role to user
            role = ctx.guild.get_role(room["role_id"])
            if role:
                await member.add_roles(role)
                await self.db.add_room_member(member.id, room["room_id"], ctx.guild.id)

                embed = create_embed(
                    title="Room invitation",
                    description=f"{member.mention} has been invited to your private room!",
                    color=discord.Color.green()
                )

                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ Room role not found!")

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to manage roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while inviting the user!")

    @commands.command(name="kick")
    async def kick_from_room(self, ctx, member: discord.Member):
        """Kick user from your room"""
        rooms = await self.db.get_user_rooms(ctx.author.id, ctx.guild.id)
        if not rooms:
            await ctx.send("❌ You don't have any private rooms!")
            return

        room = rooms[0]  # Take the first room
        room_data = await self.db.get_room_data(room["room_id"])

        if not room_data or room_data["owner_id"] != ctx.author.id:
            await ctx.send("❌ You are not the owner of this room!")
            return

        try:
            # Remove role from user
            role = ctx.guild.get_role(room["role_id"])
            if role:
                await member.remove_roles(role)

                embed = create_embed(
                    title="User kicked",
                    description=f"{member.mention} has been kicked from your private room!",
                    color=discord.Color.red()
                )

                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ Room role not found!")

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to manage roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while kicking the user!")

    @commands.command(name="rename")
    async def rename_room(self, ctx, *, new_name: str):
        """Rename your room"""
        if not is_valid_room_name(new_name):
            await ctx.send("❌ Invalid room name! Use only letters, numbers, spaces and -_ symbols")
            return

        rooms = await self.db.get_user_rooms(ctx.author.id, ctx.guild.id)
        if not rooms:
            await ctx.send("❌ You don't have any private rooms!")
            return

        room = rooms[0]  # Take the first room
        room_data = await self.db.get_room_data(room["room_id"])

        if not room_data or room_data["owner_id"] != ctx.author.id:
            await ctx.send("❌ You are not the owner of this room!")
            return

        try:
            # Rename channel
            voice_channel = ctx.guild.get_channel(room["voice_id"])
            if voice_channel:
                await voice_channel.edit(name=new_name)
                await self.db.update_room_name(room["room_id"], new_name)

                embed = create_embed(
                    title="Room renamed",
                    description=f"Your private room has been renamed to {new_name}!",
                    color=discord.Color.green()
                )

                await ctx.send(embed=embed)
            else:
                await ctx.send("❌ Voice channel not found!")

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to manage channels!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while renaming the room!")

    @commands.command(name="myrooms")
    async def my_rooms(self, ctx):
        """Show your private rooms"""
        rooms = await self.db.get_user_rooms(ctx.author.id, ctx.guild.id)

        if not rooms:
            await ctx.send("You don't have any private rooms yet!")
            return

        fields = []
        for room in rooms:
            voice_channel = ctx.guild.get_channel(room["voice_id"])
            role = ctx.guild.get_role(room["role_id"])

            if voice_channel and role:
                fields.append({
                    "name": voice_channel.name,
                    "value": f"Channel: {voice_channel.mention}\nRole: {role.mention}",
                    "inline": True
                })

        embed = create_embed(
            title="Your Private Rooms",
            fields=fields
        )

        await ctx.send(embed=embed)

    @commands.command(name="members")
    async def room_members(self, ctx):
        """Show members of your room"""
        rooms = await self.db.get_user_rooms(ctx.author.id, ctx.guild.id)
        if not rooms:
            await ctx.send("❌ You don't have any private rooms!")
            return

        room = rooms[0]  # Take the first room
        members = await self.db.get_room_members(room["room_id"])

        if not members:
            await ctx.send("There are no members in the room yet!")
            return

        fields = []
        for member_data in members:
            member = ctx.guild.get_member(member_data["user_id"])
            if member:
                role = "👑 Owner" if member_data["is_owner"] else "⭐ Co-owner" if member_data["is_coowner"] else "👤 Member"
                fields.append({
                    "name": member.name,
                    "value": f"Role: {role}\nTime in room: {member_data['total_time']} minutes",
                    "inline": True
                })

        embed = create_embed(
            title="Room Members",
            fields=fields
        )

        await ctx.send(embed=embed)


def setup(bot):
    bot.add_cog(Rooms(bot))
