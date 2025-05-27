import discord
from discord.ext import commands
from typing import Optional
import asyncio

from database import Database
from utils import create_embed, format_currency


class Roles(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()

    @commands.command(name="createrole")
    @commands.has_permissions(manage_roles=True)
    async def create_role(self, ctx, name: str, color: Optional[discord.Color] = None):
        """Create a new role"""
        try:
            role = await ctx.guild.create_role(
                name=name,
                color=color or discord.Color.blue(),
                reason=f"Created by user {ctx.author}"
            )

            await self.db.add_role(role.id, ctx.guild.id, ctx.author.id)

            embed = create_embed(
                title="Role created",
                description=f"Role {role.mention} has been successfully created!",
                color=role.color
            )

            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to create roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while creating the role!")

    @commands.command(name="deleterole")
    @commands.has_permissions(manage_roles=True)
    async def delete_role(self, ctx, role: discord.Role):
        """Delete a role"""
        try:
            role_data = await self.db.get_user_roles(ctx.author.id, ctx.guild.id)
            if not any(r["role_id"] == role.id and r["is_owner"] for r in role_data):
                await ctx.send("❌ You are not the owner of this role!")
                return

            await role.delete(reason=f"Deleted by user {ctx.author}")
            await self.db.delete_role(role.id, ctx.guild.id)

            embed = create_embed(
                title="Role deleted",
                description=f"Role {role.name} has been successfully deleted!",
                color=discord.Color.red()
            )

            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to delete roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while deleting the role!")

    @commands.command(name="enablerole")
    async def enable_role(self, ctx, role: discord.Role):
        """Enable a role"""
        role_data = await self.db.get_user_roles(ctx.author.id, ctx.guild.id)
        if not any(r["role_id"] == role.id and r["is_owner"] for r in role_data):
            await ctx.send("❌ You are not the owner of this role!")
            return

        await self.db.toggle_role(ctx.author.id, role.id, ctx.guild.id, True)

        embed = create_embed(
            title="Role enabled",
            description=f"Role {role.mention} is now active!",
            color=role.color
        )

        await ctx.send(embed=embed)

    @commands.command(name="disablerole")
    async def disable_role(self, ctx, role: discord.Role):
        """Disable a role"""
        role_data = await self.db.get_user_roles(ctx.author.id, ctx.guild.id)
        if not any(r["role_id"] == role.id and r["is_owner"] for r in role_data):
            await ctx.send("❌ You are not the owner of this role!")
            return

        await self.db.toggle_role(ctx.author.id, role.id, ctx.guild.id, False)

        embed = create_embed(
            title="Role disabled",
            description=f"Role {role.mention} is now inactive!",
            color=discord.Color.red()
        )

        await ctx.send(embed=embed)

    @commands.command(name="myroles")
    async def my_roles(self, ctx):
        """Show your roles"""
        roles = await self.db.get_user_roles(ctx.author.id, ctx.guild.id)

        if not roles:
            await ctx.send("You don't have any created roles yet!")
            return

        fields = []
        for role_data in roles:
            role = ctx.guild.get_role(role_data["role_id"])
            if role:
                status = "✅ Active" if role_data["is_enabled"] else "❌ Inactive"
                fields.append({
                    "name": role.name,
                    "value": f"ID: {role.id}\nStatus: {status}",
                    "inline": True
                })

        embed = create_embed(
            title="Your Roles",
            fields=fields
        )

        await ctx.send(embed=embed)

    @commands.command(name="buyrole")
    async def buy_role(self, ctx, name: str, color: Optional[discord.Color] = None):
        """Buy a new role"""
        balance = await self.db.get_balance(ctx.author.id)
        if balance["coins"] < 1000:
            await ctx.send("❌ You don't have enough coins! You need 1000 coins.")
            return

        try:
            role = await ctx.guild.create_role(
                name=name,
                color=color or discord.Color.blue(),
                reason=f"Purchased by user {ctx.author}"
            )

            await self.db.add_role(role.id, ctx.guild.id, ctx.author.id)
            await self.db.add_currency(ctx.author.id, "coins", -1000)

            embed = create_embed(
                title="Role purchased",
                description=f"You've purchased role {role.mention} for 1000 coins!",
                color=role.color
            )

            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to create roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while creating the role!")

    @commands.command(name="transferrole")
    async def transfer_role(self, ctx, role: discord.Role, member: discord.Member):
        """Transfer role to another user"""
        role_data = await self.db.get_user_roles(ctx.author.id, ctx.guild.id)
        if not any(r["role_id"] == role.id and r["is_owner"] for r in role_data):
            await ctx.send("❌ You are not the owner of this role!")
            return

        await self.db.add_role(role.id, ctx.guild.id, member.id)

        embed = create_embed(
            title="Role transferred",
            description=f"Role {role.mention} has been transferred to {member.mention}!",
            color=role.color
        )

        await ctx.send(embed=embed)

    @commands.command(name="rolecolor")
    async def role_color(self, ctx, role: discord.Role, color: discord.Color):
        """Change role color"""
        role_data = await self.db.get_user_roles(ctx.author.id, ctx.guild.id)
        if not any(r["role_id"] == role.id and r["is_owner"] for r in role_data):
            await ctx.send("❌ You are not the owner of this role!")
            return

        try:
            await role.edit(color=color)

            embed = create_embed(
                title="Role color changed",
                description=f"Color of role {role.mention} has been changed!",
                color=color
            )

            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to edit roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while editing the role!")

    @commands.command(name="rolename")
    async def role_name(self, ctx, role: discord.Role, *, new_name: str):
        """Change role name"""
        role_data = await self.db.get_user_roles(ctx.author.id, ctx.guild.id)
        if not any(r["role_id"] == role.id and r["is_owner"] for r in role_data):
            await ctx.send("❌ You are not the owner of this role!")
            return

        try:
            await role.edit(name=new_name)

            embed = create_embed(
                title="Role name changed",
                description=f"Role name has been changed to {new_name}!",
                color=role.color
            )

            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I don't have permissions to edit roles!")
        except discord.HTTPException:
            await ctx.send("❌ An error occurred while editing the role!")


def setup(bot):
    bot.add_cog(Roles(bot))
