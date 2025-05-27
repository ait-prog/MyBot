import aiosqlite
import asyncio
from datetime import datetime


class Database:
    def __init__(self):
        self.db_path = "bot.db"
        asyncio.create_task(self.init_db())

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Таблица пользователей
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER,
                    guild_id INTEGER,
                    coins INTEGER DEFAULT 0,
                    diamonds INTEGER DEFAULT 0,
                    messages INTEGER DEFAULT 0,
                    voice_time INTEGER DEFAULT 0,
                    last_voice_join TIMESTAMP,
                    PRIMARY KEY (user_id, guild_id)
                )
            ''')

            # Таблица ролей
            await db.execute('''
                CREATE TABLE IF NOT EXISTS roles (
                    role_id INTEGER,
                    guild_id INTEGER,
                    owner_id INTEGER,
                    is_enabled BOOLEAN DEFAULT TRUE,
                    PRIMARY KEY (role_id, guild_id)
                )
            ''')

            # Таблица комнат
            await db.execute('''
                CREATE TABLE IF NOT EXISTS rooms (
                    room_id INTEGER,
                    voice_id INTEGER,
                    role_id INTEGER,
                    guild_id INTEGER,
                    owner_id INTEGER,
                    name TEXT,
                    PRIMARY KEY (room_id, guild_id)
                )
            ''')

            # Таблица участников комнат
            await db.execute('''
                CREATE TABLE IF NOT EXISTS room_members (
                    user_id INTEGER,
                    room_id INTEGER,
                    guild_id INTEGER,
                    is_owner BOOLEAN DEFAULT FALSE,
                    is_coowner BOOLEAN DEFAULT FALSE,
                    total_time INTEGER DEFAULT 0,
                    last_join TIMESTAMP,
                    PRIMARY KEY (user_id, room_id, guild_id)
                )
            ''')

            # Таблица браков
            await db.execute('''
                CREATE TABLE IF NOT EXISTS marriages (
                    user1_id INTEGER,
                    user2_id INTEGER,
                    guild_id INTEGER,
                    PRIMARY KEY (user1_id, user2_id, guild_id)
                )
            ''')

            await db.commit()

    async def get_balance(self, user_id: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    "SELECT coins, diamonds FROM users WHERE user_id = ?",
                    (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {"coins": row["coins"], "diamonds": row["diamonds"]}
                return {"coins": 0, "diamonds": 0}

    async def add_currency(self, user_id: int, currency_type: str, amount: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            column = "coins" if currency_type == "coins" else "diamonds"

            # Обновляем или создаем запись
            await db.execute(f'''
                INSERT INTO users (user_id, {column})
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                {column} = {column} + ?
            ''', (user_id, amount, amount))

            await db.commit()

            # Получаем обновленный баланс
            async with db.execute(
                    f"SELECT coins, diamonds FROM users WHERE user_id = ?",
                    (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return {"coins": row["coins"], "diamonds": row["diamonds"]}

    async def get_user_stats(self, user_id: int, guild_id: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    "SELECT messages, voice_time FROM users WHERE user_id = ? AND guild_id = ?",
                    (user_id, guild_id)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "messages": row["messages"],
                        "voice_time": row["voice_time"]
                    }
                return {"messages": 0, "voice_time": 0}

    async def increment_messages(self, user_id: int, guild_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO users (user_id, guild_id, messages)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id, guild_id) DO UPDATE SET
                messages = messages + 1
            ''', (user_id, guild_id))
            await db.commit()

    async def update_voice_activity(self, user_id: int, guild_id: int, is_joining: bool):
        async with aiosqlite.connect(self.db_path) as db:
            if is_joining:
                await db.execute('''
                    INSERT INTO users (user_id, guild_id, last_voice_join)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id, guild_id) DO UPDATE SET
                    last_voice_join = ?
                ''', (user_id, guild_id, datetime.utcnow(), datetime.utcnow()))
            else:
                # Обновляем общее время в голосовом канале
                await db.execute('''
                    UPDATE users
                    SET voice_time = voice_time + strftime('%s', 'now') - strftime('%s', last_voice_join)
                    WHERE user_id = ? AND guild_id = ? AND last_voice_join IS NOT NULL
                ''', (user_id, guild_id))
            await db.commit()

    async def get_marriage(self, user_id: int) -> tuple:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                    "SELECT user1_id, user2_id FROM marriages WHERE user1_id = ? OR user2_id = ?",
                    (user_id, user_id)
            ) as cursor:
                return await cursor.fetchone()

    async def add_marriage(self, user1_id: int, user2_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO marriages (user1_id, user2_id) VALUES (?, ?)",
                (user1_id, user2_id)
            )
            await db.commit()

    async def remove_marriage(self, user1_id: int, user2_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM marriages WHERE (user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?)",
                (user1_id, user2_id, user2_id, user1_id)
            )
            await db.commit()

    async def get_user_roles(self, user_id: int, guild_id: int) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    "SELECT role_id, is_enabled, owner_id = ? as is_owner FROM roles WHERE guild_id = ?",
                    (user_id, guild_id)
            ) as cursor:
                return await cursor.fetchall()

    async def get_user_rooms(self, user_id: int, guild_id: int) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('''
                SELECT r.room_id, r.voice_id, r.role_id, r.name,
                       rm.is_owner, rm.is_coowner
                FROM rooms r
                JOIN room_members rm ON r.room_id = rm.room_id
                WHERE rm.user_id = ? AND r.guild_id = ?
            ''', (user_id, guild_id)) as cursor:
                return await cursor.fetchall()

    async def get_room_data(self, room_id: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    "SELECT * FROM rooms WHERE room_id = ?",
                    (room_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_member_data(self, user_id: int, room_id: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    "SELECT * FROM room_members WHERE user_id = ? AND room_id = ?",
                    (user_id, room_id)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_room_members(self, room_id: int) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    "SELECT user_id, is_owner, is_coowner, total_time, last_join FROM room_members WHERE room_id = ?",
                    (room_id,)
            ) as cursor:
                return await cursor.fetchall()

    async def add_role(self, role_id: int, guild_id: int, owner_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO roles (role_id, guild_id, owner_id) VALUES (?, ?, ?)",
                (role_id, guild_id, owner_id)
            )
            await db.commit()

    async def toggle_role(self, user_id: int, role_id: int, guild_id: int, is_enabled: bool):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE roles SET is_enabled = ? WHERE role_id = ? AND guild_id = ?",
                (is_enabled, role_id, guild_id)
            )
            await db.commit()

    async def delete_role(self, role_id: int, guild_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM roles WHERE role_id = ? AND guild_id = ?",
                (role_id, guild_id)
            )
            await db.commit()

    async def add_private_room(self, room_id: int, voice_id: int, role_id: int, guild_id: int, owner_id: int,
                               name: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO rooms (room_id, voice_id, role_id, guild_id, owner_id, name)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (room_id, voice_id, role_id, guild_id, owner_id, name))

            await db.execute('''
                INSERT INTO room_members (user_id, room_id, guild_id, is_owner)
                VALUES (?, ?, ?, TRUE)
            ''', (owner_id, room_id, guild_id))

            await db.commit()

    async def add_room_member(self, user_id: int, room_id: int, guild_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO room_members (user_id, room_id, guild_id)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, room_id, guild_id) DO NOTHING
            ''', (user_id, room_id, guild_id))
            await db.commit()

    async def update_room_name(self, room_id: int, new_name: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE rooms SET name = ? WHERE room_id = ?",
                (new_name, room_id)
            )
            await db.commit() 
