import asyncio
import os
import discord
from discord.ext import commands

from .db import SessionLocal, get_or_create_contact, find_contact, get_active_contact, set_active_contact, delete_contact
from .engine import generate_reply

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
PREFIX = os.getenv("DISCORD_PREFIX", "!")
OWNER_DISCORD_USER_ID = os.getenv("OWNER_DISCORD_USER_ID", "")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents)


def owner_only():
    async def predicate(ctx):
        if not OWNER_DISCORD_USER_ID:
            return True
        return str(ctx.author.id) == OWNER_DISCORD_USER_ID
    return commands.check(predicate)


@bot.event
async def on_ready():
    print(f"Discord adapter online as {bot.user} (id={bot.user.id})")


@bot.command(name="newconvo")
@owner_only()
async def newconvo(ctx, *, name: str):
    """Create and activate a new person. Never overwrites another person's history."""
    with SessionLocal() as db:
        contact = get_or_create_contact(db, name)
        set_active_contact(db, contact)
        await ctx.reply(
            f"🧠 New conversation created and activated: **{contact.display_name}**\n"
            f"Use `!contconvo {contact.display_name}` later to resume."
        )


@bot.command(name="contconvo")
@owner_only()
async def contconvo(ctx, *, name: str):
    """Activate an existing person. This command never creates."""
    from sqlalchemy import select
    from .db import Contact
    with SessionLocal() as db:
        canonical = " ".join(name.strip().split()).casefold()
        contact = db.scalar(select(Contact).where(Contact.canonical_name == canonical))
        if not contact:
            await ctx.reply(
                f"❌ No existing conversation for **{name}**. "
                f"Use `!newconvo {name}` only when you really mean to create one."
            )
            return
        set_active_contact(db, contact)
        await ctx.reply(f"🔄 Active conversation: **{contact.display_name}**")


@bot.command(name="deleteconvo")
@owner_only()
async def deleteconvo(ctx, *, name: str):
    """Permanently delete one contact and all associated messages, memories and events."""
    with SessionLocal() as db:
        contact = find_contact(db, name)
        if not contact:
            await ctx.reply(f"❌ No existing conversation for **{name}**.")
            return
        display = contact.display_name
        delete_contact(db, contact.id)
        await ctx.reply(f"🗑️ Permanently deleted **{display}** and its stored conversation memory/history.")


@bot.command(name="active")
@owner_only()
async def active(ctx):
    with SessionLocal() as db:
        contact = get_active_contact(db)
        if not contact:
            await ctx.reply("No active conversation.")
            return
        await ctx.reply(
            f"🎯 Active: **{contact.display_name}**\n"
            f"Stage: `{contact.relationship_stage}`"
        )


@bot.command(name="reply")
@owner_only()
async def reply(ctx, *, incoming: str):
    """Analyze an incoming message and return a draft + conversation intelligence."""
    with SessionLocal() as db:
        contact = get_active_contact(db)
        if not contact:
            await ctx.reply("❌ Activate a conversation first with `!newconvo Name` or `!contconvo Name`.")
            return

        # Reuse the same safety boundary as the HTTP layer: only active-contact memory is supplied.
        from .db import save_message, recent_messages, get_memories
        save_message(db, contact.id, "incoming", incoming)
        result = generate_reply(
            contact,
            recent_messages(db, contact.id, 30),
            get_memories(db, contact.id, 30),
            incoming,
        )

        for item in result.get("memory_updates", []):
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                from .db import upsert_memory
                upsert_memory(
                    db, contact.id, key, value,
                    max(1, min(5, int(item.get("importance", 3))))
                )

        contact.relationship_stage = result.get(
            "relationship_stage", contact.relationship_stage
        )
        db.commit()

    embed = discord.Embed(title=f"💬 {contact.display_name}", description=result.get("reply", ""))
    embed.add_field(name="Stage", value=str(result.get("relationship_stage", "unknown")), inline=True)
    embed.add_field(name="Vibe", value=str(result.get("vibe", "unknown")), inline=True)
    embed.add_field(name="Boredom", value=f"{result.get('boredom_score', 0)}/100", inline=True)
    embed.add_field(name="Date readiness", value=f"{result.get('date_readiness', 0)}/100", inline=True)
    embed.add_field(name="Send mode", value=str(result.get("send_mode", "text")), inline=True)
    embed.add_field(name="Next move", value=result.get("next_goal", "Keep the interaction natural."), inline=False)

    voice_script = result.get("voice_script", "")
    if voice_script:
        embed.add_field(name="🎙️ Voice-note draft", value=voice_script[:1000], inline=False)

    intimacy = result.get("intimacy_guidance", "")
    if intimacy:
        embed.add_field(name="❤️ Intimacy guidance", value=intimacy[:1000], inline=False)

    await ctx.reply(embed=embed)


@bot.command(name="helpme")
@owner_only()
async def helpme(ctx):
    await ctx.reply(
        "**Romance Copilot commands**\n"
        "`!newconvo Esther` — create + activate\n"
        "`!contconvo Esther` — activate an existing conversation\n"
        "`!active` — show active person\n"
        "`!reply <incoming message>` — analyze + draft\n"
    )


def run():
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing")
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    run()
