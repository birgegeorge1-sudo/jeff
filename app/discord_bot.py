import json
import os
import re

import discord
from discord.ext import commands

from .db import (
    SessionLocal,
    add_event,
    delete_contact,
    find_contact,
    get_active_contact,
    get_memories,
    get_or_create_contact,
    recent_messages,
    save_message,
    set_active_contact,
    upsert_memory,
)
from .engine import generate_reply

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
PREFIX = os.getenv("DISCORD_PREFIX", "!")
OWNER_DISCORD_USER_ID = os.getenv("OWNER_DISCORD_USER_ID", "")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

KNOWN_COMMANDS = {"newconvo", "deleteconvo", "active", "helpme", "reply"}


def owner_only():
    async def predicate(ctx):
        if not OWNER_DISCORD_USER_ID:
            return True
        return str(ctx.author.id) == OWNER_DISCORD_USER_ID
    return commands.check(predicate)


def _parse_contact_message(content: str):
    """Parse !Esther: message or !contEsther: message."""
    match = re.match(r"^!(?P<prefix>cont)?(?P<name>[^:\n]{1,255}):\s*(?P<message>.+)$", content, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    prefix = (match.group("prefix") or "").casefold()
    name = " ".join(match.group("name").strip().split())
    message = match.group("message").strip()
    if not name or not message or name.casefold() in KNOWN_COMMANDS:
        return None
    return name, message, bool(prefix)


def _parse_newconvo_payload(payload: str):
    if ":" not in payload:
        return " ".join(payload.strip().split()), None
    name, raw_history = payload.split(":", 1)
    name = " ".join(name.strip().split())
    raw_history = raw_history.strip()
    if not raw_history:
        return name, None
    try:
        parsed = json.loads(raw_history)
    except json.JSONDecodeError as exc:
        raise ValueError("History after the colon must be valid JSON, e.g. [\"Hi\", \"Hey!\"]") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Conversation history must be a non-empty JSON array.")
    normalized = []
    for index, item in enumerate(parsed):
        sender = "me" if index % 2 == 0 else "incoming"
        content = item
        if isinstance(item, dict):
            sender = str(item.get("sender", item.get("role", sender))).strip().lower()
            content = item.get("message", item.get("content", ""))
            if sender in {"her", "them", "contact", "incoming"}:
                sender = "incoming"
            elif sender in {"me", "i", "jeff", "owner"}:
                sender = "me"
            else:
                raise ValueError("Each history object must use sender/role 'me' or 'incoming'.")
        if not str(content).strip():
            raise ValueError(f"History item {index + 1} is empty.")
        normalized.append((sender, str(content).strip()))
    return name, normalized


async def _send_analysis(ctx, contact, incoming: str):
    with SessionLocal() as db:
        contact = find_contact(db, name=contact.display_name) or find_contact(db, contact.display_name)
        if not contact:
            await ctx.reply("❌ That contact does not exist yet. Create them with `!newconvo Name: [\"your first message\", \"their response\"]`.")
            return
        set_active_contact(db, contact, commit=False)
        incoming_message = save_message(db, contact.id, "incoming", incoming, platform="discord")
        result = generate_reply(contact, recent_messages(db, contact.id, 36), get_memories(db, contact.id, 50), incoming)
        for item in result.get("memory_updates", [])[:12]:
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                upsert_memory(db, contact.id, key, value, max(1, min(5, int(item.get("importance", 3) or 3))), incoming_message.id)
        vibe = result.get("vibe", {}) if isinstance(result.get("vibe", {}), dict) else {}
        boredom = result.get("boredom", {}) if isinstance(result.get("boredom", {}), dict) else {}
        contact.relationship_stage = result.get("relationship_stage", contact.relationship_stage)
        contact.vibe_score = float(vibe.get("momentum", contact.vibe_score) or contact.vibe_score)
        contact.engagement_score = float(vibe.get("reciprocity", contact.engagement_score) or contact.engagement_score)
        contact.date_readiness = float(result.get("date_readiness", contact.date_readiness) or contact.date_readiness)
        add_event(db, contact.id, "analysis", detail=str(boredom.get("reasons", [])), score=float(boredom.get("score", result.get("boredom_score", 0)) or 0))
        db.commit()

    await _render_analysis(ctx, contact, result)


async def _render_analysis(ctx, contact, result):
    embed = discord.Embed(title=f"💬 {contact.display_name}", description=result.get("reply", ""))
    embed.add_field(name="Stage", value=str(result.get("relationship_stage", "unknown")), inline=True)
    embed.add_field(name="Vibe", value=str(result.get("vibe", "unknown")), inline=True)
    embed.add_field(name="Boredom", value=f"{result.get('boredom_score', 0)}/100", inline=True)
    embed.add_field(name="Date readiness", value=f"{result.get('date_readiness', 0)}/100", inline=True)
    embed.add_field(name="Send mode", value=str(result.get("send_mode", "text")), inline=True)
    embed.add_field(name="Next move", value=result.get("next_goal", "Keep the interaction natural."), inline=False)
    if result.get("alternatives"):
        embed.add_field(name="Alternative drafts", value="\n\n".join(str(x) for x in result["alternatives"][:2])[:1900], inline=False)
    if result.get("voice_script"):
        embed.add_field(name="🎙️ Voice-note draft", value=str(result["voice_script"])[:1000], inline=False)
    if result.get("intimacy_guidance"):
        embed.add_field(name="❤️ Intimacy guidance", value=str(result["intimacy_guidance"])[:1000], inline=False)
    await ctx.reply(embed=embed)


@bot.event
async def on_ready():
    print(f"Discord adapter online as {bot.user} (id={bot.user.id})")


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if OWNER_DISCORD_USER_ID and str(message.author.id) != OWNER_DISCORD_USER_ID:
        return

    parsed = _parse_contact_message(message.content.strip())
    if parsed:
        name, incoming, _explicit_cont = parsed
        with SessionLocal() as db:
            contact = find_contact(db, name=name)
            if not contact:
                await message.channel.send(
                    f"❌ No contact named **{name}**. Create one first with `!newconvo {name}: [\\\"your first message\\\", \\\"their response\\\"]`."
                )
                return
        await _send_analysis(message, contact, incoming)
        return

    await bot.process_commands(message)


@bot.command(name="newconvo")
@owner_only()
async def newconvo(ctx, *, payload: str):
    """Create a contact and optionally seed the exact conversation history."""
    try:
        name, history = _parse_newconvo_payload(payload)
    except ValueError as exc:
        await ctx.reply(f"❌ {exc}")
        return
    if not name:
        await ctx.reply("❌ A contact name is required.")
        return
    with SessionLocal() as db:
        existing = find_contact(db, name=name)
        if existing:
            await ctx.reply(f"❌ **{existing.display_name}** already exists. Use `!{existing.display_name}: message` to continue that conversation.")
            return
        contact = get_or_create_contact(db, name=name)
        set_active_contact(db, contact, commit=False)
        if history:
            for sender, content in history:
                save_message(db, contact.id, sender, content, platform="discord")
        add_event(db, contact.id, "conversation_created", detail=f"seeded_messages={len(history or [])}")
        db.commit()
    seeded = len(history or [])
    await ctx.reply(
        f"✅ Created and activated **{contact.display_name}**.\n"
        f"Persisted `{seeded}` message{'s' if seeded != 1 else ''}.\n"
        f"Use `!{contact.display_name}: their next message` whenever they message you."
    )


@bot.command(name="deleteconvo")
@owner_only()
async def deleteconvo(ctx, *, name: str):
    with SessionLocal() as db:
        contact = find_contact(db, name=name)
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
        await ctx.reply(f"🎯 Active: **{contact.display_name}**\nStage: `{contact.relationship_stage}`")


@bot.command(name="reply")
@owner_only()
async def reply(ctx, *, incoming: str):
    with SessionLocal() as db:
        contact = get_active_contact(db)
    if not contact:
        await ctx.reply("❌ No active contact. Use `!Name: their message` instead.")
        return
    await _send_analysis(ctx, contact, incoming)


@bot.command(name="helpme")
@owner_only()
async def helpme(ctx):
    await ctx.reply(
        "**Relationship Copilot commands**\n"
        "`!newconvo Esther: [\"Hi Esther\", \"Hey!\"]` — create + persist history (alternates me/her)\n"
        "`!Esther: her next message` — add her message, persist it, and draft your reply\n"
        "`!contEsther: her next message` — same as above, explicit contact prefix\n"
        "`!deleteconvo Esther` — permanently delete contact + history\n"
        "`!active` — show the most recently addressed contact\n"
    )


def run():
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing")
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    run()
