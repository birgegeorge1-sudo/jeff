import json
import os
import re

import discord
from discord.ext import commands

from .db import (
    SessionLocal,
    get_or_create_contact,
    find_contact,
    get_active_contact,
    set_active_contact,
    delete_contact,
    save_message,
    recent_messages,
    get_memories,
    upsert_memory,
    add_event,
)
from .engine import generate_reply
from .persona import load_persona
from .planner import boredom_heuristic, question_frequency


DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
PREFIX = os.getenv("DISCORD_PREFIX", "!")
OWNER_DISCORD_USER_ID = os.getenv("OWNER_DISCORD_USER_ID", "")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents,
    help_command=None,
)


def owner_only():
    async def predicate(ctx):
        if not OWNER_DISCORD_USER_ID:
            return True
        return str(ctx.author.id) == OWNER_DISCORD_USER_ID

    return commands.check(predicate)


def normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


def parse_seed_messages(raw: str):
    """
    Parse the message array from:

        !newconvo Esther: ["Hey Esther", "Hey!", "I've been good"]

    JSON is preferred, but a Python-literal-looking list is also accepted
    for convenience.
    """
    raw = raw.strip()

    try:
        messages = json.loads(raw)
    except json.JSONDecodeError:
        # Accept single-quoted arrays as a convenience.
        import ast

        try:
            messages = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raise ValueError(
                "The conversation history must be an array of messages, "
                'for example: ["My first message", "Their response"].'
            )

    if not isinstance(messages, list):
        raise ValueError(
            "The conversation history must be an array of messages."
        )

    cleaned = []

    for message in messages:
        if not isinstance(message, str):
            raise ValueError(
                "Every item in the conversation history must be a text message."
            )

        message = message.strip()

        if message:
            cleaned.append(message)

    if not cleaned:
        raise ValueError("The conversation history cannot be empty.")

    return cleaned


async def send_reply_result(ctx, contact, result):
    """
    Render the generated draft consistently for every contact.
    """
    embed = discord.Embed(
        title=f"💬 {contact.display_name}",
        description=result.get("reply", ""),
    )

    embed.add_field(
        name="Stage",
        value=str(result.get("relationship_stage", "unknown")),
        inline=True,
    )

    vibe = result.get("vibe", {}) or {}

    if isinstance(vibe, dict):
        vibe_value = str(vibe.get("momentum", "unknown"))
    else:
        vibe_value = str(vibe)

    embed.add_field(
        name="Vibe",
        value=vibe_value,
        inline=True,
    )

    boredom = result.get("boredom", {}) or {}

    if isinstance(boredom, dict):
        boredom_score = boredom.get("score", 0)
    else:
        boredom_score = boredom

    embed.add_field(
        name="Boredom",
        value=f"{boredom_score}/100",
        inline=True,
    )

    embed.add_field(
        name="Date readiness",
        value=f"{result.get('date_readiness', 0)}/100",
        inline=True,
    )

    embed.add_field(
        name="Send mode",
        value=str(result.get("send_mode", "text")),
        inline=True,
    )

    embed.add_field(
        name="Next move",
        value=result.get(
            "next_goal",
            "Keep the interaction natural.",
        ),
        inline=False,
    )

    voice_script = result.get("voice_script", "")
    if voice_script:
        embed.add_field(
            name="🎙️ Voice-note draft",
            value=str(voice_script)[:1000],
            inline=False,
        )

    intimacy = result.get("intimacy_guidance", "")
    if intimacy:
        embed.add_field(
            name="❤️ Intimacy guidance",
            value=str(intimacy)[:1000],
            inline=False,
        )

    await ctx.reply(embed=embed)


async def process_contact_message(ctx, contact_name: str, incoming: str):
    """
    Process a message addressed directly to a contact.

    Supported forms:

        !Esther: Hey, how are you?

        !contEsther: Hey, how are you?

    The contact name is resolved dynamically from the database.
    There is no Esther-specific logic here.
    """
    contact_name = normalize_name(contact_name)

    if contact_name.lower().startswith("cont") and len(contact_name) > 4:
        # Explicit alias: !contEsther:
        possible_name = contact_name[4:].strip()

        # Only remove "cont" when it actually resolves to a contact.
        if possible_name:
            with SessionLocal() as db:
                explicit_contact = find_contact(db, possible_name)

            if explicit_contact:
                contact_name = possible_name

    with SessionLocal() as db:
        contact = find_contact(db, contact_name)

        if not contact:
            await ctx.reply(
                f"❌ No contact named **{contact_name}**. "
                f"Create one first with "
                f"`!newconvo {contact_name}: [\"your first message\", \"their response\"]`."
            )
            return

        # Directly addressed messages automatically activate that contact.
        set_active_contact(db, contact)

        incoming_message = save_message(
            db,
            contact.id,
            "incoming",
            incoming,
            "discord",
            str(ctx.message.id),
        )

        history = recent_messages(db, contact.id, 30)
        memories = get_memories(db, contact.id, 50)

        heuristic = boredom_heuristic(history)

        result = generate_reply(
            load_persona(),
            contact,
            history,
            memories,
            incoming,
        )

        result.setdefault("boredom", {})["heuristic"] = heuristic

        result["coaching_note"] = (
            result.get("coaching_note", "")
            + f" Question frequency in recent drafts: "
            f"{question_frequency(history):.0%}."
        ).strip()

        vibe = result.get("vibe", {}) or {}
        boredom = result.get("boredom", {}) or {}

        contact.relationship_stage = result.get(
            "relationship_stage",
            contact.relationship_stage,
        )

        if isinstance(vibe, dict):
            contact.vibe_score = float(
                vibe.get("momentum", contact.vibe_score)
                or contact.vibe_score
            )

            contact.engagement_score = float(
                vibe.get("reciprocity", contact.engagement_score)
                or contact.engagement_score
            )

        contact.date_readiness = float(
            result.get(
                "date_readiness",
                contact.date_readiness,
            )
            or contact.date_readiness
        )

        add_event(
            db,
            contact.id,
            "analysis",
            detail=str(
                boredom.get("reasons", [])
                if isinstance(boredom, dict)
                else []
            ),
            score=float(
                boredom.get("score", 0)
                if isinstance(boredom, dict)
                else 0
            ),
        )

        for item in result.get("memory_updates", [])[:12]:
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()

            if key and value:
                try:
                    importance = int(item.get("importance", 3) or 3)
                except (TypeError, ValueError):
                    importance = 3

                upsert_memory(
                    db,
                    contact.id,
                    key,
                    value,
                    max(1, min(5, importance)),
                    incoming_message.id,
                )

        db.commit()

    await send_reply_result(ctx, contact, result)


@bot.event
async def on_ready():
    print(
        f"Discord adapter online as {bot.user} "
        f"(id={bot.user.id})"
    )


@bot.event
async def on_message(message):
    """
    Custom dispatcher for the human-friendly contact command syntax.

    This runs before discord.py's normal command parser so:

        !newconvo Esther: ["Hi", "Hello"]

    cannot accidentally become:

        contact = "newconvo Esther"

    Contact messages are similarly handled directly:

        !Esther: I miss you

    The same parser works for every contact in the database.
    """
    if message.author.bot:
        return

    content = message.content.strip()

    # ------------------------------------------------------------
    # !newconvo Name: ["first message", "their response", ...]
    # ------------------------------------------------------------
    new_match = re.match(
        r"^!newconvo\s+(.+?)\s*:\s*(.+)$",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if new_match:
        if OWNER_DISCORD_USER_ID and str(message.author.id) != OWNER_DISCORD_USER_ID:
            return

        name = normalize_name(new_match.group(1))
        raw_history = new_match.group(2)

        if not name:
            await message.reply(
                "❌ A contact name is required.\n"
                'Example: `!newconvo Esther: ["Hey", "Hey!"]`'
            )
            return

        try:
            seed_messages = parse_seed_messages(raw_history)
        except ValueError as exc:
            await message.reply(f"❌ {exc}")
            return

        with SessionLocal() as db:
            contact = get_or_create_contact(db, name)
            set_active_contact(db, contact)

            # New conversation history is stored in the exact order
            # supplied by the user:
            #
            # index 0 -> me
            # index 1 -> incoming
            # index 2 -> me
            # index 3 -> incoming
            #
            # This makes the initial conversation fully persistent.
            for index, text in enumerate(seed_messages):
                sender = "me" if index % 2 == 0 else "incoming"
                save_message(
                    db,
                    contact.id,
                    sender,
                    text,
                    "discord",
                )

            add_event(
                db,
                contact.id,
                "conversation_created",
                detail=f"Seeded with {len(seed_messages)} messages.",
            )

            db.commit()

            display_name = contact.display_name

        await message.reply(
            f"🧠 Conversation created and activated for "
            f"**{display_name}**.\n\n"
            f"Stored **{len(seed_messages)} messages** in order.\n"
            f"From now on, send messages directly with:\n"
            f"`!{display_name}: message`"
        )
        return

    # ------------------------------------------------------------
    # !newconvo Name
    #
    # Keep a simple creation form as a fallback, but it does not
    # create fake history. The structured colon form above is the
    # preferred command.
    # ------------------------------------------------------------
    simple_new_match = re.match(
        r"^!newconvo\s+(.+?)\s*$",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if simple_new_match:
        if OWNER_DISCORD_USER_ID and str(message.author.id) != OWNER_DISCORD_USER_ID:
            return

        name = normalize_name(simple_new_match.group(1))

        with SessionLocal() as db:
            contact = get_or_create_contact(db, name)
            set_active_contact(db, contact)
            db.commit()

        await message.reply(
            f"🧠 Conversation created and activated for "
            f"**{contact.display_name}**.\n"
            f"Add initial history with:\n"
            f'`!newconvo {contact.display_name}: ["your message", "their response"]`'
        )
        return

    # ------------------------------------------------------------
    # !deleteconvo Name
    # ------------------------------------------------------------
    delete_match = re.match(
        r"^!deleteconvo\s+(.+?)\s*$",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if delete_match:
        if OWNER_DISCORD_USER_ID and str(message.author.id) != OWNER_DISCORD_USER_ID:
            return

        name = normalize_name(delete_match.group(1))

        with SessionLocal() as db:
            contact = find_contact(db, name)

            if not contact:
                await message.reply(
                    f"❌ No existing conversation for **{name}**."
                )
                return

            display = contact.display_name
            delete_contact(db, contact.id)
            db.commit()

        await message.reply(
            f"🗑️ Permanently deleted **{display}** "
            f"and its stored conversation history and memory."
        )
        return

    # ------------------------------------------------------------
    # !contName: message
    #
    # Explicit contact-addressing alias.
    # ------------------------------------------------------------
    cont_match = re.match(
        r"^!cont(.+?)\s*:\s*(.+)$",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if cont_match:
        if OWNER_DISCORD_USER_ID and str(message.author.id) != OWNER_DISCORD_USER_ID:
            return

        await process_contact_message(
            message,
            cont_match.group(1),
            cont_match.group(2).strip(),
        )
        return

    # ------------------------------------------------------------
    # !Name: message
    #
    # Generic contact-addressed message.
    #
    # This intentionally does NOT contain any hard-coded contact
    # names. Every contact is resolved dynamically.
    # ------------------------------------------------------------
    contact_match = re.match(
        r"^!(.+?)\s*:\s*(.+)$",
        content,
        flags=re.DOTALL,
    )

    if contact_match:
        if OWNER_DISCORD_USER_ID and str(message.author.id) != OWNER_DISCORD_USER_ID:
            return

        contact_name = normalize_name(contact_match.group(1))

        # Don't let reserved commands fall through as contact names.
        if contact_name.casefold() in {
            "newconvo",
            "deleteconvo",
            "active",
            "helpme",
            "reply",
            "contconvo",
        }:
            return

        await process_contact_message(
            message,
            contact_name,
            contact_match.group(2).strip(),
        )
        return

    # Normal discord.py commands still work for administrative/help
    # commands that remain registered below.
    await bot.process_commands(message)


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


@bot.command(name="helpme")
@owner_only()
async def helpme(ctx):
    await ctx.reply(
        "**Relationship Copilot**\n\n"
        "**Create a conversation**\n"
        '`!newconvo Esther: ["Hey Esther", "Hey! How are you?"]`\n\n'
        "**Message any contact**\n"
        "`!Esther: I was thinking about you`\n"
        "`!Sarah: How was your day?`\n"
        "`!Jessica: Are you free tonight?`\n\n"
        "**Explicit contact form**\n"
        "`!contEsther: I was thinking about you`\n\n"
        "**Other commands**\n"
        "`!active` — show active conversation\n"
        "`!deleteconvo Esther` — permanently delete Esther\n"
    )


def run():
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing")

    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    run()
