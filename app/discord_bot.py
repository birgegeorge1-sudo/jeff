import json
import os
import re
from typing import Any, Optional

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


DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
PREFIX = os.getenv("DISCORD_PREFIX", "!")
OWNER_DISCORD_USER_ID = os.getenv("OWNER_DISCORD_USER_ID", "").strip()


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents,
)


# These are real Discord commands and MUST NEVER be interpreted
# as dynamic contact names by the !Name: message parser.
KNOWN_COMMANDS = {
    "newconvo",
    "deleteconvo",
    "active",
    "helpme",
    "reply",
}


def owner_only():
    async def predicate(ctx: commands.Context) -> bool:
        if not OWNER_DISCORD_USER_ID:
            return True

        return str(ctx.author.id) == OWNER_DISCORD_USER_ID

    return commands.check(predicate)


def _is_registered_command(content: str) -> bool:
    """
    Detect a normal Discord command before attempting to parse
    dynamic contact-addressed syntax.

    Examples:

        !newconvo Esther: ["Hi", "Hey"]
        !deleteconvo Esther
        !active
        !helpme
        !reply hello

    This is the critical fix for the previous bug where:

        !newconvo Esther: [...]

    was interpreted as contact name "newconvo Esther".
    """
    if not content:
        return False

    prefix = PREFIX or "!"

    if not content.startswith(prefix):
        return False

    remainder = content[len(prefix):].lstrip()

    if not remainder:
        return False

    # Take the first whitespace-delimited token.
    # Also strip ':' so forms such as !newconvo: ... are recognized.
    command_token = remainder.split(None, 1)[0]
    command_token = command_token.split(":", 1)[0].casefold()

    return command_token in KNOWN_COMMANDS


def _parse_contact_message(content: str):
    """
    Parse dynamic contact-addressed messages.

    Supported:

        !Esther: hello
        !Sarah: what are you doing?
        !contEsther: hey

    The contact name is looked up dynamically in the database.

    Returns:

        (name, message, explicit_cont_prefix)

    or None when the content is not contact-addressed.
    """
    if not content:
        return None

    match = re.match(
        r"^!(?P<prefix>cont)?(?P<name>[^:\n]{1,255}):\s*(?P<message>.+)$",
        content,
        re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None

    prefix = (match.group("prefix") or "").casefold()

    name = " ".join(
        match.group("name").strip().split()
    )

    message = match.group("message").strip()

    if not name or not message:
        return None

    # Never allow a registered command to fall through to the
    # dynamic-contact parser.
    if name.casefold() in KNOWN_COMMANDS:
        return None

    return name, message, bool(prefix)


def _normalize_history_sender(
    sender: Any,
    default_sender: str,
) -> str:
    """
    Normalize the sender value supplied in seeded history.
    """
    normalized = str(
        sender if sender is not None else default_sender
    ).strip().lower()

    if normalized in {
        "her",
        "them",
        "their",
        "contact",
        "incoming",
        "received",
        "other",
    }:
        return "incoming"

    if normalized in {
        "me",
        "i",
        "myself",
        "owner",
        "jeff",
        "outgoing",
        "sent",
    }:
        return "me"

    raise ValueError(
        "Each history object must use sender/role "
        "'me' or 'incoming'."
    )


def _parse_newconvo_payload(payload: str):
    """
    Parse either:

        !newconvo Esther

    or:

        !newconvo Esther: ["Hey", "Hey!", "How are you?"]

    Also supports explicit sender objects:

        !newconvo Esther: [
            {"sender": "me", "content": "Hey"},
            {"sender": "incoming", "content": "Hey!"}
        ]

    Positional string arrays alternate automatically:

        item 1 -> me
        item 2 -> incoming
        item 3 -> me
        item 4 -> incoming

    Returns:

        (name, normalized_history)

    where normalized_history is:

        [
            ("me", "..."),
            ("incoming", "..."),
        ]

    When no history is supplied, normalized_history is None.
    """
    payload = (payload or "").strip()

    if not payload:
        return "", None

    if ":" not in payload:
        return " ".join(payload.split()), None

    name, raw_history = payload.split(":", 1)

    name = " ".join(
        name.strip().split()
    )

    raw_history = raw_history.strip()

    if not raw_history:
        return name, None

    try:
        parsed = json.loads(raw_history)
    except json.JSONDecodeError as exc:
        raise ValueError(
            'History after the colon must be valid JSON, '
            'for example: ["Hi", "Hey!"]'
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            "Conversation history must be a JSON array."
        )

    if not parsed:
        raise ValueError(
            "Conversation history must be a non-empty JSON array."
        )

    normalized = []

    for index, item in enumerate(parsed):
        default_sender = (
            "me"
            if index % 2 == 0
            else "incoming"
        )

        sender = default_sender
        content = item

        if isinstance(item, dict):
            sender = _normalize_history_sender(
                item.get(
                    "sender",
                    item.get(
                        "role",
                        default_sender,
                    ),
                ),
                default_sender,
            )

            content = item.get(
                "message",
                item.get(
                    "content",
                    "",
                ),
            )

        elif isinstance(item, str):
            sender = default_sender

        else:
            raise ValueError(
                f"History item {index + 1} must be either "
                "a string or an object."
            )

        content = str(content).strip()

        if not content:
            raise ValueError(
                f"History item {index + 1} is empty."
            )

        normalized.append(
            (
                sender,
                content,
            )
        )

    return name, normalized


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """
    Convert arbitrary values to float without crashing analysis
    when the model returns null, strings, or malformed values.
    """
    if value is None:
        return fallback

    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int = 3) -> int:
    """
    Convert arbitrary values to int safely.
    """
    if value is None:
        return fallback

    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


async def _send_analysis(
    source,
    contact,
    incoming: str,
):
    """
    Generate and render analysis for an incoming contact message.

    `source` can be:
        - discord.Message
        - commands.Context

    Both expose .reply().
    """
    contact_display_name: Optional[str] = None
    result = None

    with SessionLocal() as db:
        # Always re-resolve using the current session.
        #
        # Avoid the old positional fallback call because the database
        # helper should be accessed consistently by keyword.
        resolved_contact = find_contact(
            db,
            name=contact.display_name,
        )

        if not resolved_contact:
            await source.reply(
                "❌ That contact does not exist yet. "
                'Create them with `!newconvo Name: '
                '["your first message", "their response"]`.'
            )
            return

        contact = resolved_contact
        contact_display_name = contact.display_name

        set_active_contact(
            db,
            contact,
            commit=False,
        )

        external_message_id = getattr(
            source,
            "id",
            None,
        )

        incoming_message = save_message(
            db,
            contact.id,
            "incoming",
            incoming,
            platform="discord",
            external_message_id=(
                str(external_message_id)
                if external_message_id is not None
                else None
            ),
        )

        result = generate_reply(
            contact,
            recent_messages(
                db,
                contact.id,
                36,
            ),
            get_memories(
                db,
                contact.id,
                50,
            ),
            incoming,
        )

        if not isinstance(result, dict):
            result = {}

        # ---------------------------------------------------------
        # Persist model-generated memory updates
        # ---------------------------------------------------------
        memory_updates = result.get(
            "memory_updates",
            [],
        )

        if not isinstance(memory_updates, list):
            memory_updates = []

        for item in memory_updates[:12]:
            if not isinstance(item, dict):
                continue

            key = str(
                item.get(
                    "key",
                    "",
                )
            ).strip()

            value = str(
                item.get(
                    "value",
                    "",
                )
            ).strip()

            if not key or not value:
                continue

            importance = _safe_int(
                item.get(
                    "importance",
                    3,
                ),
                fallback=3,
            )

            importance = max(
                1,
                min(
                    5,
                    importance,
                ),
            )

            upsert_memory(
                db,
                contact.id,
                key,
                value,
                importance,
                incoming_message.id,
            )

        # ---------------------------------------------------------
        # Normalize result fields
        # ---------------------------------------------------------
        vibe = result.get(
            "vibe",
            {},
        )

        if not isinstance(vibe, dict):
            vibe = {}

        boredom = result.get(
            "boredom",
            {},
        )

        if not isinstance(boredom, dict):
            boredom = {}

        relationship_stage = result.get(
            "relationship_stage",
            contact.relationship_stage,
        )

        if relationship_stage:
            contact.relationship_stage = str(
                relationship_stage
            ).strip()

        contact.vibe_score = _safe_float(
            vibe.get(
                "momentum",
                contact.vibe_score,
            ),
            fallback=_safe_float(
                contact.vibe_score,
                0.0,
            ),
        )

        contact.engagement_score = _safe_float(
            vibe.get(
                "reciprocity",
                contact.engagement_score,
            ),
            fallback=_safe_float(
                contact.engagement_score,
                0.0,
            ),
        )

        contact.date_readiness = _safe_float(
            result.get(
                "date_readiness",
                contact.date_readiness,
            ),
            fallback=_safe_float(
                contact.date_readiness,
                0.0,
            ),
        )

        boredom_score = _safe_float(
            boredom.get(
                "score",
                result.get(
                    "boredom_score",
                    0,
                ),
            ),
            fallback=0.0,
        )

        boredom_reasons = boredom.get(
            "reasons",
            [],
        )

        if isinstance(boredom_reasons, list):
            boredom_detail = str(
                boredom_reasons
            )
        else:
            boredom_detail = str(
                boredom_reasons
            )

        add_event(
            db,
            contact.id,
            "analysis",
            detail=boredom_detail,
            score=boredom_score,
        )

        db.commit()

    # Do not rely on a SQLAlchemy ORM object after its session closes.
    # Pass a stable display name into the renderer instead.
    await _render_analysis(
        source,
        contact_display_name or "Contact",
        result or {},
    )


async def _render_analysis(
    source,
    contact_display_name: str,
    result: dict,
):
    """
    Render generated relationship analysis.
    """
    def _field_value(raw: Any, default: str = "—") -> str:
        """
        Guarantee a non-empty, safe embed field value.

        Discord rejects embeds where any field value is an empty string
        (error 50035 'Invalid Form Body'). This helper coerces any
        input to a non-empty string under 1024 chars.
        """
        s = str(raw).strip() if raw is not None else ""
        if not s:
            s = default
        if len(s) > 1024:
            s = s[:1021] + "…"
        return s

    def _desc_value(raw: Any, default: str = "No reply draft was generated.") -> str:
        """
        Same guard for the embed description (max 4096).
        """
        s = str(raw).strip() if raw is not None else ""
        if not s:
            s = default
        if len(s) > 4096:
            s = s[:4093] + "…"
        return s

    reply_text = str(
        result.get(
            "reply",
            "",
        )
        or ""
    )

    embed = discord.Embed(
        title=f"🤔 {contact_display_name}",
        description=_desc_value(reply_text),
    )

    embed.add_field(
        name="Stage",
        value=_field_value(
            result.get(
                "relationship_stage",
                "unknown",
            )
        ),
        inline=True,
    )

    embed.add_field(
        name="Vibe",
        value=_field_value(
            result.get(
                "vibe",
                "unknown",
            )
        ),
        inline=True,
    )

    embed.add_field(
        name="Boredom",
        value=f"{result.get('boredom_score', 0)}/100",
        inline=True,
    )

    embed.add_field(
        name="Date readiness",
        value=f"{result.get('date_readiness', 0)}/100",
        inline=True,
    )

    embed.add_field(
        name="Send mode",
        value=_field_value(
            result.get(
                "send_mode",
                "text",
            )
        ),
        inline=True,
    )

    embed.add_field(
        name="Next move",
        value=_field_value(
            result.get(
                "next_goal",
                "Keep the interaction natural.",
            )
        ),
        inline=False,
    )

    alternatives = result.get(
        "alternatives",
        [],
    )

    if isinstance(alternatives, list) and alternatives:
        alternative_text = "\n\n".join(
            str(item)
            for item in alternatives[:2]
        ).strip()

        if alternative_text:
            embed.add_field(
                name="Alternative drafts",
                value=alternative_text[:1900] or "—",
                inline=False,
            )

    voice_script = result.get(
        "voice_script",
        "",
    )

    if voice_script:
        vs = str(voice_script).strip()
        if vs:
            embed.add_field(
                name="📝 Voice-note draft",
                value=vs[:1000],
                inline=False,
            )

    intimacy_guidance = result.get(
        "intimacy_guidance",
        "",
    )

    if intimacy_guidance:
        ig = str(intimacy_guidance).strip()
        if ig:
            embed.add_field(
                name="💬 Intimacy guidance",
                value=ig[:1000],
                inline=False,
            )

    await source.reply(
        embed=embed,
    )


@bot.event
async def on_ready():
    print(
        f"Discord adapter online as {bot.user} "
        f"(id={bot.user.id})"
    )


@bot.event
async def on_message(message: discord.Message):
    """
    Main dispatcher.

    IMPORTANT ORDER:

    1. Ignore bots / unauthorized users.
    2. Detect registered commands FIRST.
    3. Process normal Discord commands.
    4. Only then attempt dynamic contact syntax.

    This means:

        !newconvo Esther: ["Hi", "Hey!"]

    can NEVER become:

        contact = "newconvo Esther"
    """
    if message.author.bot:
        return

    if (
        OWNER_DISCORD_USER_ID
        and str(message.author.id) != OWNER_DISCORD_USER_ID
    ):
        return

    content = message.content.strip()

    if not content:
        return

    # ---------------------------------------------------------
    # CRITICAL FIX:
    # Registered commands ALWAYS get first priority.
    # ---------------------------------------------------------
    if _is_registered_command(content):
        await bot.process_commands(message)
        return

    # ---------------------------------------------------------
    # Dynamic contact-addressed syntax
    #
    # !Esther: hello
    # !contEsther: hello
    # !Sarah: what are you doing?
    # ---------------------------------------------------------
    parsed = _parse_contact_message(
        content
    )

    if parsed:
        name, incoming, _explicit_cont = parsed

        with SessionLocal() as db:
            contact = find_contact(
                db,
                name=name,
            )

            if not contact:
                await message.channel.send(
                    f"❌ No contact named **{name}**. "
                    f'Create one first with `!newconvo {name}: '
                    f'["your first message", "their response"]`.'
                )
                return

        await _send_analysis(
            message,
            contact,
            incoming,
        )
        return

    # Anything else goes through regular Discord command handling.
    await bot.process_commands(message)


@bot.command(name="newconvo")
@owner_only()
async def newconvo(
    ctx: commands.Context,
    *,
    payload: str,
):
    """
    Create a contact and optionally seed their complete
    conversation history.

    Examples:

        !newconvo Esther

        !newconvo Esther: ["Hey Esther", "Hey!", "How are you?"]

    Explicit sender objects are also supported:

        !newconvo Esther: [
            {"sender": "me", "content": "Hey Esther"},
            {"sender": "incoming", "content": "Hey!"}
        ]
    """
    try:
        name, history = _parse_newconvo_payload(
            payload
        )
    except ValueError as exc:
        await ctx.reply(
            f"❌ {exc}"
        )
        return

    if not name:
        await ctx.reply(
            "❌ A contact name is required."
        )
        return

    # Prevent accidental creation of a contact whose name is
    # actually a reserved command.
    if name.casefold() in KNOWN_COMMANDS:
        await ctx.reply(
            f"❌ **{name}** is reserved as a bot command."
        )
        return

    with SessionLocal() as db:
        existing = find_contact(
            db,
            name=name,
        )

        if existing:
            await ctx.reply(
                f"❌ **{existing.display_name}** already exists. "
                f"Use `!{existing.display_name}: message` "
                f"to continue that conversation."
            )
            return

        contact = get_or_create_contact(
            db,
            name=name,
        )

        set_active_contact(
            db,
            contact,
            commit=False,
        )

        if history:
            for sender, content in history:
                save_message(
                    db,
                    contact.id,
                    sender,
                    content,
                    platform="discord",
                )

        add_event(
            db,
            contact.id,
            "conversation_created",
            detail=(
                "seeded_messages="
                f"{len(history or [])}"
            ),
        )

        db.commit()

        # Capture before leaving the session.
        display_name = contact.display_name

    seeded = len(history or [])

    await ctx.reply(
        f"✅ Created and activated "
        f"**{display_name}**.\n"
        f"Persisted `{seeded}` message"
        f"{'s' if seeded != 1 else ''}.\n"
        f"Use `!{display_name}: their next message` "
        f"whenever they message you."
    )


@bot.command(name="deleteconvo")
@owner_only()
async def deleteconvo(
    ctx: commands.Context,
    *,
    name: str,
):
    name = " ".join(
        name.strip().split()
    )

    if not name:
        await ctx.reply(
            "❌ A contact name is required."
        )
        return

    with SessionLocal() as db:
        contact = find_contact(
            db,
            name=name,
        )

        if not contact:
            await ctx.reply(
                f"❌ No existing conversation for **{name}**."
            )
            return

        display = contact.display_name

        delete_contact(
            db,
            contact.id,
        )

        db.commit()

    await ctx.reply(
        f"🗑️ Permanently deleted **{display}** "
        f"and its stored conversation memory/history."
    )


@bot.command(name="active")
@owner_only()
async def active(ctx: commands.Context):
    with SessionLocal() as db:
        contact = get_active_contact(db)

        if not contact:
            await ctx.reply(
                "No active conversation."
            )
            return

        display_name = contact.display_name
        relationship_stage = contact.relationship_stage

    await ctx.reply(
        f"🎯 Active: **{display_name}**\n"
        f"Stage: `{relationship_stage}`"
    )


@bot.command(name="reply")
@owner_only()
async def reply(
    ctx: commands.Context,
    *,
    incoming: str,
):
    """
    Backward-compatible manual command.

    Preferred syntax:

        !Esther: their message
    """
    with SessionLocal() as db:
        contact = get_active_contact(db)

        if not contact:
            await ctx.reply(
                "❌ No active contact. "
                "Use `!Name: their message` instead."
            )
            return

        # Capture the name while the DB session is alive.
        contact_display_name = contact.display_name

    # Re-resolve inside _send_analysis using the current session.
    class ContactRef:
        def __init__(self, display_name: str):
            self.display_name = display_name

    await _send_analysis(
        ctx,
        ContactRef(contact_display_name),
        incoming,
    )


@bot.command(name="helpme")
@owner_only()
async def helpme(ctx: commands.Context):
    await ctx.reply(
        "**Relationship Copilot commands**\n\n"
        "`!newconvo Esther: [\"Hi Esther\", \"Hey!\"]` "
        "— create + persist conversation history\n"
        "`!Esther: her next message` "
        "— address any existing contact directly\n"
        "`!contEsther: her next message` "
        "— explicit contact form\n"
        "`!deleteconvo Esther` "
        "— permanently delete contact + history\n"
        "`!active` "
        "— show the most recently addressed contact\n"
        "`!reply their message` "
        "— analyze against the active contact\n\n"
        "**Any existing contact name works.**\n"
        "**Commands always take priority over contact syntax.**"
    )


async def _handle_command_error(
    ctx: commands.Context,
    error: commands.CommandError,
):
    """
    Keep command failures human-readable instead of dumping
    Discord.py exception text into the channel.
    """
    if isinstance(error, commands.CheckFailure):
        return

    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply(
            "❌ Missing required argument. Use `!helpme` "
            "to see the command format."
        )
        return

    if isinstance(error, commands.CommandNotFound):
        return

    # Truncate the error string so the reply never exceeds
    # Discord's 2000-character limit (prevents 50035 'Invalid Form Body').
    _err_msg = str(error)

    if len(_err_msg) > 500:
        _err_msg = _err_msg[:497] + "…"

    await ctx.reply(
        f"❌ Command error: {_err_msg}"
    )

    print(
        f"Discord command error in "
        f"{getattr(ctx.command, 'name', 'unknown')}: {error}"
    )


bot.add_listener(
    _handle_command_error,
    "on_command_error",
)


def run():
    import time as _time

    if not DISCORD_BOT_TOKEN:
        raise RuntimeError(
            "DISCORD_BOT_TOKEN is missing"
        )

    # Backed-off login loop.
    #
    # When the container's IP is rate-limited (HTTP 429 / Cloudflare
    # 1015), a crash makes Render restart the bot immediately, which
    # re-triggers the ban. Instead, wait with exponential backoff and
    # retry a bounded number of times.
    max_login_attempts = 6
    base_delay = 15.0
    max_delay = 300.0

    attempt = 0

    while True:
        attempt += 1

        try:
            bot.run(
                DISCORD_BOT_TOKEN,
                reconnect=True,
            )

            break

        except discord.LoginFailure as exc:
            print(
                f"[DISCORD] Login failed "
                f"(attempt {attempt}/{max_login_attempts}): {exc}"
            )

            if attempt >= max_login_attempts:
                raise

            delay = min(
                max_delay,
                base_delay * (2 ** (attempt - 1)),
            )

            print(
                f"[DISCORD] Waiting {delay:.0f}s before re-login."
            )

            _time.sleep(delay)

        except discord.HTTPException as exc:
            code = getattr(exc, "code", None)

            print(
                f"[DISCORD] HTTP {code} during start; "
                f"retrying with backoff."
            )

            if attempt >= max_login_attempts:
                raise

            delay = min(
                max_delay,
                base_delay * (2 ** (attempt - 1)),
            )

            _time.sleep(delay)


if __name__ == "__main__":
    run()
