import json
import os
import random
import time
from typing import Any, Optional

from mistralai.client import Mistral


# ============================================================
# Configuration
# ============================================================

MODEL = os.getenv(
    "MODEL",
    "mistral-large-latest",
)

PERSONA_PATH = os.getenv(
    "PERSONA_PATH",
    os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)
        ),
        "persona.json",
    ),
)


# ============================================================
# Persona
# ============================================================

with open(
    PERSONA_PATH,
    "r",
    encoding="utf-8",
) as f:
    PERSONA = json.load(f)


# ============================================================
# Mistral API key pool
#
# Supported environment variables:
#
#   MISTRAL_API_KEY
#   MISTRAL_API_KEY_1
#   MISTRAL_API_KEY_2
#   MISTRAL_API_KEY_3
#   ...
#
# The primary MISTRAL_API_KEY is used first.
# Additional numbered keys are rotated automatically when
# a key is rate-limited.
# ============================================================

def _load_mistral_keys() -> list[str]:
    keys: list[str] = []

    primary = os.getenv(
        "MISTRAL_API_KEY",
        "",
    ).strip()

    if primary:
        keys.append(primary)

    index = 1

    while True:
        key = os.getenv(
            f"MISTRAL_API_KEY_{index}",
            "",
        ).strip()

        if not key:
            break

        if key not in keys:
            keys.append(key)

        index += 1

    return keys


MISTRAL_API_KEYS = _load_mistral_keys()

if not MISTRAL_API_KEYS:
    raise RuntimeError(
        "No Mistral API keys configured. "
        "Set MISTRAL_API_KEY and optionally "
        "MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, etc."
    )


MISTRAL_CLIENTS = [
    Mistral(api_key=key)
    for key in MISTRAL_API_KEYS
]


# Shared rotation position.
#
# This is intentionally module-level so requests naturally
# move through the configured key pool.
_current_key_index = 0


def _current_client():
    return MISTRAL_CLIENTS[
        _current_key_index
    ]


def _rotate_key() -> None:
    """
    Move to the next configured Mistral API key.

    With one key this is a no-op.
    """
    global _current_key_index

    if len(MISTRAL_CLIENTS) <= 1:
        return

    _current_key_index = (
        _current_key_index + 1
    ) % len(MISTRAL_CLIENTS)


def _extract_status_code(exc: Exception) -> Optional[int]:
    """
    Try several locations used by different versions of the
    Mistral SDK / HTTP stack to expose the HTTP status code.
    """
    status_code = getattr(
        exc,
        "status_code",
        None,
    )

    if status_code is not None:
        try:
            return int(status_code)
        except (TypeError, ValueError):
            pass

    http_response = getattr(
        exc,
        "http_res",
        None,
    )

    if http_response is not None:
        status_code = getattr(
            http_response,
            "status_code",
            None,
        )

        if status_code is not None:
            try:
                return int(status_code)
            except (TypeError, ValueError):
                pass

    response = getattr(
        exc,
        "response",
        None,
    )

    if response is not None:
        status_code = getattr(
            response,
            "status_code",
            None,
        )

        if status_code is not None:
            try:
                return int(status_code)
            except (TypeError, ValueError):
                pass

    return None


def _is_rate_limited(exc: Exception) -> bool:
    """
    Detect Mistral HTTP 429 / rate-limited responses without
    relying on one exact SDK exception class.
    """
    status_code = _extract_status_code(exc)

    if status_code == 429:
        return True

    text = str(exc).lower()

    rate_limit_markers = (
        "status 429",
        "rate limit",
        "rate_limited",
        "too many requests",
        "code\": \"1300",
        "code': '1300",
        "code=1300",
        '"code":1300',
    )

    return any(
        marker in text
        for marker in rate_limit_markers
    )


def _retry_delay(attempt: int) -> float:
    """
    Exponential backoff with jitter.

    Example approximate delays:
        attempt 0 -> 1 sec
        attempt 1 -> 2 sec
        attempt 2 -> 4 sec
        attempt 3 -> 8 sec
        attempt 4 -> 16 sec
        capped at 30 sec

    Jitter prevents several simultaneous requests from
    retrying at exactly the same moment.
    """
    base = min(
        30.0,
        2.0 ** attempt,
    )

    jitter = random.uniform(
        0.0,
        0.75,
    )

    return base + jitter


def _mistral_complete(
    *,
    messages: list[dict[str, Any]],
    temperature: float = 0.55,
    response_format: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
):
    """
    Centralized Mistral request handler.

    Features:
      - multiple API-key rotation
      - automatic 429 detection
      - exponential backoff
      - jitter
      - bounded retries
      - preservation of non-429 exceptions

    A 429 rotates to another key before retrying.
    """

    # Two passes through the pool at most, while guaranteeing
    # a minimum of three attempts.
    max_attempts = max(
        3,
        len(MISTRAL_CLIENTS) * 2,
    )

    last_exception: Optional[Exception] = None

    for attempt in range(max_attempts):
        client = _current_client()
        key_number = _current_key_index + 1

        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format is not None:
            kwargs["response_format"] = response_format

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            return client.chat.complete(
                **kwargs
            )

        except Exception as exc:
            last_exception = exc

            if not _is_rate_limited(exc):
                # Do not retry unrelated API errors indefinitely.
                raise

            print(
                "[Mistral] Rate limited "
                f"on key {key_number}/"
                f"{len(MISTRAL_CLIENTS)} "
                f"(attempt {attempt + 1}/"
                f"{max_attempts})."
            )

            # Move to another key immediately when available.
            _rotate_key()

            # If another attempt remains, back off before retrying.
            if attempt < max_attempts - 1:
                delay = _retry_delay(
                    attempt
                )

                print(
                    f"[Mistral] Retrying in "
                    f"{delay:.2f}s with key "
                    f"{_current_key_index + 1}/"
                    f"{len(MISTRAL_CLIENTS)}."
                )

                time.sleep(delay)

    raise RuntimeError(
        "Mistral API remained rate-limited after "
        f"{max_attempts} attempts across "
        f"{len(MISTRAL_CLIENTS)} configured key(s)."
    ) from last_exception


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = r"""
You are the private writing and conversation coach for Jefferey Campbell.

Your output is a DRAFT for Jefferey to review and send from his own account.

Write the draft as something Jefferey himself would naturally say—not as a software agent,
virtual assistant, chatbot, narrator, or corporate representative.

Do not add AI/bot language to ordinary drafts.

If the recipient directly asks whether a human is writing the messages,
flag the situation in `safety_note` and do not manufacture an answer;
Jefferey should answer that question truthfully himself.

Never invent concrete facts about Jefferey's biography.
Use PERSONA data as the source of truth.

Core operating principles:

1. Never invent concrete facts about Jefferey's biography.
   Use PERSONA data as the source of truth.

2. The other person is an independent adult with her own choices.
   Never manipulate, pressure, guilt, negg, deceive, love-bomb,
   or engineer dependence.

3. Optimize for reciprocal connection, not message volume.

4. Conversation should alternate naturally between reacting,
   sharing, playful tension, curiosity, storytelling and depth.
   Do not run an interview.

5. Ask questions because something in her message makes you
   genuinely curious. Avoid generic questions like
   "what do you do?" or "where are you from?" when the answer
   is already known.

6. Detect boredom from short replies, falling emotional energy,
   repeated topics, low reciprocity, delayed/closed responses,
   forced questioning, or absence of new material.

   Treat boredom as a signal to change pace/topic or give space,
   not as a reason to push harder.

7. Detect positive signals such as voluntary detail,
   follow-up questions, playful teasing, vulnerability,
   callbacks, initiation, and practical openness to meeting.

8. Never infer romantic consent merely from friendliness.

   For dates and intimacy, look for reciprocal interest and
   explicit or clearly expressed willingness.

9. Sexuality:

   Mature romantic conversations can naturally become sensual
   or sexual when both adults are clearly comfortable.

   Do not force escalation.

   When intimacy becomes relevant, assess:
   mutuality,
   comfort,
   privacy,
   boundaries,
   and whether the other person is actively participating.

   Prefer a respectful check-in over guessing.

   Never pressure for sexual images.

10. If she becomes uncomfortable, backs away, says no,
    asks to slow down, or stops reciprocating,
    advise de-escalation or space.

11. If a date is appropriate, suggest one clear,
    low-pressure invitation rather than repeated asks.

12. Voice notes should be suggested only when they add warmth,
    nuance, humor, reassurance or emotional presence.

    Keep them speakable and natural.

Use the supplied persona to match the man's voice.

Tennessee influence means warmth and cadence,
not fake dialect spelling or stereotypes.

Return ONLY JSON with this schema:

{
  "reply": "primary draft",
  "alternatives": ["alternative 1", "alternative 2"],
  "send_mode": "text" | "voice",
  "voice_script": "",
  "voice_reason": "",
  "vibe": "flat" | "warming" | "playful" | "deep" | "romantic" | "sexual",
  "boredom_score": 0,
  "reciprocity_score": 0,
  "date_readiness": 0,
  "relationship_stage": "new" | "warming" | "interested" | "flirting" | "date_ready" | "cooling_off",
  "conversation_mode": "react" | "share" | "curiosity" | "play" | "deepen" | "lighten" | "space",
  "next_goal": "",
  "question_strategy": "",
  "memory_updates": [
    {
      "key": "",
      "value": "",
      "importance": 3
    }
  ],
  "positive_signals": [],
  "concerns": [],
  "intimacy_guidance": "",
  "safety_note": ""
}

Do not include markdown fences.
"""


# ============================================================
# Utility functions
# ============================================================

def _clean_content(content: Any) -> str:
    """
    Normalize different Mistral response-content shapes into
    plain text suitable for json.loads().
    """
    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, dict):
                parts.append(
                    str(
                        part.get(
                            "text",
                            "",
                        )
                    )
                )
            else:
                parts.append(
                    str(part)
                )

        return "".join(parts)

    if content is None:
        return ""

    return str(content)


def _clamp_score(
    value: Any,
    default: int = 0,
) -> int:
    """
    Safely normalize model score values to 0-100.
    """
    try:
        value = int(float(value))
    except (
        TypeError,
        ValueError,
    ):
        value = default

    return max(
        0,
        min(
            100,
            value,
        ),
    )


# ============================================================
# Deterministic conversation signals
# ============================================================

def deterministic_signals(
    incoming,
    recent,
):
    text = incoming.strip()
    words = text.split()

    if not words:
        return 100, 0, "flat"

    very_short = len(words) <= 3
    question_marks = incoming.count("?")
    recent_len = len(recent)

    boredom = 0

    if very_short:
        boredom += 25

    if len(text) < 12:
        boredom += 15

    if question_marks == 0:
        boredom += 5

    if recent_len >= 6:
        last = recent[-6:]

        me = sum(
            1
            for x in last
            if x.sender == "me"
        )

        incoming_count = sum(
            1
            for x in last
            if x.sender == "incoming"
        )

        if me > incoming_count:
            boredom += 15

    boredom = min(
        boredom,
        100,
    )

    reciprocity = 40

    if question_marks:
        reciprocity += 15

    if len(words) >= 20:
        reciprocity += 15

    lowered = text.lower()

    if any(
        marker in lowered
        for marker in (
            "haha",
            "lol",
            "😂",
            "😊",
            "😄",
        )
    ):
        reciprocity += 10

    reciprocity = min(
        reciprocity,
        100,
    )

    vibe = (
        "flat"
        if boredom >= 55
        else (
            "warming"
            if reciprocity < 60
            else "playful"
        )
    )

    return (
        boredom,
        reciprocity,
        vibe,
    )


# ============================================================
# Context construction
# ============================================================

def build_context(
    contact,
    messages,
    memories,
    incoming,
):
    mem_text = "\n".join(
        f"- {m.key}: {m.value}"
        for m in memories
    ) or "- none"

    history = []

    for msg in messages[-36:]:
        sender = (
            "HER"
            if msg.sender == "incoming"
            else "ME"
        )

        history.append(
            f"{sender}: {msg.content}"
        )

    return f"""
PERSONA:
{json.dumps(PERSONA, ensure_ascii=False)}

ACTIVE CONTACT:
name: {contact.display_name}
relationship_stage: {contact.relationship_stage}

MEMORY FOR THIS CONTACT ONLY:
{mem_text}

RECENT HISTORY FOR THIS CONTACT ONLY:
{chr(10).join(history) or "(none)"}

NEW MESSAGE:
{incoming}
"""


# ============================================================
# Main generation function
# ============================================================

def generate_reply(
    contact,
    messages,
    memories,
    incoming,
):
    """
    Generate the relationship-coaching response.

    Mistral requests go through _mistral_complete(), which provides:
      - multiple API keys
      - automatic rotation
      - 429 handling
      - retry/backoff
    """

    boredom, reciprocity, fallback_vibe = deterministic_signals(
        incoming,
        messages,
    )

    context = build_context(
        contact,
        messages,
        memories,
        incoming,
    )

    response = _mistral_complete(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": context,
            },
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0.55,
    )

    # --------------------------------------------------------
    # Extract response safely
    # --------------------------------------------------------

    try:
        raw_content = response.choices[
            0
        ].message.content
    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            "Mistral returned an unexpected response structure."
        ) from exc

    cleaned = _clean_content(
        raw_content
    ).strip()

    if not cleaned:
        raise RuntimeError(
            "Mistral returned an empty response."
        )

    try:
        data = json.loads(
            cleaned
        )
    except json.JSONDecodeError as exc:
        # Keep a useful diagnostic while avoiding an enormous
        # response dump in logs.
        preview = cleaned[:1000]

        raise RuntimeError(
            "Mistral returned invalid JSON. "
            f"Response preview: {preview}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Mistral returned JSON that was not an object."
        )

    # --------------------------------------------------------
    # Normalize required fields
    # --------------------------------------------------------

    reply_text = data.get(
        "reply",
        "",
    )

    if not isinstance(
        reply_text,
        str,
    ):
        reply_text = str(
            reply_text
        )

    data["reply"] = reply_text.strip()

    alternatives = data.get(
        "alternatives",
        [],
    )

    if not isinstance(
        alternatives,
        list,
    ):
        alternatives = []

    data["alternatives"] = [
        str(item).strip()
        for item in alternatives[:3]
        if str(item).strip()
    ]

    memory_updates = data.get(
        "memory_updates",
        [],
    )

    if not isinstance(
        memory_updates,
        list,
    ):
        memory_updates = []

    normalized_memory_updates = []

    for item in memory_updates[:12]:
        if not isinstance(
            item,
            dict,
        ):
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

        importance = _clamp_score(
            item.get(
                "importance",
                3,
            ),
            default=3,
        )

        # Memory importance is 1-5, not 0-100.
        importance = max(
            1,
            min(
                5,
                importance,
            ),
        )

        normalized_memory_updates.append(
            {
                "key": key,
                "value": value,
                "importance": importance,
            }
        )

    data["memory_updates"] = normalized_memory_updates

    # --------------------------------------------------------
    # Score normalization
    # --------------------------------------------------------

    model_boredom = _clamp_score(
        data.get(
            "boredom_score",
            0,
        ),
        default=0,
    )

    model_reciprocity = _clamp_score(
        data.get(
            "reciprocity_score",
            0,
        ),
        default=0,
    )

    data["boredom_score"] = max(
        model_boredom,
        boredom,
    )

    # Use the model's reciprocity judgment when valid,
    # but keep deterministic scoring as a conservative floor
    # only when the model returned essentially no signal.
    if model_reciprocity <= 0:
        data["reciprocity_score"] = reciprocity
    else:
        data["reciprocity_score"] = model_reciprocity

    data["vibe"] = (
        data.get("vibe")
        or fallback_vibe
    )

    # --------------------------------------------------------
    # Deterministic safety adjustments
    # --------------------------------------------------------

    if (
        data["boredom_score"] >= 65
        and data.get("conversation_mode")
        not in {
            "space",
            "lighten",
        }
    ):
        data["conversation_mode"] = "lighten"

        if not data.get("next_goal"):
            data["next_goal"] = (
                "Change the rhythm with a lighter, "
                "more specific topic."
            )

    date_readiness = _clamp_score(
        data.get(
            "date_readiness",
            0,
        ),
        default=0,
    )

    data["date_readiness"] = date_readiness

    if (
        date_readiness >= 70
        and data.get("relationship_stage")
        not in {
            "date_ready",
            "cooling_off",
        }
    ):
        if not data.get("next_goal"):
            data["next_goal"] = (
                "Make one relaxed, concrete invitation "
                "when the conversation naturally opens the door."
            )

    # --------------------------------------------------------
    # Ensure array fields remain arrays
    # --------------------------------------------------------

    for field in (
        "positive_signals",
        "concerns",
    ):
        value = data.get(
            field,
            [],
        )

        if not isinstance(
            value,
            list,
        ):
            value = [str(value)]

        data[field] = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    # --------------------------------------------------------
    # Ensure scalar fields exist
    # --------------------------------------------------------

    defaults = {
        "send_mode": "text",
        "voice_script": "",
        "voice_reason": "",
        "vibe": fallback_vibe,
        "relationship_stage": (
            getattr(
                contact,
                "relationship_stage",
                "new",
            )
            or "new"
        ),
        "conversation_mode": "react",
        "next_goal": "",
        "question_strategy": "",
        "intimacy_guidance": "",
        "safety_note": "",
    }

    for field, default in defaults.items():
        if data.get(field) is None:
            data[field] = default

    return data
