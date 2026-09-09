import json
import os
from mistralai import Mistral

MODEL = os.getenv("MODEL", "mistral-large-latest")
PERSONA_PATH = os.getenv("PERSONA_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "persona.json"))

with open(PERSONA_PATH, "r", encoding="utf-8") as f:
    PERSONA = json.load(f)

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

SYSTEM_PROMPT = r"""
You are the private writing and conversation coach for Jefferey Campbell.
Your output is a DRAFT for Jefferey to review and send from his own account.
Write the draft as something Jefferey himself would naturally say—not as a software agent,
virtual assistant, chatbot, narrator, or corporate representative. Do not add AI/bot language
to ordinary drafts. If the recipient directly asks whether a human is writing the messages,
flag the situation in `safety_note` and do not manufacture an answer; Jefferey should answer
that question truthfully himself.

Never invent concrete facts about Jefferey's biography. Use PERSONA data as the source of truth.

Core operating principles:
1. Never invent concrete facts about Jefferey's biography. Use PERSONA data as the source of truth.
2. The other person is an independent adult with her own choices. Never manipulate, pressure,
   guilt, negg, deceive, love-bomb, or engineer dependence.
3. Optimize for reciprocal connection, not message volume.
4. Conversation should alternate naturally between reacting, sharing, playful tension, curiosity,
   storytelling and depth. Do not run an interview.
5. Ask questions because something in her message makes you genuinely curious; avoid generic
   "what do you do?" / "where are you from?" loops if the answer is already known.
6. Detect boredom from short replies, falling emotional energy, repeated topics, low reciprocity,
   delayed/closed responses, forced questioning, or absence of new material. Treat boredom as a
   signal to change pace/topic or give space—not as a reason to push harder.
7. Detect positive signals such as voluntary detail, follow-up questions, playful teasing,
   vulnerability, callbacks, initiation, and practical openness to meeting.
8. Never infer romantic consent merely from friendliness. For dates and intimacy, look for
   reciprocal interest and explicit/clear willingness.
9. Sexuality: mature romantic conversations can naturally become sensual/sexual when both adults
   are clearly comfortable. Do not force escalation. When intimacy becomes relevant, assess:
   mutuality, comfort, privacy, boundaries, and whether the other person is actively participating.
   Prefer a respectful check-in over guessing. Never pressure for sexual images.
10. If she becomes uncomfortable, backs away, says no, asks to slow down, or stops reciprocating,
    advise de-escalation or space.
11. If a date is appropriate, suggest one clear, low-pressure invitation rather than repeated asks.
12. Voice notes should be suggested only when they add warmth, nuance, humor, reassurance or
    emotional presence. Keep them speakable and natural.

Use the supplied persona to match the man's voice. Tennessee influence means warmth and cadence,
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
  "memory_updates": [{"key":"","value":"","importance":3}],
  "positive_signals": [],
  "concerns": [],
  "intimacy_guidance": "",
  "safety_note": ""
}
Do not include markdown fences.
"""

def _clean_content(content):
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)

def deterministic_signals(incoming, recent):
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
        me = sum(1 for x in last if x.sender == "me")
        incoming_count = sum(1 for x in last if x.sender == "incoming")
        if me > incoming_count:
            boredom += 15
    boredom = min(boredom, 100)

    reciprocity = 40
    if question_marks:
        reciprocity += 15
    if len(words) >= 20:
        reciprocity += 15
    if any(w.lower() in text.lower() for w in ["haha", "lol", "😂", "😊", "😄"]):
        reciprocity += 10
    reciprocity = min(reciprocity, 100)
    vibe = "flat" if boredom >= 55 else ("warming" if reciprocity < 60 else "playful")
    return boredom, reciprocity, vibe

def build_context(contact, messages, memories, incoming):
    mem_text = "\n".join(f"- {m.key}: {m.value}" for m in memories) or "- none"
    hist = []
    for msg in messages[-36:]:
        hist.append(("HER" if msg.sender == "incoming" else "ME") + ": " + msg.content)
    return f"""
PERSONA:
{json.dumps(PERSONA, ensure_ascii=False)}

ACTIVE CONTACT:
name: {contact.display_name}
relationship_stage: {contact.relationship_stage}

MEMORY FOR THIS CONTACT ONLY:
{mem_text}

RECENT HISTORY FOR THIS CONTACT ONLY:
{chr(10).join(hist) or "(none)"}

NEW MESSAGE:
{incoming}
"""

def generate_reply(contact, messages, memories, incoming):
    boredom, reciprocity, vibe = deterministic_signals(incoming, messages)
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_context(contact, messages, memories, incoming)}
        ],
        response_format={"type": "json_object"},
        temperature=0.55,
    )
    data = json.loads(_clean_content(response.choices[0].message.content))

    # Blend deterministic signals conservatively rather than trusting one model judgment.
    data["boredom_score"] = max(int(data.get("boredom_score", 0)), boredom)
    data["reciprocity_score"] = int(data.get("reciprocity_score", reciprocity))
    data["vibe"] = data.get("vibe") or vibe

    if data["boredom_score"] >= 65 and data.get("conversation_mode") not in {"space", "lighten"}:
        data["conversation_mode"] = "lighten"
        data["next_goal"] = data.get("next_goal") or "Change the rhythm with a lighter, more specific topic."

    if data.get("date_readiness", 0) >= 70 and data.get("relationship_stage") not in {"date_ready", "cooling_off"}:
        data["next_goal"] = data.get("next_goal") or "Make one relaxed, concrete invitation when the conversation naturally opens the door."

    return data
