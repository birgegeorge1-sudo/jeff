from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from .config import HOST, MAX_HISTORY, PORT
from .db import (
    Contact,
    SessionLocal,
    add_event,
    delete_contact,
    find_contact,
    get_active_contact,
    get_memories,
    get_or_create_contact,
    init_db,
    recent_messages,
    save_message,
    set_active_contact,
    upsert_memory,
)
from .engine import generate_reply
from .persona import load_persona
from .planner import boredom_heuristic, question_frequency

app = FastAPI(title="Relationship Copilot", version="1.1.0")


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CommandIn(BaseModel):
    command: str = Field(min_length=1, max_length=20000)


class MessageIn(BaseModel):
    contact: str = Field(min_length=1, max_length=160)
    message: str = Field(min_length=1, max_length=12000)
    platform: str = "manual"
    external_message_id: str | None = None


class SendDraftIn(BaseModel):
    contact: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=12000)
    platform: str = "manual"


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"ok": True, "service": "relationship-copilot", "version": "1.1.0"}


@app.get("/persona")
def persona():
    return load_persona()


@app.get("/conversations")
def conversations():
    with db_session() as db:
        rows = list(db.scalars(select(Contact).order_by(Contact.display_name)))
        active = get_active_contact(db)
        return [
            {
                "id": r.id,
                "name": r.display_name,
                "active": bool(active and active.id == r.id),
                "stage": r.relationship_stage,
                "vibe": r.vibe_score,
                "engagement": r.engagement_score,
                "date_readiness": r.date_readiness,
            }
            for r in rows
        ]


@app.get("/conversations/{name}")
def conversation(name: str):
    with db_session() as db:
        c = find_contact(db, name=name)
        if not c:
            raise HTTPException(404, f"No conversation exists for {name}.")
        msgs = recent_messages(db, c.id, 100)
        mems = get_memories(db, c.id, 100)
        return {
            "id": c.id,
            "name": c.display_name,
            "stage": c.relationship_stage,
            "vibe": c.vibe_score,
            "engagement": c.engagement_score,
            "date_readiness": c.date_readiness,
            "messages": [
                {
                    "id": m.id,
                    "sender": m.sender,
                    "content": m.content,
                    "platform": m.platform,
                    "created_at": m.created_at.isoformat(),
                }
                for m in msgs
            ],
            "memories": [{"key": m.key, "value": m.value, "importance": m.importance} for m in mems],
        }


def _parse_command(cmd: str):
    if not cmd.startswith("!"):
        return None
    body = cmd[1:].strip()
    lowered = body.casefold()
    if lowered.startswith("newconvo ") or lowered.startswith("newconvo:"):
        return "newconvo", body[len("newconvo"):].strip()
    if lowered.startswith("deleteconvo ") or lowered.startswith("deleteconvo:"):
        return "deleteconvo", body[len("deleteconvo"):].strip()
    match = __import__("re").match(r"^(?:cont)?([^:\n]{1,255}):\s*(.+)$", body, __import__("re").IGNORECASE | __import__("re").DOTALL)
    if match:
        return "contact_message", (" ".join(match.group(1).strip().split()), match.group(2).strip())
    return None


@app.post("/command")
def command(body: CommandIn):
    parsed = _parse_command(body.command.strip())
    if not parsed:
        raise HTTPException(400, "Unknown command. Use !newconvo Name: [\"first message\", \"response\"], !Name: incoming message, or !deleteconvo Name.")

    mode, payload = parsed
    if mode == "deleteconvo":
        name = payload.strip()
        with db_session() as db:
            c = find_contact(db, name=name)
            if not c:
                raise HTTPException(404, f"No conversation exists for {name}.")
            delete_contact(db, c.id)
            return {"ok": True, "mode": "delete", "deleted_contact": name}

    if mode == "newconvo":
        name, raw_history = (payload.split(":", 1) + [""])[:2] if ":" in payload else (payload, "")
        name = " ".join(name.strip().split())
        if not name:
            raise HTTPException(400, "A name is required.")
        with db_session() as db:
            if find_contact(db, name=name):
                raise HTTPException(409, f"Contact {name} already exists. Use !{name}: message to continue.")
            c = get_or_create_contact(db, name=name)
            set_active_contact(db, c, commit=False)
            seeded = 0
            if raw_history.strip():
                import json
                try:
                    history = json.loads(raw_history.strip())
                except json.JSONDecodeError as exc:
                    raise HTTPException(400, "History must be a valid JSON array.") from exc
                if not isinstance(history, list) or not history:
                    raise HTTPException(400, "History must be a non-empty JSON array.")
                for i, item in enumerate(history):
                    sender = "me" if i % 2 == 0 else "incoming"
                    content = item
                    if isinstance(item, dict):
                        sender = str(item.get("sender", item.get("role", sender))).lower()
                        sender = "incoming" if sender in {"her", "them", "contact", "incoming"} else "me"
                        content = item.get("message", item.get("content", ""))
                    if not str(content).strip():
                        raise HTTPException(400, f"History item {i + 1} is empty.")
                    save_message(db, c.id, sender, str(content).strip(), platform="api")
                    seeded += 1
            add_event(db, c.id, "conversation_created", detail=f"seeded_messages={seeded}")
            db.commit()
            return {"ok": True, "mode": "new", "active_contact": c.display_name, "contact_id": c.id, "seeded_messages": seeded}

    name, incoming = payload
    with db_session() as db:
        c = find_contact(db, name=name)
        if not c:
            raise HTTPException(404, f"No conversation exists for {name}. Create it with !newconvo {name}: [...].")
        set_active_contact(db, c, commit=False)
        incoming_message = save_message(db, c.id, "incoming", incoming, platform="api")
        history = recent_messages(db, c.id, MAX_HISTORY)
        result = generate_reply(c, history, get_memories(db, c.id, 50), incoming)
        result.setdefault("boredom", {})["heuristic"] = boredom_heuristic(history)
        result["coaching_note"] = (result.get("coaching_note", "") + f" Question frequency in recent drafts: {question_frequency(history):.0%}.").strip()
        for item in result.get("memory_updates", [])[:12]:
            key = str(item.get("key", "")).strip()
            val = str(item.get("value", "")).strip()
            if key and val:
                upsert_memory(db, c.id, key, val, max(1, min(5, int(item.get("importance", 3) or 3))), incoming_message.id)
        c.relationship_stage = result.get("relationship_stage", c.relationship_stage)
        add_event(db, c.id, "analysis", detail=str(result.get("concerns", [])), score=float(result.get("boredom_score", 0) or 0))
        db.commit()
        return {"contact": c.display_name, "active_contact_id": c.id, "draft": result, "guard": "Draft only — review before sending."}


@app.post("/message")
def incoming_message(body: MessageIn):
    with db_session() as db:
        c = find_contact(db, name=body.contact)
        if not c:
            raise HTTPException(404, f"No conversation exists for {body.contact}.")
        set_active_contact(db, c, commit=False)
        incoming = save_message(db, c.id, "incoming", body.message, body.platform, body.external_message_id)
        history = recent_messages(db, c.id, MAX_HISTORY)
        heuristic = boredom_heuristic(history)
        result = generate_reply(c, history, get_memories(db, c.id, 50), body.message)
        result.setdefault("boredom", {})["heuristic"] = heuristic
        result["coaching_note"] = (result.get("coaching_note", "") + f" Question frequency in recent drafts: {question_frequency(history):.0%}.").strip()
        c.relationship_stage = result.get("relationship_stage", c.relationship_stage)
        for item in result.get("memory_updates", [])[:12]:
            key = str(item.get("key", "")).strip()
            val = str(item.get("value", "")).strip()
            if key and val:
                upsert_memory(db, c.id, key, val, max(1, min(5, int(item.get("importance", 3) or 3))), incoming.id)
        db.commit()
        return {"contact": c.display_name, "active_contact_id": c.id, "draft": result, "guard": "Draft only — review before sending."}


@app.post("/draft-sent")
def draft_sent(body: SendDraftIn):
    with db_session() as db:
        c = find_contact(db, name=body.contact)
        if not c:
            raise HTTPException(404, f"No conversation exists for {body.contact}.")
        m = save_message(db, c.id, "me", body.content, body.platform)
        db.commit()
        return {"ok": True, "message_id": m.id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
