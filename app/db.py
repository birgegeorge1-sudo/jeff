from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import re
from urllib.parse import urlsplit, parse_qs

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import MYSQL_URL


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _prepare_mysql_url(raw_url: Optional[str]) -> str:
    if raw_url is None:
        raise RuntimeError("MYSQL_URL is not set in the environment.")
    url = str(raw_url).strip()
    if len(url) >= 2 and url[0] == url[-1] and url[0] in {"'", '"'}:
        url = url[1:-1].strip()
    url = re.sub(r"[\r\n\t]", "", url).strip()
    if not url:
        raise RuntimeError("MYSQL_URL is present but empty.")
    while url.startswith("MYSQL_URL="):
        url = url[len("MYSQL_URL="):].strip()
    if url.startswith("mysql://"):
        url = "mysql+mysqlconnector://" + url[len("mysql://"):]
    try:
        parsed = make_url(url)
    except Exception as exc:
        try:
            parts = urlsplit(url)
            safe_scheme = parts.scheme or "<missing>"
            safe_host = parts.hostname or "<missing>"
            safe_port = parts.port or "<missing>"
            safe_path = parts.path or "<missing>"
            safe_user = parts.username or "<missing>"
            query_keys = sorted(parse_qs(parts.query).keys())
        except Exception:
            safe_scheme = safe_host = safe_port = safe_path = safe_user = "<unparseable>"
            query_keys = []
        raise RuntimeError(
            "MYSQL_URL could not be parsed by SQLAlchemy.\n"
            f"scheme={safe_scheme}\nusername={safe_user}\nhost={safe_host}\n"
            f"port={safe_port}\ndatabase={safe_path}\nquery_keys={query_keys}\n"
            "The password is intentionally not shown."
        ) from exc
    if parsed.get_backend_name() != "mysql":
        raise RuntimeError(f"MYSQL_URL uses backend '{parsed.get_backend_name()}'; expected 'mysql'.")
    if parsed.get_driver_name() != "mysqlconnector":
        raise RuntimeError(f"MYSQL_URL uses driver '{parsed.get_driver_name()}'; expected 'mysqlconnector'.")
    for label, value in (
        ("username", parsed.username),
        ("password", parsed.password),
        ("host", parsed.host),
        ("port", parsed.port),
        ("database", parsed.database),
    ):
        if not value:
            raise RuntimeError(f"MYSQL_URL is missing the {label}.")
    return url


MYSQL_CONNECTION_URL = _prepare_mysql_url(MYSQL_URL)
DATABASE_URL = make_url(MYSQL_CONNECTION_URL)


def _build_engine():
    global DATABASE_URL
    query = dict(DATABASE_URL.query)
    ssl_mode: Optional[str] = None
    for key in list(query.keys()):
        normalized = key.lower().replace("_", "-")
        if normalized == "ssl-mode":
            value = query.pop(key)
            ssl_mode = str(value[0] if isinstance(value, (tuple, list)) and value else value)
    DATABASE_URL = DATABASE_URL.set(query=query)
    connect_args: dict[str, Any] = {}
    if ssl_mode:
        mode = ssl_mode.upper()
        if mode in {"REQUIRED", "VERIFY_CA", "VERIFY_IDENTITY"}:
            connect_args["ssl_disabled"] = False
        if mode == "VERIFY_CA":
            connect_args["ssl_verify_cert"] = True
        elif mode == "VERIFY_IDENTITY":
            connect_args["ssl_verify_cert"] = True
            connect_args["ssl_verify_identity"] = True
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        future=True,
        connect_args=connect_args,
    )


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Contact(Base):
    __tablename__ = "contacts"
    __table_args__ = (UniqueConstraint("discord_user_id", name="uq_contacts_discord_user_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    discord_user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    discord_username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    relationship_stage: Mapped[str] = mapped_column(String(50), nullable=False, default="new", server_default="new")
    vibe_score: Mapped[float] = mapped_column(Float, nullable=False, default=0, server_default="0")
    engagement_score: Mapped[float] = mapped_column(Float, nullable=False, default=0, server_default="0")
    date_readiness: Mapped[float] = mapped_column(Float, nullable=False, default=0, server_default="0")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    @hybrid_property
    def display_name(self) -> str:
        return self.name or self.discord_username or self.discord_user_id or f"Contact #{self.id}"

    @display_name.expression
    def display_name(cls):
        return func.coalesce(cls.name, cls.discord_username, cls.discord_user_id, func.concat("Contact #", cls.id))

    @hybrid_property
    def canonical_name(self) -> str:
        return self.display_name

    @canonical_name.expression
    def canonical_name(cls):
        return func.coalesce(cls.name, cls.discord_username, cls.discord_user_id, func.concat("Contact #", cls.id))


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"), nullable=False, index=True)
    sender: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(String(40), nullable=False, default="manual", server_default="manual")
    external_message_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[int] = mapped_column(Integer, nullable=False, default=3, server_default="3")
    source_message_id: Mapped[Optional[int]] = mapped_column(ForeignKey("messages.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


def _ensure_schema_updates() -> None:
    """Add columns introduced by this release to an older contacts table."""
    required = {
        "relationship_stage": "VARCHAR(50) NOT NULL DEFAULT 'new'",
        "vibe_score": "FLOAT NOT NULL DEFAULT 0",
        "engagement_score": "FLOAT NOT NULL DEFAULT 0",
        "date_readiness": "FLOAT NOT NULL DEFAULT 0",
    }
    try:
        with engine.begin() as connection:
            rows = connection.execute(
                text("""
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'contacts'
                """)
            ).scalars().all()
            existing = {str(x) for x in rows}
            for column, ddl in required.items():
                if column not in existing:
                    connection.execute(text(f"ALTER TABLE contacts ADD COLUMN {column} {ddl}"))
    except SQLAlchemyError as exc:
        raise RuntimeError("Failed to upgrade the existing database schema.") from exc


def init_db() -> None:
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        _ensure_schema_updates()
    except SQLAlchemyError as exc:
        raise RuntimeError("Failed to initialize the database schema.") from exc


def initialize_database() -> None:
    init_db()


def _normalize_optional_string(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def find_contact(db, identifier: Any = None, *, discord_user_id: Optional[Any] = None,
                 discord_username: Optional[Any] = None, email: Optional[Any] = None,
                 name: Optional[Any] = None) -> Optional[Contact]:
    if discord_user_id is not None:
        value = _normalize_optional_string(discord_user_id)
        if not value:
            return None
        return db.scalar(select(Contact).where(Contact.discord_user_id == value))
    if discord_username is not None:
        value = _normalize_optional_string(discord_username)
        if not value:
            return None
        return db.scalar(select(Contact).where(Contact.discord_username == value))
    if email is not None:
        value = _normalize_optional_string(email)
        if not value:
            return None
        return db.scalar(select(Contact).where(Contact.email == value))
    if name is not None:
        value = _normalize_optional_string(name)
        if not value:
            return None
        return db.scalar(select(Contact).where(func.lower(Contact.name) == value.casefold()).order_by(Contact.id.desc()))
    value = _normalize_optional_string(identifier)
    if not value:
        return None
    conditions = [
        func.lower(Contact.name) == value.casefold(),
        func.lower(Contact.discord_username) == value.casefold(),
        Contact.discord_user_id == value,
        func.lower(Contact.email) == value.casefold(),
    ]
    try:
        numeric_id = int(value)
    except (TypeError, ValueError):
        numeric_id = None
    if numeric_id is not None:
        conditions.insert(0, Contact.id == numeric_id)
    return db.scalar(select(Contact).where(or_(*conditions)).order_by(Contact.id.desc()))


def get_or_create_contact(db, identifier: Any = None, name: Optional[Any] = None, *,
                          discord_user_id: Optional[Any] = None, discord_username: Optional[Any] = None,
                          email: Optional[Any] = None, phone: Optional[Any] = None,
                          notes: Optional[Any] = None, activate: bool = False, **extra_fields: Any) -> Contact:
    normalized_identifier = _normalize_optional_string(identifier)
    normalized_discord_id = _normalize_optional_string(discord_user_id)
    normalized_username = _normalize_optional_string(discord_username)
    normalized_name = _normalize_optional_string(name)
    normalized_email = _normalize_optional_string(email)
    normalized_phone = _normalize_optional_string(phone)
    normalized_notes = _normalize_optional_string(notes)

    contact = None
    if normalized_discord_id:
        contact = find_contact(db, discord_user_id=normalized_discord_id)
    if contact is None and normalized_email:
        contact = find_contact(db, email=normalized_email)
    if contact is None and normalized_username:
        contact = find_contact(db, discord_username=normalized_username)
    if contact is None and normalized_name:
        contact = find_contact(db, name=normalized_name)
    if contact is None and normalized_identifier:
        contact = find_contact(db, normalized_identifier)

    if contact is not None:
        changed = False
        for attr, value in {
            "discord_user_id": normalized_discord_id,
            "discord_username": normalized_username,
            "name": normalized_name,
            "email": normalized_email,
            "phone": normalized_phone,
            "notes": normalized_notes,
        }.items():
            if value and getattr(contact, attr) != value:
                setattr(contact, attr, value)
                changed = True
        for key, value in extra_fields.items():
            if hasattr(contact, key) and getattr(contact, key) != value:
                setattr(contact, key, value)
                changed = True
        if activate:
            set_active_contact(db, contact, commit=False)
            changed = True
        if changed:
            contact.updated_at = utc_now()
        db.flush()
        return contact

    contact = Contact(
        discord_user_id=normalized_discord_id,
        discord_username=normalized_username,
        name=normalized_name or normalized_identifier,
        email=normalized_email,
        phone=normalized_phone,
        notes=normalized_notes,
        is_active=False,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    for key, value in extra_fields.items():
        if hasattr(contact, key):
            setattr(contact, key, value)
    db.add(contact)
    db.flush()
    if activate:
        set_active_contact(db, contact, commit=False)
    return contact


def get_active_contact(db) -> Optional[Contact]:
    return db.scalar(
        select(Contact)
        .where(Contact.is_active.is_(True))
        .order_by(Contact.updated_at.desc(), Contact.id.desc())
        .limit(1)
    )


def set_active_contact(db, contact: Any, *, commit: bool = True) -> Optional[Contact]:
    selected = contact if isinstance(contact, Contact) else find_contact(db, contact)
    if selected is None:
        return None
    now = utc_now()
    db.execute(update(Contact).values(is_active=False, updated_at=now))
    selected.is_active = True
    selected.updated_at = now
    db.flush()
    if commit:
        db.commit()
        db.refresh(selected)
    return selected


def save_message(db, contact_id: int, sender: str, content: str, platform: str = "manual",
                 external_message_id: Optional[str] = None) -> Message:
    sender = str(sender).strip().lower()
    if sender not in {"me", "incoming"}:
        raise ValueError("sender must be 'me' or 'incoming'")
    content = str(content).strip()
    if not content:
        raise ValueError("message content cannot be empty")
    message = Message(
        contact_id=int(contact_id), sender=sender, content=content,
        platform=str(platform or "manual"), external_message_id=external_message_id,
        created_at=utc_now(),
    )
    db.add(message)
    db.flush()
    return message


def recent_messages(db, contact_id: int, limit: int = 40) -> list[Message]:
    rows = list(db.scalars(
        select(Message)
        .where(Message.contact_id == int(contact_id))
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(max(1, int(limit)))
    ))
    rows.reverse()
    return rows


def get_memories(db, contact_id: int, limit: int = 50) -> list[Memory]:
    return list(db.scalars(
        select(Memory)
        .where(Memory.contact_id == int(contact_id))
        .order_by(Memory.importance.desc(), Memory.updated_at.desc(), Memory.id.desc())
        .limit(max(1, int(limit)))
    ))


def upsert_memory(db, contact_id: int, key: str, value: str, importance: int = 3,
                  source_message_id: Optional[int] = None) -> Memory:
    key = str(key).strip()
    value = str(value).strip()
    existing = db.scalar(
        select(Memory)
        .where(Memory.contact_id == int(contact_id), func.lower(Memory.key) == key.casefold())
        .order_by(Memory.id.desc())
    )
    if existing:
        existing.value = value
        existing.importance = max(1, min(5, int(importance)))
        existing.source_message_id = source_message_id
        existing.updated_at = utc_now()
        db.flush()
        return existing
    memory = Memory(
        contact_id=int(contact_id), key=key, value=value,
        importance=max(1, min(5, int(importance))), source_message_id=source_message_id,
        updated_at=utc_now(),
    )
    db.add(memory)
    db.flush()
    return memory


def add_event(db, contact_id: int, event_type: str, detail: Optional[str] = None,
              score: Optional[float] = None) -> Event:
    event = Event(
        contact_id=int(contact_id), event_type=str(event_type), detail=detail,
        score=score, created_at=utc_now(),
    )
    db.add(event)
    db.flush()
    return event


def delete_contact(db, contact: Any, *, commit: bool = True) -> bool:
    selected = contact if isinstance(contact, Contact) else find_contact(db, contact)
    if selected is None:
        return False
    db.query(Event).filter(Event.contact_id == selected.id).delete(synchronize_session=False)
    db.query(Memory).filter(Memory.contact_id == selected.id).delete(synchronize_session=False)
    db.query(Message).filter(Message.contact_id == selected.id).delete(synchronize_session=False)
    db.delete(selected)
    db.flush()
    if commit:
        db.commit()
    return True


def list_contacts(db, *, active_only: bool = False, limit: Optional[int] = None) -> list[Contact]:
    stmt = select(Contact).order_by(Contact.created_at.desc(), Contact.id.desc())
    if active_only:
        stmt = stmt.where(Contact.is_active.is_(True))
    if limit is not None:
        stmt = stmt.limit(max(1, int(limit)))
    return list(db.scalars(stmt))


def count_contacts(db) -> int:
    return int(db.scalar(select(func.count(Contact.id))) or 0)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def close_session(db) -> None:
    try:
        db.close()
    except Exception:
        pass


def commit_session(db) -> None:
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise


def rollback_session(db) -> None:
    try:
        db.rollback()
    except Exception:
        pass


def check_database_connection() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


_startup_database_init = init_db
# Keep schema initialization automatic for the existing deployment contract.
init_db()


__all__ = [
    "Base", "Contact", "Message", "Memory", "Event", "DATABASE_URL", "MYSQL_CONNECTION_URL",
    "SessionLocal", "engine", "utc_now", "init_db", "initialize_database", "get_db", "close_session",
    "commit_session", "rollback_session", "check_database_connection", "get_or_create_contact",
    "find_contact", "get_active_contact", "set_active_contact", "save_message", "recent_messages",
    "get_memories", "upsert_memory", "add_event", "delete_contact", "list_contacts", "count_contacts",
]
