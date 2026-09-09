from datetime import datetime, timezone
from typing import Optional
import re

from sqlalchemy import (
    create_engine,
    String,
    Text,
    DateTime,
    Integer,
    ForeignKey,
    UniqueConstraint,
    Float,
    select,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship

from .config import MYSQL_URL


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

def _clean_mysql_url(raw_url: Optional[str]) -> str:
    """
    Normalize the MYSQL_URL coming from the environment.

    Handles common deployment mistakes such as:
      - surrounding single/double quotes
      - accidental whitespace
      - embedded newlines
      - mysql:// URLs that don't specify a SQLAlchemy driver

    The function deliberately does NOT log the password.
    """
    if raw_url is None:
        raise RuntimeError(
            "MYSQL_URL is required. Set MYSQL_URL in the Render environment."
        )

    url = str(raw_url).strip()

    if not url:
        raise RuntimeError(
            "MYSQL_URL is empty. Set MYSQL_URL in the Render environment."
        )

    # Remove accidental wrapping quotes from Render/environment configuration.
    if len(url) >= 2 and url[0] == url[-1] and url[0] in {"'", '"'}:
        url = url[1:-1].strip()

    # Environment variables sometimes contain pasted line breaks.
    url = re.sub(r"[\r\n\t]", "", url).strip()

    if not url:
        raise RuntimeError(
            "MYSQL_URL became empty after normalization."
        )

    # If someone supplied a generic mysql:// URL, explicitly select
    # MySQL Connector/Python because this project uses mysqlconnector.
    if url.startswith("mysql://"):
        url = "mysql+mysqlconnector://" + url[len("mysql://"):]

    # Validate it before creating the engine so deployment errors are obvious.
    try:
        parsed = make_url(url)
    except Exception as exc:
        raise RuntimeError(
            "MYSQL_URL is malformed and could not be parsed by SQLAlchemy. "
            "Expected format: "
            "mysql+mysqlconnector://USER:PASSWORD@HOST:PORT/DATABASE"
        ) from exc

    if parsed.get_backend_name() != "mysql":
        raise RuntimeError(
            f"MYSQL_URL uses unsupported backend "
            f"'{parsed.get_backend_name()}'. Expected MySQL."
        )

    if parsed.get_driver_name() != "mysqlconnector":
        raise RuntimeError(
            "MYSQL_URL must use the mysqlconnector driver. "
            "Expected: mysql+mysqlconnector://..."
        )

    if not parsed.host:
        raise RuntimeError("MYSQL_URL is missing the database host.")

    if not parsed.database:
        raise RuntimeError("MYSQL_URL is missing the database name.")

    if not parsed.username:
        raise RuntimeError("MYSQL_URL is missing the database username.")

    if not parsed.port:
        raise RuntimeError("MYSQL_URL is missing the database port.")

    return url


MYSQL_CONNECTION_URL = _clean_mysql_url(MYSQL_URL)

# Use SQLAlchemy's parsed URL object so query-string handling is safe.
DATABASE_URL = make_url(MYSQL_CONNECTION_URL)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ============================================================
# ORM BASE
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# CONTACT
# ============================================================

class Contact(Base):
    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    canonical_name: Mapped[str] = mapped_column(
        String(160),
        unique=True,
        index=True,
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )

    notes: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    relationship_stage: Mapped[str] = mapped_column(
        String(40),
        default="new",
        nullable=False,
    )

    vibe_score: Mapped[float] = mapped_column(
        Float,
        default=50.0,
        nullable=False,
    )

    engagement_score: Mapped[float] = mapped_column(
        Float,
        default=50.0,
        nullable=False,
    )

    date_readiness: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    messages = relationship(
        "Message",
        back_populates="contact",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    memories = relationship(
        "Memory",
        back_populates="contact",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events = relationship(
        "ConversationEvent",
        back_populates="contact",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# ============================================================
# MESSAGE
# ============================================================

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    contact_id: Mapped[int] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    sender: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    platform: Mapped[str] = mapped_column(
        String(50),
        default="manual",
        nullable=False,
    )

    external_message_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )

    contact = relationship(
        "Contact",
        back_populates="messages",
    )


# ============================================================
# MEMORY
# ============================================================

class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    contact_id: Mapped[int] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    key: Mapped[str] = mapped_column(
        String(140),
        nullable=False,
    )

    value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    importance: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )

    source_message_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    contact = relationship(
        "Contact",
        back_populates="memories",
    )

    __table_args__ = (
        UniqueConstraint(
            "contact_id",
            "key",
            name="uq_contact_memory_key",
        ),
    )


# ============================================================
# CONVERSATION EVENTS
# ============================================================

class ConversationEvent(Base):
    __tablename__ = "conversation_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    contact_id: Mapped[int] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(60),
        nullable=False,
    )

    detail: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    contact = relationship(
        "Contact",
        back_populates="events",
    )


# ============================================================
# RUNTIME STATE
# ============================================================

class RuntimeState(Base):
    __tablename__ = "runtime_state"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    active_contact_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db() -> None:
    """
    Create missing tables.

    Existing tables/data are preserved.
    """
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Unable to initialize the MySQL database. "
            "Check MYSQL_URL, database availability, credentials, "
            "and the installed MySQL driver."
        ) from exc


# ============================================================
# NAME HELPERS
# ============================================================

def normalize_name(name: str) -> str:
    """
    Normalize contact names for reliable lookups.

    Example:
        '  Jane   Doe ' -> 'jane doe'
    """
    if name is None:
        return ""

    return " ".join(str(name).strip().split()).casefold()


# ============================================================
# CONTACT OPERATIONS
# ============================================================

def get_or_create_contact(db, name: str) -> Contact:
    canonical = normalize_name(name)

    if not canonical:
        raise ValueError("Contact name cannot be empty.")

    contact = db.scalar(
        select(Contact).where(
            Contact.canonical_name == canonical
        )
    )

    if contact:
        return contact

    contact = Contact(
        canonical_name=canonical,
        display_name=name.strip(),
    )

    try:
        db.add(contact)
        db.commit()
        db.refresh(contact)
        return contact

    except Exception:
        db.rollback()
        raise


def find_contact(db, name: str) -> Optional[Contact]:
    canonical = normalize_name(name)

    if not canonical:
        return None

    return db.scalar(
        select(Contact).where(
            Contact.canonical_name == canonical
        )
    )


def delete_contact(db, contact_id: int) -> bool:
    """
    Permanently delete a contact and its related messages,
    memories, and conversation events.

    Also clears the active contact when necessary.
    """
    contact = db.get(Contact, contact_id)

    if not contact:
        return False

    state = db.get(RuntimeState, 1)

    if state and state.active_contact_id == contact_id:
        state.active_contact_id = None

    try:
        db.delete(contact)
        db.commit()
        return True

    except Exception:
        db.rollback()
        raise


# ============================================================
# ACTIVE CONTACT
# ============================================================

def set_active_contact(db, contact: Contact) -> None:
    state = db.get(RuntimeState, 1)

    if not state:
        state = RuntimeState(
            id=1,
            active_contact_id=contact.id,
        )
        db.add(state)
    else:
        state.active_contact_id = contact.id

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def get_active_contact(db) -> Optional[Contact]:
    state = db.get(RuntimeState, 1)

    if not state or not state.active_contact_id:
        return None

    return db.get(
        Contact,
        state.active_contact_id,
    )


# ============================================================
# MESSAGE OPERATIONS
# ============================================================

def save_message(
    db,
    contact_id: int,
    sender: str,
    content: str,
    platform: str = "manual",
    external_message_id: Optional[str] = None,
) -> Message:
    if not content or not str(content).strip():
        raise ValueError("Message content cannot be empty.")

    message = Message(
        contact_id=contact_id,
        sender=sender,
        content=content,
        platform=platform,
        external_message_id=external_message_id,
    )

    try:
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    except Exception:
        db.rollback()
        raise


def recent_messages(
    db,
    contact_id: int,
    limit: int = 40,
) -> list[Message]:
    limit = max(1, min(int(limit), 500))

    rows = list(
        db.scalars(
            select(Message)
            .where(Message.contact_id == contact_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
    )

    rows.reverse()
    return rows


# ============================================================
# MEMORY OPERATIONS
# ============================================================

def upsert_memory(
    db,
    contact_id: int,
    key: str,
    value: str,
    importance: int = 3,
    source_message_id: Optional[int] = None,
) -> Memory:
    key = str(key).strip()

    if not key:
        raise ValueError("Memory key cannot be empty.")

    existing = db.scalar(
        select(Memory).where(
            Memory.contact_id == contact_id,
            Memory.key == key,
        )
    )

    if existing:
        existing.value = value
        existing.importance = importance
        existing.source_message_id = source_message_id
        memory = existing
    else:
        memory = Memory(
            contact_id=contact_id,
            key=key,
            value=value,
            importance=importance,
            source_message_id=source_message_id,
        )
        db.add(memory)

    try:
        db.commit()
        db.refresh(memory)
        return memory

    except Exception:
        db.rollback()
        raise


def get_memories(
    db,
    contact_id: int,
    limit: int = 50,
) -> list[Memory]:
    limit = max(1, min(int(limit), 500))

    return list(
        db.scalars(
            select(Memory)
            .where(Memory.contact_id == contact_id)
            .order_by(
                Memory.importance.desc(),
                Memory.updated_at.desc(),
            )
            .limit(limit)
        )
    )


# ============================================================
# CONVERSATION EVENTS
# ============================================================

def add_event(
    db,
    contact_id: int,
    event_type: str,
    detail: str = "",
    score: Optional[float] = None,
) -> ConversationEvent:
    event = ConversationEvent(
        contact_id=contact_id,
        event_type=event_type,
        detail=detail,
        score=score,
    )

    try:
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    except Exception:
        db.rollback()
        raise
