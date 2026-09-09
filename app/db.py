from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import os
import re
from urllib.parse import urlsplit, parse_qs

from sqlalchemy import (
    create_engine,
    String,
    Text,
    DateTime,
    Integer,
    ForeignKey,
    UniqueConstraint,
    Float,
    Boolean,
    select,
    or_,
    update,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    relationship,
)

from .config import MYSQL_URL


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    """
    Return a timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


# ============================================================
# DATABASE URL
# ============================================================

def _prepare_mysql_url(raw_url: Optional[str]) -> str:
    """
    Normalize and validate MYSQL_URL.

    Supports:
        mysql://...
        mysql+mysqlconnector://...

    Also removes the common:
        MYSQL_URL=...
    mistake when the full environment assignment was pasted
    into the environment variable itself.
    """

    if raw_url is None:
        raise RuntimeError(
            "MYSQL_URL is not set in the environment."
        )

    url = str(raw_url).strip()

    # --------------------------------------------------------
    # Remove accidental wrapping quotes.
    # --------------------------------------------------------
    if (
        len(url) >= 2
        and url[0] == url[-1]
        and url[0] in {"'", '"'}
    ):
        url = url[1:-1].strip()

    # --------------------------------------------------------
    # Remove accidental whitespace/control characters.
    # --------------------------------------------------------
    url = re.sub(r"[\r\n\t]", "", url).strip()

    if not url:
        raise RuntimeError(
            "MYSQL_URL is present but empty."
        )

    # --------------------------------------------------------
    # Handle accidental:
    #
    # MYSQL_URL=MYSQL_URL=mysql://...
    #
    # and:
    #
    # MYSQL_URL=mysql://...
    # --------------------------------------------------------
    while url.startswith("MYSQL_URL="):
        url = url[len("MYSQL_URL="):].strip()

    # --------------------------------------------------------
    # Convert generic MySQL URL to the driver used here.
    # --------------------------------------------------------
    if url.startswith("mysql://"):
        url = "mysql+mysqlconnector://" + url[len("mysql://"):]

    # --------------------------------------------------------
    # Parse safely.
    # --------------------------------------------------------
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

            query_keys = sorted(
                parse_qs(parts.query).keys()
            )

        except Exception:
            safe_scheme = "<unparseable>"
            safe_host = "<unknown>"
            safe_port = "<unknown>"
            safe_path = "<unknown>"
            safe_user = "<unknown>"
            query_keys = []

        raise RuntimeError(
            "MYSQL_URL could not be parsed by SQLAlchemy.\n"
            f"scheme={safe_scheme}\n"
            f"username={safe_user}\n"
            f"host={safe_host}\n"
            f"port={safe_port}\n"
            f"database={safe_path}\n"
            f"query_keys={query_keys}\n"
            "The password is intentionally not shown."
        ) from exc

    # --------------------------------------------------------
    # Validate backend.
    # --------------------------------------------------------
    if parsed.get_backend_name() != "mysql":
        raise RuntimeError(
            "MYSQL_URL uses backend "
            f"'{parsed.get_backend_name()}'; expected 'mysql'."
        )

    # --------------------------------------------------------
    # Validate driver.
    # --------------------------------------------------------
    if parsed.get_driver_name() != "mysqlconnector":
        raise RuntimeError(
            "MYSQL_URL uses driver "
            f"'{parsed.get_driver_name()}'; "
            "expected 'mysqlconnector'."
        )

    # --------------------------------------------------------
    # Validate required components.
    # --------------------------------------------------------
    if not parsed.username:
        raise RuntimeError(
            "MYSQL_URL is missing the username."
        )

    if not parsed.password:
        raise RuntimeError(
            "MYSQL_URL is missing the password."
        )

    if not parsed.host:
        raise RuntimeError(
            "MYSQL_URL is missing the host."
        )

    if not parsed.port:
        raise RuntimeError(
            "MYSQL_URL is missing the port."
        )

    if not parsed.database:
        raise RuntimeError(
            "MYSQL_URL is missing the database name."
        )

    return url


MYSQL_CONNECTION_URL = _prepare_mysql_url(MYSQL_URL)

DATABASE_URL = make_url(MYSQL_CONNECTION_URL)


# ============================================================
# MYSQL CONNECTOR / SSL NORMALIZATION
# ============================================================

def _build_engine():
    """
    Build the SQLAlchemy engine.

    Aiven commonly supplies URLs containing:

        ?ssl-mode=REQUIRED

    That parameter is not passed blindly to mysql-connector-python.
    We remove it from the URL and explicitly enable SSL through
    connector arguments.
    """

    global DATABASE_URL

    query = dict(DATABASE_URL.query)

    ssl_mode = None

    # Handle several possible spellings/cases defensively.
    for key in list(query.keys()):
        normalized = key.lower().replace("_", "-")

        if normalized == "ssl-mode":
            values = query.pop(key)

            if isinstance(values, tuple):
                ssl_mode = values[0] if values else None
            else:
                ssl_mode = values

    # Rebuild URL without ssl-mode because the connector is
    # configured through connect_args below.
    DATABASE_URL = DATABASE_URL.set(query=query)

    connect_args: dict[str, Any] = {}

    if ssl_mode:
        normalized_ssl_mode = str(ssl_mode).upper()

        if normalized_ssl_mode in {
            "REQUIRED",
            "VERIFY_CA",
            "VERIFY_IDENTITY",
        }:
            connect_args["ssl_disabled"] = False

            # The driver can perform encryption without requiring
            # us to expose certificate paths here.
            #
            # If your provider requires explicit CA verification,
            # those connector settings can be supplied separately.
            if normalized_ssl_mode == "VERIFY_IDENTITY":
                connect_args["ssl_verify_identity"] = True

            elif normalized_ssl_mode == "VERIFY_CA":
                connect_args["ssl_verify_cert"] = True

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


# ============================================================
# SESSION
# ============================================================

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
# CONTACT MODEL
# ============================================================

class Contact(Base):
    """
    Stored contact used by the Discord bot.

    The model intentionally supports several identifiers because
    Discord integrations commonly need a stable user ID while still
    displaying a username/name to the operator.
    """

    __tablename__ = "contacts"

    __table_args__ = (
        UniqueConstraint(
            "discord_user_id",
            name="uq_contacts_discord_user_id",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Discord identity
    discord_user_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    discord_username: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # Human-readable contact information
    name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    email: Mapped[Optional[str]] = mapped_column(
        String(320),
        nullable=True,
        index=True,
    )

    phone: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    # Additional arbitrary notes/data
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Active contact state
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def __repr__(self) -> str:
        return (
            f"<Contact "
            f"id={self.id!r} "
            f"name={self.name!r} "
            f"discord_user_id={self.discord_user_id!r} "
            f"is_active={self.is_active!r}>"
        )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db() -> None:
    """
    Create missing tables.

    Existing tables are left untouched.
    """
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Failed to initialize the database."
        ) from exc


# ============================================================
# CONTACT INTERNAL HELPERS
# ============================================================

def _normalize_optional_string(
    value: Optional[Any],
) -> Optional[str]:
    """
    Convert optional values to stripped strings.
    Empty strings become None.
    """
    if value is None:
        return None

    value = str(value).strip()

    return value or None


def _find_contact_query(
    db,
    identifier: Any,
):
    """
    Build a flexible contact lookup query.

    Supports:
      - integer database ID
      - Discord user ID
      - username
      - email
      - name
    """

    identifier = _normalize_optional_string(identifier)

    if identifier is None:
        return None

    numeric_id = None

    try:
        numeric_id = int(identifier)
    except (TypeError, ValueError):
        pass

    conditions = [
        Contact.discord_user_id == identifier,
        Contact.discord_username == identifier,
        Contact.email == identifier,
        Contact.name == identifier,
    ]

    if numeric_id is not None:
        conditions.insert(
            0,
            Contact.id == numeric_id,
        )

    return select(Contact).where(
        or_(*conditions)
    )


# ============================================================
# FIND CONTACT
# ============================================================

def find_contact(
    db,
    identifier: Any = None,
    *,
    discord_user_id: Optional[Any] = None,
    discord_username: Optional[Any] = None,
    email: Optional[Any] = None,
    name: Optional[Any] = None,
) -> Optional[Contact]:
    """
    Find a contact.

    Usage examples:

        find_contact(db, "123456789")

        find_contact(
            db,
            discord_user_id="123456789"
        )

        find_contact(
            db,
            email="user@example.com"
        )
    """

    # Explicit keyword identifier takes priority.
    if discord_user_id is not None:
        stmt = select(Contact).where(
            Contact.discord_user_id
            == _normalize_optional_string(discord_user_id)
        )

    elif discord_username is not None:
        stmt = select(Contact).where(
            Contact.discord_username
            == _normalize_optional_string(discord_username)
        )

    elif email is not None:
        stmt = select(Contact).where(
            Contact.email
            == _normalize_optional_string(email)
        )

    elif name is not None:
        stmt = select(Contact).where(
            Contact.name
            == _normalize_optional_string(name)
        )

    else:
        stmt = _find_contact_query(
            db,
            identifier,
        )

        if stmt is None:
            return None

    return db.execute(stmt).scalars().first()


# ============================================================
# GET OR CREATE CONTACT
# ============================================================

def get_or_create_contact(
    db,
    identifier: Any = None,
    name: Optional[Any] = None,
    *,
    discord_user_id: Optional[Any] = None,
    discord_username: Optional[Any] = None,
    email: Optional[Any] = None,
    phone: Optional[Any] = None,
    notes: Optional[Any] = None,
    activate: bool = False,
    **extra_fields: Any,
) -> Contact:
    """
    Find an existing contact or create one.

    The function is intentionally flexible so Discord handlers
    can use either an ID-based or keyword-based call.
    """

    normalized_discord_id = _normalize_optional_string(
        discord_user_id
        if discord_user_id is not None
        else identifier
    )

    normalized_username = _normalize_optional_string(
        discord_username
    )

    normalized_name = _normalize_optional_string(name)
    normalized_email = _normalize_optional_string(email)
    normalized_phone = _normalize_optional_string(phone)
    normalized_notes = _normalize_optional_string(notes)

    # First try the strongest unique identifier.
    contact = None

    if normalized_discord_id:
        contact = find_contact(
            db,
            discord_user_id=normalized_discord_id,
        )

    # Fall back to other identifying information.
    if contact is None and normalized_email:
        contact = find_contact(
            db,
            email=normalized_email,
        )

    if contact is None and normalized_username:
        contact = find_contact(
            db,
            discord_username=normalized_username,
        )

    # --------------------------------------------------------
    # Existing contact
    # --------------------------------------------------------
    if contact is not None:

        changed = False

        if normalized_username:
            if contact.discord_username != normalized_username:
                contact.discord_username = normalized_username
                changed = True

        if normalized_name:
            if contact.name != normalized_name:
                contact.name = normalized_name
                changed = True

        if normalized_email:
            if contact.email != normalized_email:
                contact.email = normalized_email
                changed = True

        if normalized_phone:
            if contact.phone != normalized_phone:
                contact.phone = normalized_phone
                changed = True

        if normalized_notes:
            if contact.notes != normalized_notes:
                contact.notes = normalized_notes
                changed = True

        # Apply supported extra fields safely.
        for key, value in extra_fields.items():
            if hasattr(contact, key):
                setattr(contact, key, value)
                changed = True

        if activate:
            set_active_contact(
                db,
                contact.id,
                commit=False,
            )
            changed = True

        if changed:
            contact.updated_at = utc_now()

        db.flush()

        return contact

    # --------------------------------------------------------
    # Create new contact.
    # --------------------------------------------------------
    contact = Contact(
        discord_user_id=normalized_discord_id,
        discord_username=normalized_username,
        name=normalized_name,
        email=normalized_email,
        phone=normalized_phone,
        notes=normalized_notes,
        is_active=False,
    )

    # Apply supported extra fields.
    for key, value in extra_fields.items():
        if hasattr(contact, key):
            setattr(contact, key, value)

    db.add(contact)
    db.flush()

    if activate:
        set_active_contact(
            db,
            contact.id,
            commit=False,
        )

    return contact


# ============================================================
# GET ACTIVE CONTACT
# ============================================================

def get_active_contact(db) -> Optional[Contact]:
    """
    Return the currently active contact.

    Only one contact is intended to be active at a time.
    """

    stmt = (
        select(Contact)
        .where(Contact.is_active.is_(True))
        .order_by(
            Contact.updated_at.desc(),
            Contact.id.desc(),
        )
        .limit(1)
    )

    return db.execute(stmt).scalars().first()


# ============================================================
# SET ACTIVE CONTACT
# ============================================================

def set_active_contact(
    db,
    contact: Any,
    *,
    commit: bool = True,
) -> Optional[Contact]:
    """
    Make a contact the active contact.

    `contact` may be:
      - Contact instance
      - database ID
      - Discord user ID
      - username
    """

    if isinstance(contact, Contact):
        selected = contact

    else:
        # First try normal identifier lookup.
        selected = find_contact(
            db,
            contact,
        )

    if selected is None:
        return None

    # Deactivate all contacts.
    db.execute(
        update(Contact)
        .values(
            is_active=False,
            updated_at=utc_now(),
        )
    )

    # Activate selected contact.
    selected.is_active = True
    selected.updated_at = utc_now()

    db.flush()

    if commit:
        db.commit()
        db.refresh(selected)

    return selected


# ============================================================
# DELETE CONTACT
# ============================================================

def delete_contact(
    db,
    contact: Any,
    *,
    commit: bool = True,
) -> bool:
    """
    Delete a contact.

    `contact` may be:
      - Contact instance
      - database ID
      - Discord user ID
      - username
    """

    if isinstance(contact, Contact):
        selected = contact
    else:
        selected = find_contact(
            db,
            contact,
        )

    if selected is None:
        return False

    db.delete(selected)
    db.flush()

    if commit:
        db.commit()

    return True


# ============================================================
# LIST CONTACTS
# ============================================================

def list_contacts(
    db,
    *,
    active_only: bool = False,
    limit: Optional[int] = None,
) -> list[Contact]:
    """
    Return contacts ordered from newest to oldest.
    """

    stmt = select(Contact).order_by(
        Contact.created_at.desc(),
        Contact.id.desc(),
    )

    if active_only:
        stmt = stmt.where(
            Contact.is_active.is_(True)
        )

    if limit is not None:
        stmt = stmt.limit(max(1, int(limit)))

    return list(
        db.execute(stmt).scalars().all()
    )


# ============================================================
# COUNT CONTACTS
# ============================================================

def count_contacts(db) -> int:
    """
    Return the total number of contacts.
    """
    from sqlalchemy import func

    stmt = select(
        func.count(Contact.id)
    )

    return int(
        db.execute(stmt).scalar_one()
    )


# ============================================================
# SAFE SESSION HELPERS
# ============================================================

def get_db():
    """
    Generator-style database dependency.

    Useful for FastAPI/Flask-style integrations.
    """

    db = SessionLocal()

    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def close_session(db) -> None:
    """
    Safely close a SQLAlchemy session.
    """
    try:
        db.close()
    except Exception:
        pass


# ============================================================
# TRANSACTION HELPERS
# ============================================================

def commit_session(db) -> None:
    """
    Commit and refresh managed objects where appropriate.
    """
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise


def rollback_session(db) -> None:
    """
    Safely rollback the current transaction.
    """
    try:
        db.rollback()
    except Exception:
        pass


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def check_database_connection() -> bool:
    """
    Test whether the database is reachable.
    """

    from sqlalchemy import text

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return True

    except SQLAlchemyError:
        return False


# ============================================================
# OPTIONAL STARTUP INITIALIZATION
# ============================================================

def initialize_database() -> None:
    """
    Compatibility alias for application startup code.
    """
    init_db()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "Base",
    "Contact",
    "DATABASE_URL",
    "MYSQL_CONNECTION_URL",
    "SessionLocal",
    "engine",
    "init_db",
    "initialize_database",
    "get_db",
    "close_session",
    "commit_session",
    "rollback_session",
    "check_database_connection",
    "get_or_create_contact",
    "find_contact",
    "get_active_contact",
    "set_active_contact",
    "delete_contact",
    "list_contacts",
    "count_contacts",
]
