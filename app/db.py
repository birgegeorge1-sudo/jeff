from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import re
from urllib.parse import urlsplit, parse_qs

from sqlalchemy import (
    Boolean,
    DateTime,
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
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

from .config import MYSQL_URL


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# ============================================================
# DATABASE URL
# ============================================================

def _prepare_mysql_url(raw_url: Optional[str]) -> str:
    """
    Normalize and validate MYSQL_URL.

    Supports:

        mysql://user:password@host:port/database
        mysql+mysqlconnector://user:password@host:port/database

    Also handles accidental values such as:

        MYSQL_URL=mysql://...
        MYSQL_URL=MYSQL_URL=mysql://...
    """

    if raw_url is None:
        raise RuntimeError(
            "MYSQL_URL is not set in the environment."
        )

    url = str(raw_url).strip()

    # Remove accidental surrounding quotes.
    if (
        len(url) >= 2
        and url[0] == url[-1]
        and url[0] in {"'", '"'}
    ):
        url = url[1:-1].strip()

    # Remove accidental control characters.
    url = re.sub(r"[\r\n\t]", "", url).strip()

    if not url:
        raise RuntimeError(
            "MYSQL_URL is present but empty."
        )

    # Remove accidental environment-variable prefixes.
    while url.startswith("MYSQL_URL="):
        url = url[len("MYSQL_URL="):].strip()

    # Use mysqlconnector for MySQL.
    if url.startswith("mysql://"):
        url = (
            "mysql+mysqlconnector://"
            + url[len("mysql://"):]
        )

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

    # Validate backend.
    if parsed.get_backend_name() != "mysql":
        raise RuntimeError(
            "MYSQL_URL uses backend "
            f"'{parsed.get_backend_name()}'; "
            "expected 'mysql'."
        )

    # Validate driver.
    if parsed.get_driver_name() != "mysqlconnector":
        raise RuntimeError(
            "MYSQL_URL uses driver "
            f"'{parsed.get_driver_name()}'; "
            "expected 'mysqlconnector'."
        )

    # Validate required connection fields.
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
# DATABASE ENGINE
# ============================================================

def _build_engine():
    """
    Build the SQLAlchemy MySQL engine.

    Handles Aiven-style:

        ?ssl-mode=REQUIRED

    by removing ssl-mode from the URL and passing the appropriate
    SSL settings through mysql-connector connect_args.
    """

    global DATABASE_URL

    query = dict(DATABASE_URL.query)
    ssl_mode: Optional[str] = None

    for key in list(query.keys()):
        normalized = key.lower().replace("_", "-")

        if normalized == "ssl-mode":
            value = query.pop(key)

            if isinstance(value, (tuple, list)):
                ssl_mode = (
                    str(value[0])
                    if value
                    else None
                )
            else:
                ssl_mode = (
                    str(value)
                    if value is not None
                    else None
                )

    DATABASE_URL = DATABASE_URL.set(
        query=query
    )

    connect_args: dict[str, Any] = {}

    if ssl_mode:
        mode = ssl_mode.upper()

        if mode in {
            "REQUIRED",
            "VERIFY_CA",
            "VERIFY_IDENTITY",
        }:
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
    Stored Discord contact/conversation identity.

    `name` is the actual database column.

    `display_name` and `canonical_name` are compatibility
    properties used by the Discord bot.

    `canonical_name` is a SQLAlchemy hybrid property, so code such as:

        select(Contact).where(
            Contact.canonical_name == "Esther"
        )

    works correctly.
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

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
        index=True,
    )

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

    # --------------------------------------------------------
    # DISPLAY NAME
    # --------------------------------------------------------

    @hybrid_property
    def display_name(self) -> str:
        """
        Human-readable contact name.
        """

        if self.name:
            return self.name

        if self.discord_username:
            return self.discord_username

        if self.discord_user_id:
            return self.discord_user_id

        return f"Contact #{self.id}"

    @display_name.expression
    def display_name(cls):
        """
        SQL expression for display_name.
        """

        return func.coalesce(
            cls.name,
            cls.discord_username,
            cls.discord_user_id,
            func.concat(
                "Contact #",
                cls.id,
            ),
        )

    # --------------------------------------------------------
    # CANONICAL NAME
    # --------------------------------------------------------

    @hybrid_property
    def canonical_name(self) -> str:
        """
        Canonical human-readable name.

        Instance usage:

            contact.canonical_name

        and class-level SQL usage:

            Contact.canonical_name == "Esther"
        """

        if self.name:
            return self.name

        if self.discord_username:
            return self.discord_username

        if self.discord_user_id:
            return self.discord_user_id

        return f"Contact #{self.id}"

    @canonical_name.expression
    def canonical_name(cls):
        """
        SQL expression for canonical_name.
        """

        return func.coalesce(
            cls.name,
            cls.discord_username,
            cls.discord_user_id,
            func.concat(
                "Contact #",
                cls.id,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"<Contact "
            f"id={self.id!r} "
            f"canonical_name={self.canonical_name!r} "
            f"display_name={self.display_name!r} "
            f"discord_user_id={self.discord_user_id!r} "
            f"is_active={self.is_active!r}>"
        )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db() -> None:
    """
    Create all missing tables.

    Existing data and existing tables are not deleted.
    """

    try:
        Base.metadata.create_all(
            bind=engine,
            checkfirst=True,
        )

    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Failed to initialize the database schema."
        ) from exc


def initialize_database() -> None:
    """
    Compatibility alias for init_db().
    """
    init_db()


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_optional_string(
    value: Optional[Any],
) -> Optional[str]:
    """
    Convert a value to a trimmed string or None.
    """

    if value is None:
        return None

    value = str(value).strip()

    return value if value else None


# ============================================================
# CONTACT LOOKUP
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

    Explicit fields are preferred.

    A generic value such as "Esther" is searched against:
        - name
        - discord username
        - discord user ID
        - email

    Numeric values can additionally match the database ID.
    """

    # --------------------------------------------------------
    # Discord user ID
    # --------------------------------------------------------

    if discord_user_id is not None:
        value = _normalize_optional_string(
            discord_user_id
        )

        if not value:
            return None

        stmt = select(Contact).where(
            Contact.discord_user_id == value
        )

        return (
            db.execute(stmt)
            .scalars()
            .first()
        )

    # --------------------------------------------------------
    # Discord username
    # --------------------------------------------------------

    if discord_username is not None:
        value = _normalize_optional_string(
            discord_username
        )

        if not value:
            return None

        stmt = select(Contact).where(
            Contact.discord_username == value
        )

        return (
            db.execute(stmt)
            .scalars()
            .first()
        )

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    if email is not None:
        value = _normalize_optional_string(email)

        if not value:
            return None

        stmt = select(Contact).where(
            Contact.email == value
        )

        return (
            db.execute(stmt)
            .scalars()
            .first()
        )

    # --------------------------------------------------------
    # Human-readable name
    # --------------------------------------------------------

    if name is not None:
        value = _normalize_optional_string(name)

        if not value:
            return None

        stmt = (
            select(Contact)
            .where(Contact.name == value)
            .order_by(Contact.id.desc())
        )

        return (
            db.execute(stmt)
            .scalars()
            .first()
        )

    # --------------------------------------------------------
    # Generic lookup
    # --------------------------------------------------------

    value = _normalize_optional_string(identifier)

    if not value:
        return None

    conditions = [
        Contact.name == value,
        Contact.discord_username == value,
        Contact.discord_user_id == value,
        Contact.email == value,
    ]

    # Numeric identifiers can also refer to database primary key.
    try:
        numeric_id = int(value)
    except (TypeError, ValueError):
        numeric_id = None

    if numeric_id is not None:
        conditions.insert(
            0,
            Contact.id == numeric_id,
        )

    stmt = (
        select(Contact)
        .where(or_(*conditions))
        .order_by(Contact.id.desc())
    )

    return (
        db.execute(stmt)
        .scalars()
        .first()
    )


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
    Find an existing contact or create a new contact.
    """

    normalized_identifier = (
        _normalize_optional_string(identifier)
    )

    normalized_discord_id = (
        _normalize_optional_string(
            discord_user_id
        )
    )

    normalized_username = (
        _normalize_optional_string(
            discord_username
        )
    )

    normalized_name = (
        _normalize_optional_string(name)
    )

    normalized_email = (
        _normalize_optional_string(email)
    )

    normalized_phone = (
        _normalize_optional_string(phone)
    )

    normalized_notes = (
        _normalize_optional_string(notes)
    )

    contact: Optional[Contact] = None

    # --------------------------------------------------------
    # Search in strongest order.
    # --------------------------------------------------------

    if normalized_discord_id:
        contact = find_contact(
            db,
            discord_user_id=normalized_discord_id,
        )

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

    if contact is None and normalized_name:
        contact = find_contact(
            db,
            name=normalized_name,
        )

    if contact is None and normalized_identifier:
        contact = find_contact(
            db,
            normalized_identifier,
        )

    # ========================================================
    # EXISTING CONTACT
    # ========================================================

    if contact is not None:

        changed = False

        if (
            normalized_discord_id
            and contact.discord_user_id
            != normalized_discord_id
        ):
            contact.discord_user_id = (
                normalized_discord_id
            )
            changed = True

        if (
            normalized_username
            and contact.discord_username
            != normalized_username
        ):
            contact.discord_username = (
                normalized_username
            )
            changed = True

        if (
            normalized_name
            and contact.name
            != normalized_name
        ):
            contact.name = normalized_name
            changed = True

        if (
            normalized_email
            and contact.email
            != normalized_email
        ):
            contact.email = normalized_email
            changed = True

        if (
            normalized_phone
            and contact.phone
            != normalized_phone
        ):
            contact.phone = normalized_phone
            changed = True

        if (
            normalized_notes
            and contact.notes
            != normalized_notes
        ):
            contact.notes = normalized_notes
            changed = True

        for key, value in extra_fields.items():
            if hasattr(contact, key):
                if getattr(contact, key) != value:
                    setattr(contact, key, value)
                    changed = True

        if activate:
            set_active_contact(
                db,
                contact,
                commit=False,
            )
            changed = True

        if changed:
            contact.updated_at = utc_now()

        db.flush()

        return contact

    # ========================================================
    # CREATE CONTACT
    # ========================================================

    contact = Contact(
        discord_user_id=normalized_discord_id,
        discord_username=normalized_username,
        name=(
            normalized_name
            or normalized_identifier
        ),
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
        set_active_contact(
            db,
            contact,
            commit=False,
        )

    return contact


# ============================================================
# ACTIVE CONTACT
# ============================================================

def get_active_contact(
    db,
) -> Optional[Contact]:
    """
    Return the currently active contact.
    """

    stmt = (
        select(Contact)
        .where(
            Contact.is_active.is_(True)
        )
        .order_by(
            Contact.updated_at.desc(),
            Contact.id.desc(),
        )
        .limit(1)
    )

    return (
        db.execute(stmt)
        .scalars()
        .first()
    )


def set_active_contact(
    db,
    contact: Any,
    *,
    commit: bool = True,
) -> Optional[Contact]:
    """
    Mark exactly one contact as active.

    Accepts:
        Contact instance
        database ID
        Discord ID
        Discord username
        human-readable contact name
    """

    if isinstance(contact, Contact):
        selected = contact
    else:
        selected = find_contact(
            db,
            contact,
        )

    if selected is None:
        return None

    now = utc_now()

    # Deactivate every contact.
    db.execute(
        update(Contact).values(
            is_active=False,
            updated_at=now,
        )
    )

    # Activate requested contact.
    selected.is_active = True
    selected.updated_at = now

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

    Accepts:
        Contact instance
        ID
        Discord ID
        username
        name
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
    Return contacts ordered newest-first.
    """

    stmt = (
        select(Contact)
        .order_by(
            Contact.created_at.desc(),
            Contact.id.desc(),
        )
    )

    if active_only:
        stmt = stmt.where(
            Contact.is_active.is_(True)
        )

    if limit is not None:
        stmt = stmt.limit(
            max(1, int(limit))
        )

    return list(
        db.execute(stmt)
        .scalars()
        .all()
    )


# ============================================================
# COUNT CONTACTS
# ============================================================

def count_contacts(db) -> int:
    """Return total contact count."""

    stmt = select(
        func.count(Contact.id)
    )

    return int(
        db.execute(stmt)
        .scalar_one()
    )


# ============================================================
# SESSION HELPERS
# ============================================================

def get_db():
    """
    Yield a SQLAlchemy database session.
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
    """Safely close a database session."""

    try:
        db.close()
    except Exception:
        pass


def commit_session(db) -> None:
    """Commit the current transaction."""

    try:
        db.commit()

    except SQLAlchemyError:
        db.rollback()
        raise


def rollback_session(db) -> None:
    """Rollback the current transaction."""

    try:
        db.rollback()

    except Exception:
        pass


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def check_database_connection() -> bool:
    """
    Return True if the MySQL server is reachable.
    """

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return True

    except SQLAlchemyError:
        return False


# ============================================================
# AUTOMATIC STARTUP INITIALIZATION
# ============================================================

def _startup_database_init() -> None:
    """
    Ensure required tables exist before the bot can receive
    commands.
    """
    init_db()


# Initialize schema immediately when app.db is imported.
_startup_database_init()


# ============================================================
# COMMAND-LINE SUPPORT
# ============================================================

if __name__ == "__main__":
    print(
        "[database] Initializing database..."
    )

    init_db()

    print(
        "[database] Database initialization complete."
    )

    if check_database_connection():
        print(
            "[database] Connection check: OK"
        )
    else:
        print(
            "[database] Connection check: FAILED"
        )


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "Base",
    "Contact",
    "DATABASE_URL",
    "MYSQL_CONNECTION_URL",
    "SessionLocal",
    "engine",
    "utc_now",
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
