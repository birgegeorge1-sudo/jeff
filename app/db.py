from datetime import datetime, timezone
from typing import Optional
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
    select,
)
from sqlalchemy.engine import make_url
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
# DATABASE URL
# ============================================================

def _prepare_mysql_url(raw_url: Optional[str]) -> str:
    if raw_url is None:
        raise RuntimeError(
            "MYSQL_URL is not set in the environment."
        )

    url = str(raw_url).strip()

    # Remove accidental wrapping quotes.
    if len(url) >= 2 and url[0] == url[-1] and url[0] in {"'", '"'}:
        url = url[1:-1].strip()

    # Remove accidental newlines/tabs.
    url = re.sub(r"[\r\n\t]", "", url).strip()

    if not url:
        raise RuntimeError(
            "MYSQL_URL is present but empty."
        )

    # Catch the very common mistake:
    # MYSQL_URL=MYSQL_URL=mysql+mysqlconnector://...
    if url.startswith("MYSQL_URL="):
        url = url[len("MYSQL_URL="):].strip()

    # Generic MySQL URLs are converted to the driver used by this project.
    if url.startswith("mysql://"):
        url = "mysql+mysqlconnector://" + url[len("mysql://"):]

    # Never print the password.
    try:
        parsed = make_url(url)
    except Exception as exc:
        # Safe diagnostic information only.
        try:
            parts = urlsplit(url)
            safe_scheme = parts.scheme or "<missing>"
            safe_host = parts.hostname or "<missing>"
            safe_port = parts.port or "<missing>"
            safe_path = parts.path or "<missing>"
            safe_user = parts.username or "<missing>"
            query_keys = sorted(parse_qs(parts.query).keys())
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

    if parsed.get_backend_name() != "mysql":
        raise RuntimeError(
            f"MYSQL_URL uses backend "
            f"'{parsed.get_backend_name()}'; expected 'mysql'."
        )

    if parsed.get_driver_name() != "mysqlconnector":
        raise RuntimeError(
            f"MYSQL_URL uses driver "
            f"'{parsed.get_driver_name()}'; expected 'mysqlconnector'."
        )

    if not parsed.username:
        raise RuntimeError("MYSQL_URL is missing the username.")

    if not parsed.password:
        raise RuntimeError("MYSQL_URL is missing the password.")

    if not parsed.host:
        raise RuntimeError("MYSQL_URL is missing the host.")

    if not parsed.port:
        raise RuntimeError("MYSQL_URL is missing the port.")

    if not parsed.database:
        raise RuntimeError("MYSQL_URL is missing the database name.")

    return url


MYSQL_CONNECTION_URL = _prepare_mysql_url(MYSQL_URL)

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
