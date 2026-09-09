import os
from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)

    if value is None:
        return default

    # Remove accidental whitespace/newlines from copied environment values.
    value = value.strip()

    # Remove accidental surrounding quotes:
    # "value" -> value
    # 'value' -> value
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()

    return value


MISTRAL_API_KEY = _env("MISTRAL_API_KEY")

MODEL = _env(
    "MODEL",
    "mistral-large-latest",
)

MYSQL_URL = _env("MYSQL_URL")

HOST = _env(
    "HOST",
    "127.0.0.1",
)

PORT = int(
    _env("PORT", "8000")
)

MAX_HISTORY = int(
    _env("MAX_HISTORY", "40")
)

MIN_QUESTION_INTERVAL = int(
    _env("MIN_QUESTION_INTERVAL", "3")
)
