```python
import os
import threading

import uvicorn
from fastapi import FastAPI

from app.discord_bot import run as run_bot


# Render requires the web service to listen on 0.0.0.0,
# not 127.0.0.1 / localhost.
HOST = "0.0.0.0"

# Render injects PORT automatically.
# Keep 8000 as the local-development fallback.
PORT = int(os.getenv("PORT", "8000"))


# Lightweight HTTP server used only to keep the Render Web Service
# reachable while the Discord bot runs in the same process.
keepalive_app = FastAPI(
    title="Relationship Copilot Keepalive",
    version="1.2.0",
)


@keepalive_app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "ok": True,
        "service": "relationship-copilot",
        "status": "online",
    }


@keepalive_app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {
        "ok": True,
        "service": "relationship-copilot",
        "status": "healthy",
    }


def run_keepalive():
    """
    Run the HTTP server on a Render-compatible public interface.

    IMPORTANT:
    Do not use HOST from the environment here. Some environments
    may provide HOST=127.0.0.1, which prevents Render from detecting
    the listening port externally.
    """
    uvicorn.run(
        keepalive_app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
    )


def main():
    # Start HTTP server in the background.
    web_thread = threading.Thread(
        target=run_keepalive,
        name="render-keepalive",
        daemon=True,
    )
    web_thread.start()

    # Keep the Discord bot as the main process.
    run_bot()


if __name__ == "__main__":
    main()
```
