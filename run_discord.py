import os
import threading

import uvicorn

from app.discord_bot import run as run_bot


def run_keepalive():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("app.keepalive:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    thread = threading.Thread(target=run_keepalive, name="render-keepalive", daemon=True)
    thread.start()
    run_bot()
