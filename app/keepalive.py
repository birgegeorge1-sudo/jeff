from fastapi import FastAPI

app = FastAPI(title="Relationship Copilot Keepalive", version="1.1.0")


@app.get("/")
def root():
    return {"ok": True, "service": "relationship-copilot", "status": "online"}


@app.get("/health")
def health():
    return {"ok": True, "service": "relationship-copilot", "status": "healthy"}
