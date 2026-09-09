from pathlib import Path


def test_command_semantics():
    source = Path("app/main.py").read_text(encoding="utf-8")
    assert "!newconvo_" in source
    assert "!contconvo_" in source
    assert "!deleteconvo_" in source


def test_delete_clears_active_pointer():
    source = Path("app/db.py").read_text(encoding="utf-8")
    assert "state.active_contact_id = None" in source
    assert "db.delete(contact)" in source


def test_read_does_not_create_contacts():
    source = Path("app/main.py").read_text(encoding="utf-8")
    endpoint = source.split("@app.get('/conversations/{name}')", 1)[1].split("@app.post('/command')", 1)[0]
    assert "get_or_create_contact" not in endpoint
    assert "find_contact" in endpoint


def test_human_draft_policy_is_explicit():
    source = Path("app/engine.py").read_text(encoding="utf-8")
    assert "DRAFT for Jefferey to review" in source
    assert "Do not add AI/bot language" in source
    assert "do not manufacture an answer" in source


def test_discord_delete_command_exists():
    source = Path("app/discord_bot.py").read_text(encoding="utf-8")
    assert '@bot.command(name="deleteconvo")' in source
    assert "delete_contact" in source
