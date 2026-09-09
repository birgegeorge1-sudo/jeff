import json
from pathlib import Path

PERSONA_PATH = Path(__file__).resolve().parent.parent / 'persona.json'

def load_persona():
    return json.loads(PERSONA_PATH.read_text(encoding='utf-8'))
