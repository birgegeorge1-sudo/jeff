"""Deterministic conversation heuristics that supplement the LLM."""
import re

QUESTION_STARTS = re.compile(r"\b(what|why|how|where|when|who|which|would|could|do|did|have|has|are|is)\b", re.I)


def boredom_heuristic(messages):
    if not messages:
        return {"score": 0, "reasons": []}
    recent = messages[-8:]
    hers = [m.content.strip() for m in recent if m.sender == 'incoming']
    if not hers:
        return {"score": 0, "reasons": []}
    avg_len = sum(len(x.split()) for x in hers) / len(hers)
    questions = sum(1 for x in hers if '?' in x or QUESTION_STARTS.search(x))
    unique = len({x.casefold() for x in hers})
    score = 15
    reasons=[]
    if avg_len < 5:
        score += 25; reasons.append('recent replies are unusually brief')
    if len(hers) >= 3 and questions == 0:
        score += 15; reasons.append('low curiosity coming from her side')
    if unique < len(hers):
        score += 20; reasons.append('repetition is appearing')
    return {"score": min(score,100), "reasons": reasons}


def question_frequency(messages):
    recent = [m for m in messages[-10:] if m.sender == 'me']
    if not recent: return 0.0
    return sum('?' in m.content for m in recent) / len(recent)
