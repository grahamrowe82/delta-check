import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

STOP_WORDS = {"to", "the", "a", "an", "and"}
PUNCT_RE = re.compile(r"[^\w\s]")
COMPLETION_MARKERS = [
    "done",
    "completed",
    "sent",
    "shipped",
    "booked",
    "emailed",
    "fixed",
    "delivered",
]
SLIP_MARKERS = [
    "delay",
    "delayed",
    "pushed",
    "blocked",
    "won't be ready",
    "wont be ready",
    "next week",
    "later",
    "postpone",
]
DECISION_CHANGE_MARKERS = [
    "change",
    "switch",
    "reverse",
    "revisit",
    "cancel",
    "instead",
    "no longer",
    "go with",
]


def _clean_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped in {"-", "--", "—"}:
        return None
    return stripped


def tokenize(text: str) -> List[str]:
    cleaned = PUNCT_RE.sub(" ", text.lower())
    tokens: List[str] = []
    for raw in cleaned.split():
        if not raw or raw in STOP_WORDS:
            continue
        token = raw
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tokens


def normalize_action(text: str) -> str:
    return " ".join(tokenize(text))


def token_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def parse_prev(text_or_json: str) -> Dict[str, List]:
    actions: List[Dict[str, Optional[str]]] = []
    decisions: List[str] = []
    raw = (text_or_json or "").strip()
    if not raw:
        return {"actions": actions, "decisions": decisions}

    def _from_dict(obj: Dict) -> None:
        for item in obj.get("actions", []):
            if isinstance(item, dict):
                actions.append(
                    {
                        "action": item.get("action") or item.get("text") or "",
                        "owner": _clean_field(item.get("owner")),
                        "due": _clean_field(item.get("due")),
                    }
                )
        for item in obj.get("decisions", []) or []:
            if isinstance(item, str):
                decisions.append(item.strip())

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        _from_dict(parsed)
        if actions:
            return {"actions": actions, "decisions": decisions}
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                actions.append(
                    {
                        "action": item.get("action") or item.get("text") or "",
                        "owner": _clean_field(item.get("owner")),
                        "due": _clean_field(item.get("due")),
                    }
                )
            elif isinstance(item, str):
                actions.append({"action": item, "owner": None, "due": None})
        return {"actions": actions, "decisions": decisions}

    current_section: Optional[str] = None
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("actions"):
            current_section = "actions"
            remainder = line.split(":", 1)[1].strip() if ":" in line else ""
            if remainder:
                _push_action(actions, remainder)
            continue
        if lowered.startswith("decisions"):
            current_section = "decisions"
            remainder = line.split(":", 1)[1].strip() if ":" in line else ""
            if remainder:
                decisions.append(remainder)
            continue
        if current_section == "decisions":
            decisions.append(line.lstrip("-•").strip())
            continue
        _push_action(actions, line)

    return {"actions": actions, "decisions": decisions}


def _push_action(actions: List[Dict[str, Optional[str]]], text: str) -> None:
    entry = text.lstrip("-•").strip()
    if not entry:
        return
    parts = [p.strip() for p in re.split(r"\s+[—-]\s+", entry)]
    while len(parts) < 3:
        parts.append(None)
    actions.append(
        {
            "action": parts[0] or "",
            "owner": _clean_field(parts[1]) if parts[1] else None,
            "due": _clean_field(parts[2]) if parts[2] else None,
        }
    )


def extract_due(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    due_keywords = [
        "today",
        "tomorrow",
        "this week",
        "next week",
        "next month",
        "eod",
    ]
    for keyword in due_keywords:
        if keyword in lowered:
            return keyword
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "mon", "tue", "wed", "thu", "fri"]
    for day in days:
        if re.search(rf"\b{day}\b", lowered):
            return day
    if re.search(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", lowered):
        return re.search(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", lowered).group(0)
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered):
        return re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered).group(0)
    return None


def extract_owner(line: str) -> Optional[str]:
    if ":" in line:
        candidate = line.split(":", 1)[0].strip()
        if 1 <= len(candidate) <= 40:
            return candidate
    return None


def parse_new(transcript: str) -> Dict[str, List[Dict[str, Optional[str]]]]:
    actions: List[Dict[str, Optional[str]]] = []
    decisions: List[str] = []
    risks: List[str] = []
    lines: List[str] = []
    splitter = re.compile(r"\s+(?=[A-Z][a-z]+:)")
    for raw_line in (transcript or "").splitlines():
        chunks = splitter.split(raw_line.strip()) if raw_line.strip() else []
        for chunk in chunks or [raw_line.strip()]:
            line = chunk.strip()
            if not line:
                continue
            lines.append(line)
            lower = line.lower()
            if "risk" in lower or "blocked" in lower:
                risks.append(line)
            if any(keyword in lower for keyword in ["decide", "decision", "go with", "choose", "switch", "opt for", "instead"]):
                decisions.append(line)
            action_keywords = [
                "will",
                "need to",
                "going to",
                "gonna",
                "plan to",
                "should",
                "follow up",
                "schedule",
                "prepare",
                "draft",
                "email",
                "send",
                "book",
                "ship",
                "deliver",
                "update",
                "fix",
                "complete",
                "finish",
            ]
            if any(keyword in lower for keyword in action_keywords) and "risk" not in lower:
                owner = extract_owner(line)
                action_text = line.split(":", 1)[1].strip() if ":" in line else line
                actions.append(
                    {
                        "action": action_text,
                        "owner": owner,
                        "due": extract_due(line),
                        "evidence": line,
                    }
                )
    return {"actions": actions, "decisions": decisions, "risks": risks, "lines": lines}


def due_rank(phrase: Optional[str]) -> Optional[float]:
    if not phrase:
        return None
    lowered = phrase.lower()
    mapping = {
        "today": 1,
        "eod": 1.5,
        "tomorrow": 2,
        "this week": 3,
        "monday": 3,
        "tuesday": 3,
        "wednesday": 3,
        "thursday": 3,
        "friday": 3,
        "mon": 3,
        "tue": 3,
        "wed": 3,
        "thu": 3,
        "fri": 3,
        "next week": 4,
        "next month": 5,
    }
    if lowered in mapping:
        return mapping[lowered]
    if re.search(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday)\b", lowered):
        return 4
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered) or re.search(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", lowered):
        return 3
    return 4


def due_compare(old_due: Optional[str], new_due: Optional[str]) -> Optional[float]:
    old_rank = due_rank(old_due)
    new_rank = due_rank(new_due)
    if old_rank is None or new_rank is None:
        return None
    return new_rank - old_rank


def completion_detector(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in COMPLETION_MARKERS)


def slip_detector(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in SLIP_MARKERS)


def _matching_lines(action: str, owner: Optional[str], lines: Sequence[str]) -> List[str]:
    action_tokens = tokenize(action)
    matched: List[str] = []
    for line in lines:
        line_tokens = tokenize(line)
        if owner and owner.lower() in line.lower():
            matched.append(line)
            continue
        if token_jaccard(action_tokens, line_tokens) >= 0.5:
            matched.append(line)
    return matched


def decision_change_detector(old_decisions: Sequence[str], lines: Sequence[str]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for line in lines:
        lower = line.lower()
        if not any(marker in lower for marker in DECISION_CHANGE_MARKERS):
            continue
        if old_decisions:
            best_before, _ = _best_match(line, old_decisions)
            results.append({"before": best_before or old_decisions[0], "after": line.strip(), "evidence": line.strip()})
        else:
            results.append({"before": "—", "after": line.strip(), "evidence": line.strip()})
    return results


def _best_match(line: str, candidates: Sequence[str]) -> Tuple[Optional[str], float]:
    best_item: Optional[str] = None
    best_score = 0.0
    line_tokens = tokenize(line)
    for candidate in candidates:
        score = token_jaccard(line_tokens, tokenize(candidate))
        if score > best_score:
            best_score = score
            best_item = candidate
    return best_item, best_score


def match_and_classify(
    prev_actions: Sequence[Dict[str, Optional[str]]],
    new_data: Dict[str, Sequence],
    prev_decisions: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    prev_decisions = list(prev_decisions or [])
    kept: List[Dict[str, Optional[str]]] = []
    slipped: List[Dict[str, Optional[str]]] = []
    new_rows: List[Dict[str, Optional[str]]] = []
    at_risk: List[Dict[str, Optional[str]]] = []
    used_new: set = set()

    new_actions = list(new_data.get("actions", []))
    lines = list(new_data.get("lines", []))
    risks = list(new_data.get("risks", []))

    for prev in prev_actions:
        action_text = prev.get("action", "")
        owner = prev.get("owner")
        due = prev.get("due")
        action_tokens = tokenize(action_text)
        best_idx: Optional[int] = None
        best_score = 0.0
        for idx, candidate in enumerate(new_actions):
            score = token_jaccard(action_tokens, tokenize(candidate.get("action", "")))
            if score > best_score:
                best_score = score
                best_idx = idx
        matched_action: Optional[Dict[str, Optional[str]]] = None
        if best_idx is not None and best_score >= 0.5:
            matched_action = new_actions[best_idx]
            used_new.add(best_idx)

        matched_lines = _matching_lines(action_text, owner, lines)
        evidence_line = None
        is_kept = False
        is_slipped = False
        reason = None
        new_due = matched_action.get("due") if matched_action else None
        if not new_due:
            for line in matched_lines:
                potential_due = extract_due(line)
                if potential_due:
                    new_due = potential_due
                    break
        evidence_candidates = matched_lines.copy()
        if matched_action and matched_action.get("evidence"):
            evidence_candidates.insert(0, matched_action["evidence"])

        for line in evidence_candidates:
            if completion_detector(line):
                evidence_line = line
                is_kept = True
                break
        if not is_kept:
            for line in evidence_candidates:
                if slip_detector(line):
                    evidence_line = line
                    is_slipped = True
                    reason = "Delay indicator"
                    break
        if not is_slipped and not is_kept and due and new_due:
            diff = due_compare(due, new_due)
            if diff is not None and diff >= 1:
                is_slipped = True
                evidence_line = matched_action.get("evidence") if matched_action else None
                reason = f"Due moved to {new_due}"
        if is_kept:
            kept.append(
                {
                    "action": action_text,
                    "owner": owner,
                    "due": due,
                    "evidence": evidence_line or (matched_action or {}).get("evidence"),
                }
            )
        elif is_slipped:
            slipped.append(
                {
                    "action": action_text,
                    "owner": owner,
                    "due": new_due or due,
                    "evidence": evidence_line,
                    "reason": reason,
                }
            )

        risk_reasons: List[str] = []
        if not is_kept:
            rank = due_rank(due)
            if rank is not None and rank <= 2:
                risk_reasons.append("Due soon")
            for line in matched_lines:
                if "blocked" in line.lower():
                    risk_reasons.append("Blocked mention")
            for risk_line in risks:
                risk_tokens = tokenize(risk_line)
                overlap = [tok for tok in action_tokens if tok in risk_tokens and len(tok) > 2]
                if token_jaccard(action_tokens, risk_tokens) >= 0.3 or overlap:
                    risk_reasons.append(risk_line.strip())
            if risk_reasons:
                at_risk.append(
                    {
                        "action": action_text,
                        "owner": owner,
                        "why": "; ".join(dict.fromkeys(risk_reasons)),
                    }
                )

    for idx, action in enumerate(new_actions):
        if idx in used_new:
            continue
        new_rows.append(
            {
                "action": action.get("action"),
                "owner": action.get("owner"),
                "due": action.get("due"),
                "evidence": action.get("evidence"),
            }
        )

    decision_changes = decision_change_detector(prev_decisions, lines if lines else new_data.get("decisions", []))

    return {
        "kept": kept,
        "slipped": slipped,
        "new": new_rows,
        "at_risk": at_risk,
        "decision_changes": decision_changes,
    }


def build_delta(delta: Dict[str, List[Dict[str, Optional[str]]]]) -> Dict[str, List[Dict[str, Optional[str]]]]:
    sections = {key: list(delta.get(key, [])) for key in ["kept", "slipped", "new", "at_risk", "decision_changes"]}
    return sections
