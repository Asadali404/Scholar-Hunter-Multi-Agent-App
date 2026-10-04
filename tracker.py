"""Pure-Python tracker logic: deadlines, urgency, status counts, calendar export, persistence."""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

COLUMNS = [
    "id", "country", "scholarship_name", "provider", "level", "deadline", "days_left", "urgency",
    "verification", "requirements", "eligibility", "funding", "official_link", "application_url",
    "confidence", "fit_score", "status", "notes", "deadline_source", "verified_on",
]
EXTRA = ["top_gaps", "verified", "checks_json", "notes_json"]
STATUSES = ["Remaining", "Pending", "Applied"]
STATE_PATH = Path(__file__).parent / "data" / "tracker_state.json"

CRITICAL_DAYS = 14
SOON_DAYS = 45

_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y")


def parse_deadline(value) -> Optional[date]:
    """Return a date or None. Never guesses: anything unclear is None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "n/a", "unknown", "rolling", "tbd", "nat", "nan"}:
        return None
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    if m:
        s = m.group(0)
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None  # no fuzzy parsing: a wrong deadline is worse than an unknown one


def days_left(deadline, today: Optional[date] = None) -> Optional[int]:
    d = parse_deadline(deadline)
    if d is None:
        return None
    return (d - (today or date.today())).days


def urgency(days: Optional[int]) -> str:
    """Critical <=14d, Soon <=45d, OK later, Closed if past, Verify if the deadline is unknown."""
    if days is None or pd.isna(days):
        return "Verify"
    days = int(days)
    if days < 0:
        return "Closed"
    if days <= CRITICAL_DAYS:
        return "Critical"
    if days <= SOON_DAYS:
        return "Soon"
    return "OK"


def countdown_label(days: Optional[int]) -> str:
    if days is None or pd.isna(days):
        return "no confirmed deadline"
    days = int(days)
    if days < 0:
        return f"closed {abs(days)} day(s) ago"
    if days == 0:
        return "TODAY"
    if days == 1:
        return "tomorrow"
    return f"in {days} days"


def records_to_df(records: Iterable) -> pd.DataFrame:
    from verifier import checks_icons  # local import keeps module import light

    rows = []
    for r in sorted(records, key=lambda x: (-int(x.verified), -x.fit_score)):
        d = r.model_dump()
        d["top_gaps"] = "; ".join(r.top_gaps)
        d.pop("checks", None)       # dict cells break Arrow/data_editor; kept as JSON strings below
        d.pop("check_notes", None)
        d["verification"] = checks_icons(r.checks) if r.checks else "------"
        d["checks_json"] = json.dumps(r.checks)
        d["notes_json"] = json.dumps(r.check_notes)
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=COLUMNS + EXTRA)
    df["id"] = range(1, len(df) + 1)
    for c in COLUMNS + EXTRA:
        if c not in df.columns:
            df[c] = None
    return recompute(df)


def recompute(df: pd.DataFrame, today: Optional[date] = None) -> pd.DataFrame:
    """(Re)compute days_left and urgency; normalise deadline strings and status."""
    df = df.copy()
    if df.empty:
        return df
    parsed = df["deadline"].apply(parse_deadline)
    df["deadline"] = parsed.apply(lambda d: d.isoformat() if d else None)
    df["days_left"] = pd.array([days_left(d, today) for d in parsed], dtype="Int64")
    df["urgency"] = df["days_left"].apply(lambda x: urgency(None if pd.isna(x) else int(x)))
    df["status"] = df["status"].where(df["status"].isin(STATUSES), "Remaining")
    df["notes"] = df["notes"].fillna("")
    df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce").fillna(0).astype(int)
    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(False).astype(bool)
    return df


def status_counts(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return dict(total=0, applied=0, pending=0, remaining=0, critical=0, soon=0, verify=0, closed=0)
    open_ = df[df["status"] != "Applied"]
    return dict(
        total=len(df),
        applied=int((df["status"] == "Applied").sum()),
        pending=int((df["status"] == "Pending").sum()),
        remaining=int((df["status"] == "Remaining").sum()),
        critical=int((open_["urgency"] == "Critical").sum()),
        soon=int((open_["urgency"] == "Soon").sum()),
        verify=int((open_["urgency"] == "Verify").sum()),
        closed=int((open_["urgency"] == "Closed").sum()),
    )


def urgent_list(df: pd.DataFrame, include_soon: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    levels = ["Critical", "Soon"] if include_soon else ["Critical"]
    m = df[(df["status"] != "Applied") & (df["urgency"].isin(levels))]
    return m.sort_values("days_left")


def critical_list(df: pd.DataFrame) -> pd.DataFrame:
    return urgent_list(df, include_soon=False)


def upcoming_deadlines(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Every open item with a known future deadline, soonest first (for the 'Deadlines' view)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    m = df[(df["status"] != "Applied") & df["days_left"].notna() & (df["urgency"] != "Closed")]
    return m.sort_values("days_left").head(limit)


def plain_summary(df: pd.DataFrame) -> str:
    c = status_counts(df)
    lines = [f"- {c['applied']} applied, {c['pending']} pending, {c['remaining']} remaining out of {c['total']}."]
    for _, r in urgent_list(df).head(5).iterrows():
        lines.append(f"- {r['scholarship_name']} ({r['country']}): deadline {r['deadline']}, "
                     f"{countdown_label(r['days_left'])} ({r['urgency']}).")
    if c["verify"]:
        lines.append(f"- {c['verify']} item(s) have no confirmed deadline; check the official sites.")
    return "\n".join(lines)


# ---------- calendar export (.ics) with reminders ----------
def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def to_ics(df: pd.DataFrame) -> str:
    """All-day events on each confirmed deadline, with alarms 14 and 3 days before."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//ScholarHunter//Deadlines//EN", "CALSCALE:GREGORIAN"]
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            d = parse_deadline(r.get("deadline"))
            if not d:
                continue
            uid = re.sub(r"[^a-z0-9]+", "-", f"{r.get('scholarship_name', '')}-{d}".lower()).strip("-")
            out += [
                "BEGIN:VEVENT", f"UID:{uid}@scholarhunter", f"DTSTAMP:{stamp}",
                f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}",
                f"DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime('%Y%m%d')}",
                f"SUMMARY:{_esc('DEADLINE: ' + str(r.get('scholarship_name', '')))}",
                f"DESCRIPTION:{_esc('Apply: ' + str(r.get('application_url') or r.get('official_link') or ''))}",
            ]
            for n in (14, 3):
                out += ["BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:Deadline in {n} days",
                        f"TRIGGER:-P{n}D", "END:VALARM"]
            out.append("END:VEVENT")
    out.append("END:VCALENDAR")
    return "\r\n".join(out) + "\r\n"


# ---------- persistence (status + notes survive new runs; deadlines are NOT persisted, they are re-verified) ----------
def _key(row) -> str:
    return (str(row.get("official_link") or "") or str(row.get("scholarship_name") or "")).strip().lower().rstrip("/")


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(df: pd.DataFrame) -> None:
    """Merge into the saved state (never drop items that are not in the current table)."""
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = load_state()
        for _, r in df.iterrows():
            state[_key(r)] = {"status": r["status"], "notes": r["notes"] or ""}
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass  # read-only filesystem on some hosts: tracker state is best-effort


def apply_state(df: pd.DataFrame) -> pd.DataFrame:
    state = load_state()
    if df.empty or not state:
        return df
    df = df.copy()
    for i, r in df.iterrows():
        s = state.get(_key(r))
        if s:
            df.at[i, "status"] = s.get("status", r["status"])
            df.at[i, "notes"] = s.get("notes", "")
    return recompute(df)
