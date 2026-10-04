"""Verification Agent - nothing is shown as "authentic" unless ALL six checks pass:

  ✓ official domain   ✓ deadline   ✓ funding   ✓ degree level   ✓ eligibility   ✓ application URL

Design: an LLM may *propose* facts, but the code only accepts a fact when it can be found in the text of the
official page that was actually downloaded (evidence check). If the LLM is unavailable or hallucinates, a
deterministic extractor takes over, so the gate never relies on model memory.

This module has no LLM / pydantic imports so it is easy to unit-test; crew.py wires the agent in.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import tools

CHECKS = ["official_domain", "deadline", "funding", "degree_level", "eligibility", "application_url"]
LABELS = {
    "official_domain": "Official domain",
    "deadline": "Deadline",
    "funding": "Funding",
    "degree_level": "Degree level",
    "eligibility": "Eligibility",
    "application_url": "Application URL",
}

# ----------------------------------------------------------------------------
# Dates (never guess: ambiguous 03/04/2027 style dates are ignored)
# ----------------------------------------------------------------------------
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}
_MON = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
_DATE_RES = [
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("dmy", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})\.?,?\s+(\d{{4}})\b", re.I)),
    ("mdy", re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)),
    ("num", re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")),
]
_STRONG = re.compile(
    r"deadline|closing date|close[sd]?\b|apply by|applications? (?:must|should)|due (?:date|by|on)|last date|"
    r"no later than|submit(?:ted)? (?:by|before)|submission|until|before|by\b", re.I)
_NEGATIVE = re.compile(
    r"result|notif|announce|commence|start(?:s|ing)? (?:of|in|on)|begin|interview|decision|arriv|semester|"
    r"intake|orientation|published|updated|copyright|©|posted", re.I)


def _mk(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> List[Tuple[date, int, int]]:
    """All unambiguous calendar dates in the text as (date, start, end)."""
    out = []
    for kind, rx in _DATE_RES:
        for m in rx.finditer(text or ""):
            g = m.groups()
            d = None
            if kind == "iso":
                d = _mk(int(g[0]), int(g[1]), int(g[2]))
            elif kind == "dmy":
                d = _mk(int(g[2]), _MONTH_NAMES[g[1].lower()], int(g[0]))
            elif kind == "mdy":
                d = _mk(int(g[2]), _MONTH_NAMES[g[0].lower()], int(g[1]))
            else:  # numeric: only accept when day/month order is unambiguous
                a, b, y = int(g[0]), int(g[1]), int(g[2])
                if a > 12 >= b:
                    d = _mk(y, b, a)
                elif b > 12 >= a:
                    d = _mk(y, a, b)
            if d:
                out.append((d, m.start(), m.end()))
    return sorted(set(out), key=lambda x: x[1])


def deadline_candidates(text: str, today: Optional[date] = None, horizon_days: int = 550) -> List[dict]:
    """Dates that sit next to deadline wording, future only, nearest first. Each has the evidence snippet."""
    today = today or date.today()
    cands = []
    for d, s, e in find_dates(text):
        if not (today <= d <= today + timedelta(days=horizon_days)):
            continue
        before = text[max(0, s - 110): s]
        after = text[e: e + 30]
        window = before + " " + after
        if not _STRONG.search(window):
            continue
        near = text[max(0, s - 45): s]
        if _NEGATIVE.search(near) and not re.search(r"deadline|closing date|apply by", near, re.I):
            continue
        snippet = re.sub(r"\s+", " ", text[max(0, s - 90): e + 25]).strip()
        strong = bool(re.search(r"deadline|closing date|apply by|due|last date|no later than", window, re.I))
        cands.append({"date": d.isoformat(), "snippet": snippet, "strong": strong})
    # strong keyword first, then earliest date (conservative: earliest = safest for urgency)
    cands.sort(key=lambda c: (not c["strong"], c["date"]))
    seen, uniq = set(), []
    for c in cands:
        if c["date"] not in seen:
            seen.add(c["date"])
            uniq.append(c)
    return uniq


# ----------------------------------------------------------------------------
# Keyword families
# ----------------------------------------------------------------------------
LEVEL_WORDS = {
    "MS": re.compile(r"\bmaster'?s?\b|\bmsc\b|\bm\.?s\.?\b|\bm\.?sc\b|\bpostgraduate\b|\bmres\b|\bmphil\b|\bma\b|\bgraduate (?:degree|studies|programme|program)\b", re.I),
    "PhD": re.compile(r"\bph\.?\s?d\b|\bdoctora(?:l|te)\b|\bdphil\b|\bdoctor of philosophy\b", re.I),
    "Postdoc": re.compile(r"\bpost-?\s?doc(?:toral)?\b|\bpostdoctoral\b|\bresearch fellow", re.I),
}
_FUND_RE = re.compile(
    r"tuition|stipend|fully[- ]funded|full funding|allowance|living (?:cost|expense)|monthly|waiver|"
    r"travel|airfare|insurance|salary|bursary|grant|covers?|scholarship (?:includes|provides)|funded", re.I)
_ELIG_RE = re.compile(
    r"eligib|must (?:be|have|hold|not)|applicants? (?:must|should|need)|candidates? (?:must|should)|"
    r"citizen|national(?:s| of)|open to|required to|requirements?|qualif|bachelor|degree|gpa|english", re.I)
_APPLY_RE = re.compile(r"apply|application|portal|submit", re.I)
_SKIP_APPLY = re.compile(r"privacy|cookie|login help|newsletter|facebook|twitter|linkedin|instagram|youtube", re.I)


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [re.sub(r"\s+", " ", p).strip() for p in parts if 25 <= len(p.strip()) <= 400]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def in_text(snippet: str, text: str) -> bool:
    """Evidence check: the quoted snippet really occurs on the page (whitespace/punctuation-insensitive,
    with a tolerance for tiny LLM edits)."""
    sn, tx = _norm(snippet), _norm(text)
    if len(sn) < 12:
        return False
    if sn in tx:
        return True
    toks = sn.split()
    if len(toks) < 4:
        return False
    # allow minor deviations: >= 85% of the tokens must appear, in order, within a compact window
    pos, hit = 0, 0
    for t in toks:
        i = tx.find(t, pos)
        if i != -1 and i - pos < 200:
            hit += 1
            pos = i + len(t)
    return hit / len(toks) >= 0.85


# ----------------------------------------------------------------------------
# Domain authenticity
# ----------------------------------------------------------------------------
def _provider_tokens(provider: str, name: str) -> List[str]:
    stop = {"university", "college", "government", "foundation", "council", "ministry", "department", "of", "the",
            "and", "for", "scholarship", "scholarships", "program", "programme", "fellowship", "international",
            "national", "commission", "agency", "institute", "global", "professionals", "unknown", "excellence"}
    words = re.findall(r"[a-z]{4,}", f"{provider} {name}".lower())
    return [w for w in words if w not in stop]


def domain_verdict(url: str, provider: str = "", name: str = "", known_official: Optional[set] = None) -> Tuple[bool, str]:
    if not url or not url.lower().startswith("http"):
        return False, "No valid URL"
    host = urlparse(url).netloc.lower().split(":")[0]
    if tools.is_blocked(url):
        return False, f"{host} is a social/forum site"
    if tools.trust_score(url) == 0:
        return False, f"{host} is an aggregator/blog, not the official source"
    reg = tools.registered_domain(host)
    if known_official and reg in known_official:
        return True, f"{reg} is a known official programme domain"
    if tools.trust_score(url) == 3:
        return True, f"{reg} is an institutional/government domain"
    toks = _provider_tokens(provider, name)
    host_flat = re.sub(r"[^a-z]", "", host)
    if any(t in host_flat for t in toks):
        return True, f"{reg} matches the provider name"
    if tools.trust_score(url) == 2 and (host.endswith(".org") or host.endswith(".int")):
        return False, f"{reg} is a .org that does not clearly match the provider - treat as unconfirmed"
    return False, f"{reg} cannot be confirmed as the provider's official domain"


# ----------------------------------------------------------------------------
# Evidence gathering
# ----------------------------------------------------------------------------
_HOP_RE = re.compile(r"deadline|key dates|dates|how to apply|apply|application|eligib|requirements|funding|benefits|"
                     r"what.s (?:covered|included)|fees|costs", re.I)


def gather_page_bundle(url: str, hops: int = 2) -> dict:
    """Main page + up to `hops` same-site sub-pages (deadlines, eligibility, how to apply)."""
    main = tools.fetch_full(url)
    texts = [main.text]
    used = [main.final_url or url]
    links = list(main.links)
    if main.ok:
        picked = []
        for label, href in main.links:
            if len(picked) >= hops:
                break
            if href in used or href in picked or not tools.same_site(href, main.final_url or url):
                continue
            if _HOP_RE.search(label) or re.search(r"/(deadline|apply|eligib|how-to-apply|funding|dates)", href, re.I):
                if not re.search(r"\.(jpg|png|zip|docx?)$", href, re.I):
                    picked.append(href)
        for href in picked:
            sub = tools.fetch_full(href)
            if sub.ok and sub.text:
                texts.append(sub.text)
                used.append(href)
                links.extend(sub.links)
    return {"status": main.status, "ok": main.ok, "error": main.error, "final_url": main.final_url or url,
            "text": "\n".join(texts), "links": links, "pages": used}


def apply_link_candidates(bundle: dict, official_link: str) -> List[str]:
    out = []
    for label, href in bundle.get("links", []):
        if href in out or _SKIP_APPLY.search(label + " " + href):
            continue
        if _APPLY_RE.search(label) or re.search(r"/(apply|application|portal)", href, re.I):
            if tools.trust_score(href) > 0 and not tools.is_blocked(href):
                out.append(href)
    if official_link and re.search(r"apply|application|portal", official_link, re.I) and official_link not in out:
        out.insert(0, official_link)
    return out[:8]


def evidence_packet(rec_id: int, name: str, bundle: dict, official_link: str, today: date) -> dict:
    text = bundle.get("text", "")
    sents = _sentences(text)
    pick: List[str] = []
    for rx in (_FUND_RE, _ELIG_RE, LEVEL_WORDS["MS"], LEVEL_WORDS["PhD"]):
        pick += [s for s in sents if rx.search(s)][:4]
    seen, snippets, total = set(), [], 0
    for s in pick:
        if s not in seen and total < 2400:
            seen.add(s)
            snippets.append(s)
            total += len(s)
    return {
        "id": rec_id, "name": name, "official_link": official_link,
        "date_candidates": deadline_candidates(text, today)[:6],
        "apply_links": apply_link_candidates(bundle, official_link),
        "evidence": snippets,
    }


# ----------------------------------------------------------------------------
# Deterministic extractor (fallback and cross-check)
# ----------------------------------------------------------------------------
def heuristic_facts(bundle: dict, official_link: str, today: date) -> dict:
    text = bundle.get("text", "")
    sents = _sentences(text)
    cands = deadline_candidates(text, today)
    funding = next((s for s in sents if _FUND_RE.search(s)), "")
    elig = next((s for s in sents if _ELIG_RE.search(s) and re.search(r"eligib|must|should|citizen|national|require|open to", s, re.I)), "")
    levels = [k for k, rx in LEVEL_WORDS.items() if rx.search(text)]
    links = apply_link_candidates(bundle, official_link)
    return {
        "deadline": cands[0]["date"] if cands else None,
        "deadline_evidence": cands[0]["snippet"] if cands else "",
        "funding": funding, "funding_evidence": funding,
        "levels": levels, "eligibility": elig, "eligibility_evidence": elig,
        "application_url": links[0] if links else "",
    }


# ----------------------------------------------------------------------------
# The gate: six independent, evidence-based checks
# ----------------------------------------------------------------------------
def _level_ok(target: str, claimed: List[str], text: str) -> Tuple[bool, str]:
    target = (target or "").strip()
    keys = [k for k in LEVEL_WORDS if k.lower() == target.lower()]
    if not keys:
        return False, f"Unknown target level '{target}'"
    key = keys[0]
    page_has = bool(LEVEL_WORDS[key].search(text))
    claimed_ok = any(c.lower() == key.lower() for c in claimed) or not claimed
    if page_has and claimed_ok:
        return True, f"Page mentions {key}-level study"
    if not page_has:
        return False, f"Official page never mentions {key}-level applicants"
    return False, f"Programme is for {', '.join(claimed) or 'other levels'}, not {key}"


def run_checks(
    rec: dict, bundle: dict, facts_llm: Optional[dict], heur: dict, level: str,
    today: Optional[date] = None, known_official: Optional[set] = None,
    alive: Optional[Callable[[str], Tuple[bool, str]]] = None,
) -> dict:
    """Apply all six checks. LLM value first; if it fails the evidence test, the deterministic value is tried.
    Returns {checks, notes, facts, verified}. `rec` needs official_link, provider, scholarship_name."""
    today = today or date.today()
    alive = alive or tools.check_url_alive
    text = bundle.get("text", "")
    llm = facts_llm or {}
    checks: Dict[str, bool] = {}
    notes: Dict[str, str] = {}
    facts: Dict[str, str] = {}
    link = rec.get("official_link", "")

    # 1 - official domain (+ the page must really load)
    ok, why = domain_verdict(bundle.get("final_url") or link, rec.get("provider", ""), rec.get("scholarship_name", ""), known_official)
    if ok and not bundle.get("ok"):
        ok, why = False, f"Official page did not load ({bundle.get('error') or bundle.get('status') or 'no response'})"
    checks["official_domain"], notes["official_domain"] = ok, why

    # 2 - deadline: future date that appears on the page next to deadline wording
    cand_dates = {c["date"]: c for c in deadline_candidates(text, today)}
    chosen = None
    for val, _source in ((llm.get("deadline"), "llm"), (heur.get("deadline"), "code")):
        d = tools_parse_iso(val) if val else None
        if d and today <= d <= today + timedelta(days=550) and d.isoformat() in cand_dates:
            chosen = (d, cand_dates[d.isoformat()]["snippet"], _source)
            break
    if chosen:
        checks["deadline"], notes["deadline"] = True, f"Found on page: \"{chosen[1][:160]}\""
        facts["deadline"], facts["deadline_source"] = chosen[0].isoformat(), chosen[1][:200]
    else:
        any_past = [d for d, _, _ in find_dates(text) if d < today and d.year >= today.year - 1]
        checks["deadline"] = False
        notes["deadline"] = ("No confirmed upcoming deadline on the official page"
                             + (" (only past dates were found - the cycle may be closed)" if any_past else ""))

    # 3 - funding
    fund = ""
    for cand, ev in ((llm.get("funding"), llm.get("funding_evidence")), (heur.get("funding"), heur.get("funding_evidence"))):
        if cand and ev and in_text(ev, text) and _FUND_RE.search(cand + " " + ev):
            fund = str(cand).strip()
            break
    checks["funding"] = bool(fund)
    notes["funding"] = f"Page states: \"{fund[:160]}\"" if fund else "No funding details found on the official page"
    if fund:
        facts["funding"] = fund[:300]

    # 4 - degree level
    claimed = [str(x) for x in (llm.get("levels") or heur.get("levels") or [])]
    checks["degree_level"], notes["degree_level"] = _level_ok(level, claimed, text)
    # tighten: if level is PhD/MS but the page is clearly the other only, _level_ok already rejects
    if checks["degree_level"]:
        facts["level"] = level

    # 5 - eligibility
    elig = ""
    for cand, ev in ((llm.get("eligibility"), llm.get("eligibility_evidence")), (heur.get("eligibility"), heur.get("eligibility_evidence"))):
        if cand and ev and in_text(ev, text) and _ELIG_RE.search(cand + " " + ev):
            elig = str(cand).strip()
            break
    checks["eligibility"] = bool(elig)
    notes["eligibility"] = f"Page states: \"{elig[:160]}\"" if elig else "No eligibility criteria found on the official page"
    if elig:
        facts["eligibility"] = elig[:300]

    # 6 - application URL: a real, https, non-aggregator link that responds
    apply_options = apply_link_candidates(bundle, link)
    app_url = ""
    for cand in (llm.get("application_url"), heur.get("application_url")):
        if cand and str(cand).startswith("https://") and (cand in apply_options or cand == link):
            if tools.trust_score(cand) > 0 and not tools.is_blocked(cand):
                app_url = str(cand)
                break
    if app_url:
        up, status = alive(app_url)
        if up:
            checks["application_url"], notes["application_url"] = True, f"Responds (HTTP {status})"
            facts["application_url"] = app_url
        else:
            checks["application_url"], notes["application_url"] = False, f"Application link did not respond ({status})"
    else:
        checks["application_url"], notes["application_url"] = False, "No secure application link found on the official page"

    return {"checks": checks, "notes": notes, "facts": facts, "verified": all(checks.get(c) for c in CHECKS)}


def tools_parse_iso(v) -> Optional[date]:
    try:
        return datetime.strptime(str(v).strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def checklist_text(checks: Dict[str, bool]) -> str:
    return " ".join(f"{'✓' if checks.get(c) else '✗'} {LABELS[c]}" for c in CHECKS)


def checks_icons(checks: Dict[str, bool]) -> str:
    return "".join("✓" if checks.get(c) else "✗" for c in CHECKS)
