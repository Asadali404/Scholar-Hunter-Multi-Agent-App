"""ScholarHunter Agents - Streamlit UI."""
# --- sqlite fix for Streamlit Community Cloud (CrewAI -> chromadb needs sqlite >= 3.35) ---
import os
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import crew as sh_crew  # noqa: E402
import export  # noqa: E402
import tracker  # noqa: E402
import verifier  # noqa: E402
from tools import parse_cv  # noqa: E402

st.set_page_config(page_title="ScholarHunter Agents", page_icon="🎓", layout="wide")

COUNTRIES = [
    "United Kingdom", "United States", "Germany", "Canada", "Australia", "China", "Turkey", "Hungary",
    "Japan", "South Korea", "Sweden", "Netherlands", "France", "Italy", "Malaysia", "Norway", "Finland",
]
STEPS = ["Profile", "Scout", "Architect", "Verifier", "Tracker"]
ICON = {"wait": "⬜", "run": "🔄", "done": "✅", "fail": "❌"}
URG_EMOJI = {"Critical": "🔴 Critical", "Soon": "🟠 Soon", "OK": "🟢 OK", "Closed": "⚫ Closed", "Verify": "⚪ Unknown"}
MAX_CV_MB = 5

for key, default in {
    "profile": None, "raw": [], "queries": [], "base_df": None, "gap": None,
    "summary": "", "run_id": 0, "warnings": [],
}.items():
    st.session_state.setdefault(key, default)


def get_api_key() -> tuple:
    """(key, source). Priority: key typed in the sidebar > Streamlit secrets > environment/.env."""
    typed = (st.session_state.get("user_key") or "").strip().strip('"').strip("'")
    if typed:
        return typed, "your key"
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip().strip('"').strip("'"), "the app's key"
    except Exception:
        pass
    env = os.getenv("GROQ_API_KEY", "").strip()
    return (env, "the app's key") if env else ("", "none")


def authentic(df):
    """Only records that passed all six verification checks are 'authentic'."""
    if df is None or df.empty:
        return df
    return df[df["verified"]].copy()


def visible(df):
    if df is None or df.empty:
        return df
    return df.copy() if st.session_state.get("show_unverified") else authentic(df)


# ------------------------------------------------------------------ header + sidebar
st.title("🎓 ScholarHunter Agents")
st.caption("Find. Verify. Prepare. Track. Only scholarships that pass all six verification checks are shown as authentic.")

with st.sidebar:
    st.header("Candidate inputs")
    cv_file = st.file_uploader("Upload CV (PDF or TXT)", type=["pdf", "txt"])
    interests = st.text_area("Research interests", height=100,
                             placeholder="e.g. federated learning, medical image analysis, low-resource NLP")
    domain = st.text_input("Target domain (optional)", placeholder="e.g. Artificial Intelligence")
    level = st.selectbox("Level", ["MS", "PhD", "Postdoc"])
    selected = st.multiselect("Countries", COUNTRIES, default=["United Kingdom", "Germany", "Turkey"])
    custom = st.text_input("Other countries (comma-separated)")
    max_searches = st.slider("Max searches", 1, 8, 6, help="More searches = more candidates to verify, but slower.")

    st.divider()
    st.subheader("🔑 API key (optional)")
    st.text_input(
        "Groq API key", type="password", key="user_key", placeholder="gsk_... (leave empty to use the app's key)",
        help="Used only for this browser session. It is never written to disk or shared with other visitors.",
    )
    _k, _src = get_api_key()
    if _k:
        st.caption(f"✅ Using {_src}.")
        if _src == "your key" and not _k.startswith("gsk_"):
            st.warning("Groq keys normally start with 'gsk_'. Double-check yours.")
    else:
        st.caption("⚠️ No key found - paste one above or ask the app owner to set GROQ_API_KEY.")
    with st.expander("Advanced"):
        parallel = st.checkbox("Run Tasks 3a/3b in parallel", value=True,
                               help="Turn off if you hit Groq free-tier rate limits.")
        model_override = st.text_input("Model ID (optional)", value="", placeholder=sh_crew.DEFAULT_MODEL_ID,
                                       help="Leave empty for the default model.")
        st.checkbox("Also show NOT authenticated candidates", key="show_unverified", value=False,
                    help="Off by default. Unauthenticated rows are clearly marked and have no confirmed deadline.")
    run_clicked = st.button("🚀 Run Agents", type="primary", use_container_width=True)
    if st.button("🔌 Test Groq connection", use_container_width=True):
        import requests

        key, _ = get_api_key()
        if not key:
            st.error("No API key found. Paste one above.")
        else:
            try:
                r = requests.get("https://api.groq.com/openai/v1/models",
                                 headers={"Authorization": f"Bearer {key}"}, timeout=15)
                if r.status_code == 200:
                    st.success("Connection OK - key accepted.")
                elif r.status_code in (401, 403):
                    st.error(f"Key rejected (HTTP {r.status_code}). Check that it is correct and active.")
                else:
                    st.warning(f"HTTP {r.status_code}")
                    st.code(r.text[:300])
            except Exception as exc:
                st.error(str(exc)[:300])
    if st.session_state.base_df is not None and st.button("🧹 Clear results", use_container_width=True):
        for k in ("profile", "raw", "queries", "base_df", "gap", "summary", "warnings"):
            st.session_state[k] = None if k in ("profile", "base_df", "gap") else ([] if k in ("raw", "queries", "warnings") else "")
        st.rerun()

tab_discover, tab_deadlines, tab_gap, tab_tracker, tab_export = st.tabs(
    ["Discover", "Deadlines", "Gap Analysis", "Tracker", "Export"])


# ------------------------------------------------------------------ pipeline
def execute():
    api_key, _src = get_api_key()
    countries = list(dict.fromkeys(selected + [c.strip() for c in custom.split(",") if c.strip()]))
    if not api_key:
        st.error("No Groq API key. Paste yours in the sidebar (🔑 API key) or set GROQ_API_KEY in Secrets / .env.")
        return
    if not cv_file and not interests.strip():
        st.error("Upload a CV or enter research interests.")
        return
    if not countries:
        st.error("Select at least one country.")
        return
    if cv_file is not None and cv_file.size > MAX_CV_MB * 1024 * 1024:
        st.error(f"The CV is larger than {MAX_CV_MB} MB.")
        return

    cv_text = ""
    if cv_file is not None:
        try:
            cv_text = parse_cv(cv_file, cv_file.name)
        except Exception as exc:
            st.error(f"Could not read the CV: {exc}")
            return

    states = {s: "wait" for s in STEPS}
    status = st.status("Running agents...", expanded=True)
    with status:
        board = st.empty()

        def paint():
            board.markdown("  →  ".join(f"{ICON[states[s]]} **{s}**" for s in STEPS))

        def step(name, state):
            states[name] = state
            paint()

        paint()
        warnings = []
        try:
            llm = sh_crew.get_llm(api_key, model=model_override.strip())

            step("Profile", "run")
            profile = sh_crew.analyze_profile(cv_text, interests, domain, countries, level, llm)
            step("Profile", "done")

            step("Scout", "run")
            raw, queries, warn = sh_crew.scout_opportunities(profile, level, max_searches, llm)
            if warn:
                warnings.append(warn)
            step("Scout", "done")

            step("Architect", "run")
            records, gap, warn = sh_crew.build_database_and_gaps(profile, raw, level, llm, parallel)
            if warn:
                warnings.append(warn)
            step("Architect", "done")

            step("Verifier", "run")
            records, warn = sh_crew.run_verification(profile, records, gap, level, llm)
            if warn:
                warnings.append(warn)
            n_ok = sum(1 for r in records if r.verified)
            if n_ok == 0:
                warnings.append("No candidate passed all six checks, so nothing is shown as authentic. "
                                "Try other countries, more searches, or tick 'Also show NOT authenticated candidates' "
                                "in Advanced to see why each one failed.")
            df = tracker.apply_state(tracker.records_to_df(records))
            step("Verifier", "done")

            step("Tracker", "run")
            summary = sh_crew.tracker_summary(authentic(df), llm) if n_ok else "No authenticated scholarships yet."
            step("Tracker", "done")
        except Exception as exc:
            for s in STEPS:
                if states[s] == "run":
                    states[s] = "fail"
            paint()
            status.update(label="Run failed", state="error")
            msg = str(exc)
            st.error(f"{type(exc).__name__}: {msg[:400]}")
            low = msg.lower()
            if "access denied" in low:
                st.info("Groq blocked this request before it reached your account (network/IP level). "
                        "Use 'Test Groq connection' in the sidebar, then reboot the app to get a different host.")
            elif "401" in low or "invalid api key" in low or "authentication" in low:
                st.info("Groq rejected the API key. Paste a valid key in the sidebar and press Run again.")
            else:
                st.info("Tip: on rate-limit errors, untick 'Run Tasks 3a/3b in parallel' or retry in a minute.")
            return
        status.update(label=f"All agents finished - {n_ok} authenticated of {len(records)} checked",
                      state="complete", expanded=False)

    st.session_state.update(
        profile=profile, raw=raw, queries=queries, gap=gap, summary=summary, warnings=warnings,
        base_df=df, run_id=st.session_state.run_id + 1,
    )


# ------------------------------------------------------------------ shared widgets
def render_deadline_alerts(df, key_prefix=""):
    """Critical deadlines are never hidden: shown as red banners with a countdown and a direct apply link."""
    df = authentic(df)
    if df is None or df.empty:
        return
    crit = tracker.critical_list(df)
    soon = tracker.urgent_list(df)
    soon = soon[soon["urgency"] == "Soon"]
    if crit.empty:
        nxt = tracker.upcoming_deadlines(df, 1)
        if nxt.empty:
            st.info("No confirmed upcoming deadlines yet.")
        else:
            r = nxt.iloc[0]
            st.success(f"No critical deadlines in the next {tracker.CRITICAL_DAYS} days. Next up: "
                       f"**{r['scholarship_name']}** on **{r['deadline']}** ({tracker.countdown_label(r['days_left'])}).")
    else:
        st.error(f"🚨 {len(crit)} CRITICAL deadline(s) - {tracker.CRITICAL_DAYS} days or fewer")
        for _, r in crit.iterrows():
            link = r["application_url"] or r["official_link"]
            st.markdown(f"🔴 **{r['scholarship_name']}** ({r['country']}) - closes **{r['deadline']}** "
                        f"({tracker.countdown_label(r['days_left'])}) · [Apply]({link})")
    if not soon.empty:
        with st.expander(f"🟠 {len(soon)} deadline(s) within {tracker.SOON_DAYS} days"):
            for _, r in soon.iterrows():
                st.markdown(f"- **{r['scholarship_name']}** - {r['deadline']} ({tracker.countdown_label(r['days_left'])})")


def render_checks(row):
    import json

    try:
        checks, notes = json.loads(row["checks_json"] or "{}"), json.loads(row["notes_json"] or "{}")
    except Exception:
        checks, notes = {}, {}
    for c in verifier.CHECKS:
        mark = "✅" if checks.get(c) else "❌"
        st.markdown(f"{mark} **{verifier.LABELS[c]}** - {notes.get(c, 'not checked')}")


# ------------------------------------------------------------------ Discover
with tab_discover:
    if run_clicked:
        execute()

    base = st.session_state.base_df
    if base is None:
        st.info("Fill in the sidebar and press **Run Agents**. Optional: paste your own Groq API key in the sidebar.")
        st.markdown("**How results are authenticated.** A scholarship is shown only when its official page confirms all of: "
                    "✓ official domain · ✓ deadline · ✓ funding · ✓ degree level · ✓ eligibility · ✓ application URL.")
    else:
        for w in st.session_state.warnings:
            st.warning(w)

        n_all, n_ok = len(base), int(base["verified"].sum()) if len(base) else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Authenticated", n_ok)
        m2.metric("Checked", n_all)
        m3.metric("Rejected", n_all - n_ok)

        render_deadline_alerts(base)

        with st.expander("Extracted profile", expanded=False):
            st.json(st.session_state.profile.model_dump())
        if st.session_state.queries:
            st.caption("Searches run: " + " | ".join(st.session_state.queries))

        view_all = visible(base)
        st.subheader(f"Scholarships ({0 if view_all is None else len(view_all)})")
        if view_all is None or view_all.empty:
            st.info("Nothing to show. No candidate passed every check.")
        else:
            f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
            f_country = f1.multiselect("Country", sorted(view_all["country"].unique()))
            f_urg = f2.multiselect("Urgency", list(URG_EMOJI.values()))
            f_status = f3.multiselect("Status", tracker.STATUSES)
            f_text = f4.text_input("Search name / provider")
            g1, g2 = st.columns([2, 2])
            f_fit = g1.slider("Minimum fit", 0, 100, 0)
            sort_by = g2.selectbox("Sort by", ["Deadline (soonest)", "Fit (best)", "Name (A-Z)"])

            view = view_all.copy()
            if f_country:
                view = view[view["country"].isin(f_country)]
            if f_urg:
                view = view[view["urgency"].map(URG_EMOJI).isin(f_urg)]
            if f_status:
                view = view[view["status"].isin(f_status)]
            if f_text.strip():
                t = f_text.strip().lower()
                view = view[view["scholarship_name"].str.lower().str.contains(t, regex=False)
                            | view["provider"].str.lower().str.contains(t, regex=False)]
            view = view[view["fit_score"] >= f_fit]
            if sort_by.startswith("Deadline"):
                view = view.assign(_k=view["days_left"].fillna(10**6).astype(int)).sort_values(["_k", "fit_score"], ascending=[True, False]).drop(columns="_k")
            elif sort_by.startswith("Fit"):
                view = view.sort_values("fit_score", ascending=False)
            else:
                view = view.sort_values("scholarship_name")

            st.caption("Deadlines come from the official page (not editable). You can edit Status and Notes; they are saved.")
            cols = ["id", "country", "scholarship_name", "provider", "level", "deadline", "days_left", "urgency",
                    "verification", "fit_score", "funding", "eligibility", "official_link", "application_url",
                    "status", "notes"]
            shown = view[cols].copy()
            shown["urgency"] = shown["urgency"].map(URG_EMOJI)
            sig = hash((tuple(f_country), tuple(f_urg), tuple(f_status), f_text, f_fit, sort_by, len(view)))
            edited = st.data_editor(
                shown, key=f"editor_{st.session_state.run_id}_{sig}", hide_index=True, use_container_width=True,
                disabled=[c for c in cols if c not in ("status", "notes")],
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "scholarship_name": st.column_config.TextColumn("Name", width="large"),
                    "deadline": st.column_config.TextColumn("Deadline"),
                    "days_left": st.column_config.NumberColumn("Days left"),
                    "verification": st.column_config.TextColumn(
                        "Checks", help="Order: domain, deadline, funding, level, eligibility, application URL"),
                    "fit_score": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d"),
                    "official_link": st.column_config.LinkColumn("Official page", display_text="Open"),
                    "application_url": st.column_config.LinkColumn("Apply", display_text="Apply"),
                    "status": st.column_config.SelectboxColumn("Status", options=tracker.STATUSES, required=True),
                },
            )
            ed = pd.DataFrame(edited)
            if not ed.empty:
                full = st.session_state.base_df.set_index("id")
                e = ed.set_index("id")
                full.loc[e.index, "status"] = e["status"]
                full.loc[e.index, "notes"] = e["notes"].fillna("")
                st.session_state.base_df = tracker.recompute(full.reset_index())
                tracker.save_state(st.session_state.base_df[st.session_state.base_df["verified"]])

            with st.expander("🔎 Verification details (evidence for every check)"):
                for _, r in view.iterrows():
                    st.markdown(f"**{r['scholarship_name']}** - {'✅ authenticated' if r['verified'] else '❌ NOT authenticated'}")
                    render_checks(r)
                    if r["deadline_source"]:
                        st.caption(f"Deadline evidence: {r['deadline_source']}")
                    st.divider()


# ------------------------------------------------------------------ Deadlines
with tab_deadlines:
    work = authentic(st.session_state.base_df)
    if work is None or work.empty:
        st.info("Run the agents to see confirmed deadlines.")
    else:
        render_deadline_alerts(work, "dl")
        up = tracker.upcoming_deadlines(work, 50)
        st.subheader("🗓️ All confirmed deadlines (soonest first)")
        if up.empty:
            st.info("No open items with a confirmed deadline.")
        else:
            t = up[["scholarship_name", "country", "deadline", "days_left", "urgency", "status", "application_url"]].copy()
            t["urgency"] = t["urgency"].map(URG_EMOJI)
            st.dataframe(t, hide_index=True, use_container_width=True, column_config={
                "application_url": st.column_config.LinkColumn("Apply", display_text="Apply"),
                "scholarship_name": st.column_config.TextColumn("Name", width="large"),
            })
            st.download_button("📅 Add deadlines to my calendar (.ics, with reminders at 14 and 3 days)",
                               data=tracker.to_ics(up).encode("utf-8"), file_name="scholarship_deadlines.ics",
                               mime="text/calendar")


# ------------------------------------------------------------------ Gap analysis
with tab_gap:
    gap, work = st.session_state.gap, visible(st.session_state.base_df)
    if gap is None or work is None or work.empty:
        st.info("Run the agents to see your gap analysis.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Strengths")
            for s in gap.strengths or ["Not enough information in the CV."]:
                st.markdown(f"- {s}")
        with c2:
            st.subheader("⚠️ Gaps")
            for g in gap.gaps or ["No gaps reported."]:
                st.markdown(f"- {g}")
        if gap.recommendations:
            st.subheader("🎯 Recommended actions")
            for r in gap.recommendations:
                st.markdown(f"- {r}")
        st.subheader("Per-scholarship fit")
        for _, r in work.sort_values("fit_score", ascending=False).iterrows():
            with st.expander(f"{r['scholarship_name']} - {r['country']}  |  fit {int(r['fit_score'])}/100"):
                st.progress(int(r["fit_score"]) / 100)
                st.markdown(f"**Provider:** {r['provider']}  \n**Deadline:** {r['deadline'] or 'not confirmed'} "
                            f"({tracker.countdown_label(r['days_left'])})  \n"
                            f"**Eligibility:** {r['eligibility'] or 'not listed'}  \n"
                            f"**Requirements:** {r['requirements'] or 'not listed'}")
                if r.get("top_gaps"):
                    st.markdown("**Top gaps:** " + str(r["top_gaps"]))
                if r["official_link"]:
                    st.markdown(f"[Official page]({r['official_link']})")


# ------------------------------------------------------------------ Tracker
with tab_tracker:
    work = authentic(st.session_state.base_df)
    if work is None or work.empty:
        st.info("Run the agents to start tracking.")
    else:
        render_deadline_alerts(work, "tr")
        c = tracker.status_counts(work)
        m = st.columns(5)
        m[0].metric("Total", c["total"])
        m[1].metric("Applied", c["applied"])
        m[2].metric("Pending", c["pending"])
        m[3].metric("Remaining", c["remaining"])
        m[4].metric("Critical", c["critical"])
        st.progress(c["applied"] / c["total"] if c["total"] else 0.0, text=f"{c['applied']} of {c['total']} applied")

        st.subheader("🗓️ Weekly priorities")
        st.markdown(st.session_state.summary or tracker.plain_summary(work))
        if st.button("Regenerate summary"):
            key, _ = get_api_key()
            if key:
                with st.spinner("Writing summary..."):
                    st.session_state.summary = sh_crew.tracker_summary(work, sh_crew.get_llm(key, model=model_override.strip()))
                st.rerun()
            else:
                st.error("No Groq API key. Paste yours in the sidebar.")


# ------------------------------------------------------------------ Export
with tab_export:
    base = st.session_state.base_df
    if base is None or base.empty:
        st.info("Nothing to export yet.")
    else:
        inc = st.checkbox("Include NOT authenticated candidates (marked in the file)", value=False)
        work = base if inc else authentic(base)
        if work is None or work.empty:
            st.info("No authenticated scholarships to export.")
        else:
            st.write("Excel workbook with **Scholarships** (clickable links, urgency colours), **Verification** "
                     "(all six checks and evidence) and **Gap Report** sheets.")
            st.download_button(
                "⬇️ Download Excel", data=export.to_excel_bytes(work, st.session_state.gap),
                file_name="scholarhunter_scholarships.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
            )
            st.download_button("⬇️ Download CSV", data=work[tracker.COLUMNS].to_csv(index=False).encode("utf-8"),
                               file_name="scholarhunter_scholarships.csv", mime="text/csv")
            st.download_button("📅 Download deadlines (.ics)", data=tracker.to_ics(work).encode("utf-8"),
                               file_name="scholarship_deadlines.ics", mime="text/calendar")
