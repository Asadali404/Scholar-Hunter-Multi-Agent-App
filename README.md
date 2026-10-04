# 🎓 ScholarHunter Agents

Multi-agent scholarship finder (CrewAI + Groq + Streamlit) that only shows **authenticated** results.

## Pipeline
Profile Analyst → Scout → Architect (database + gap mentor) → **Verification Officer** → Tracker

## Authenticity gate
A scholarship is displayed only if the downloaded **official page** confirms all six:
✓ official domain · ✓ deadline · ✓ funding · ✓ degree level · ✓ eligibility · ✓ application URL

* The LLM *proposes* facts; the code accepts a fact only if its evidence text is on the page (hallucinations are dropped).
* Aggregators, blogs and social sites never count as official. Deadlines must be in the future and sit next to
  deadline wording; ambiguous dates (03/04/2027) are ignored, never guessed.
* The application link must be https, on a non-aggregator domain and must respond.
* Everything else is hidden (tick *Advanced → Also show NOT authenticated* to see why each one failed).

## Deadlines
Urgency: 🔴 Critical ≤14 days · 🟠 Soon ≤45 · 🟢 OK · ⚫ Closed · ⚪ Unknown. Critical items appear as red banners
(with Apply links) on the Discover, Deadlines and Tracker tabs. Export deadlines as `.ics` with 14- and 3-day reminders.

## API key
Paste a Groq key in the sidebar (optional, session-only, never saved). Otherwise the app uses `GROQ_API_KEY` from
Streamlit Secrets / `.env`.

## Run
```
pip install -r requirements.txt
cp .env.example .env     # optional
streamlit run app.py
python tests/test_core.py   # offline tests
```
