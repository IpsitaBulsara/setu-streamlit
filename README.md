# SETU — Streamlit prototype

Team Alucard · Maestros 2026

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501 in your browser.

## What's in each tab

- **Demand Sensing** — generates a synthetic 2-year weekly demand history and fits a real
  log-linear regression (NumPy least squares), backtested against a naive seasonal baseline
  on a 12-week holdout. Hit "Regenerate & refit" to rerun on fresh data.
- **Should-Cost** — synthetic cocoa/dairy/FX price indices (mean-reverting random walks),
  blended into a materials cost index, read for trend + z-score to issue a live
  BUY NOW / WAIT-HEDGE / NEUTRAL signal.
- **Control Tower & Agent** — a small 4-warehouse network that drains against demand each
  tick. When a node's days-of-cover drops below target, an agent computes and applies an
  actual rebalancing plan across the network. Use "Step" for a single manual tick during a
  live demo (more reliable than auto-run on stage), or "Auto-run 12 ticks" to let it play out.

All data is generated at runtime — no real Mondelez data required.
