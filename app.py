"""
SETU — functional prototype (Streamlit port)
Team Alucard, Maestros 2026

Three modules, each doing real computation:
1. Demand sensing    — log-linear regression, fit + backtested live on synthetic data, vs a naive seasonal baseline
2. Should-cost model — REAL cocoa/milk/FX prices pulled live from Yahoo Finance -> trend/z-score buy signal
                        (falls back to synthetic mean-reverting series if Yahoo Finance is unreachable)
3. Control tower     — a live small warehouse network with an agent that actually rebalances stock

Run:  streamlit run app.py
"""

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import linprog

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title="SETU — Functional Prototype", layout="wide", page_icon="🧭")

# ---------------------------------------------------------------------------
# THEME / STYLE — purple & white only
# ---------------------------------------------------------------------------
PURPLE = "#4B2E83"        # primary brand purple
PURPLE_DEEP = "#2E1C57"   # darkest purple — dark accents (event log)
PURPLE_MID = "#7C5CB8"    # secondary chart series
PURPLE_SOFT = "#A78BDA"   # tertiary chart series
PURPLE_TEXT = "#33205E"   # text-safe purple for headings/metrics/tabs
GREEN = "#2F7A54"         # functional semantic only: buy / resolved / approved
RED = "#B23A2E"           # functional semantic only: wait / alert / guardrail
INK = "#241542"           # near-black purple, primary text on white
MUTED = "#6E5C90"         # secondary/muted purple-gray text on white
BG = "#FAF9FE"            # near-white app background
PANEL = "#FFFFFF"         # card background
LINE = "#E4DEF2"          # light purple-gray border
LOG_BG = PURPLE_DEEP
LOG_TEXT = "#EDE7F7"
LOG_MUTED = "#B9A8D6"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    h1, h2, h3 {{ color: {INK}; }}
    p, label, .stMarkdown {{ color: {INK}; }}
    .setu-tag {{ font-family: monospace; color: {MUTED}; font-size: 13px; margin-top: -12px; }}
    .pill {{
        display:inline-block; font-family: monospace; font-size: 11px; letter-spacing:0.06em;
        color: {PURPLE_TEXT}; border: 1px solid {PURPLE}; border-radius: 20px; padding: 3px 11px;
        background: {PANEL};
    }}
    .signal-buy {{ background: rgba(47,122,84,0.14); color:{GREEN}; border:1px solid {GREEN};
        padding:8px 16px; border-radius:3px; font-weight:700; display:inline-block; font-family:monospace;}}
    .signal-wait {{ background: rgba(178,58,46,0.14); color:{RED}; border:1px solid {RED};
        padding:8px 16px; border-radius:3px; font-weight:700; display:inline-block; font-family:monospace;}}
    .signal-neutral {{ background: rgba(75,46,131,0.08); color:{PURPLE_TEXT}; border:1px solid {PURPLE};
        padding:8px 16px; border-radius:3px; font-weight:700; display:inline-block; font-family:monospace;}}
    .wh-card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:4px; padding:14px; border-top:3px solid {LINE};}}
    .wh-ok {{ border-top-color: {GREEN}; }}
    .wh-alert {{ border-top-color: {RED}; }}
    .wh-card .wh-sub {{ color: {INK}; }}
    .log-box {{ background:{LOG_BG}; border-radius:4px; padding:10px 4px; max-height:260px; overflow-y:auto; }}
    .log-line {{ font-family: monospace; font-size: 12.5px; padding: 4px 0 4px 10px; border-left:2px solid {LOG_MUTED}; margin-bottom:2px;}}
    .log-alert {{ border-left-color: #e2867a; color:#f1b3ae; }}
    .log-resolve {{ border-left-color: #7fd1a8; color:#bfe9d5; }}
    .log-info {{ border-left-color: {LOG_MUTED}; color:{LOG_TEXT}; }}
    [data-testid="stMetricValue"] {{ color: {PURPLE_TEXT}; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    .stTabs [data-baseweb="tab"] {{ color: {MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {PURPLE_TEXT}; font-weight:600; }}
    .stButton button {{ background-color: {PURPLE}; color: #FFFFFF; border: none; }}
    .stButton button:hover {{ background-color: {PURPLE}; opacity:0.88; color: #FFFFFF; }}
    .stButton button p, .stButton button span, .stButton button div {{ color: #FFFFFF !important; }}
</style>
""", unsafe_allow_html=True)

st.title("SETU\u200a·\u200a functional prototype")
st.markdown('<div class="setu-tag">Unify · Sense · Simulate · Act — Team Alucard, Maestros 2026</div>', unsafe_allow_html=True)
st.write("")

SECTIONS = [
    "🚨 Connected Alert",
    "🌊1 Demand Forecast", "🌊1 Cost & Negotiation", "🌊1 Warehouse Stock Levels",
    "🌊2 Supplier Risk", "🌊2 Predictive Maintenance", "🌊2 Energy Scheduling",
    "🌊2 Factory & Warehouse Planner", "🌊3 Delivery & Cold Chain", "🌊3 Actions Timeline",
]

# ---------------------------------------------------------------------------
# SHARED NETWORK STATE — defined early so the persistent status header below,
# and every section that depends on it, always sees live data.
# ---------------------------------------------------------------------------
def reset_network():
    st.session_state.warehouses = {
        "WH-N": {"label": "WH-N (Delhi)", "stock": 1200, "demand": 90},
        "WH-W": {"label": "WH-W (Mumbai)", "stock": 620, "demand": 140},
        "WH-S": {"label": "WH-S (Chennai)", "stock": 950, "demand": 70},
        "WH-E": {"label": "WH-E (Kolkata)", "stock": 700, "demand": 60},
    }
    st.session_state.history = {k: [] for k in st.session_state.warehouses}
    st.session_state.tick = 0
    st.session_state.alerts_raised = 0
    st.session_state.alerts_resolved = 0
    st.session_state.ttm_samples = []
    st.session_state.log = [("info", "Network initialised. Target cover: 5 days.")]

def days_of_cover(w):
    return w["stock"] / w["demand"]

def agent_resolve(trigger_label):
    whs = st.session_state.warehouses
    total_stock = sum(w["stock"] for w in whs.values())
    total_demand = sum(w["demand"] for w in whs.values())
    network_cover = total_stock / total_demand
    moves = []
    for k, w in whs.items():
        target = round(w["demand"] * network_cover)
        delta = target - w["stock"]
        if abs(delta) >= 1:
            moves.append(f"{w['label']} {delta:+.0f}")
        w["stock"] = target
    simulated_ttm = round(float(np.random.uniform(1.5, 3.3)), 1)
    st.session_state.ttm_samples.append(simulated_ttm)
    st.session_state.alerts_resolved += 1
    st.session_state.log.insert(0, ("resolve",
        f"<b>Agent resolved</b> {trigger_label} shortfall via network rebalance "
        f"(target = equal days-of-cover): {', '.join(moves)}. "
        f"Simulated time-to-mitigate: {simulated_ttm}d (baseline manual process: 12d)."))

def sim_tick():
    st.session_state.tick += 1
    whs = st.session_state.warehouses
    for k, w in whs.items():
        draw = w["demand"] * np.random.uniform(0.85, 1.15)
        w["stock"] = max(0, w["stock"] - draw)
        w["stock"] += w["demand"] * 0.55  # trickle replenishment
        st.session_state.history[k].append(days_of_cover(w))
        if len(st.session_state.history[k]) > 40:
            st.session_state.history[k] = st.session_state.history[k][-40:]

    for k, w in whs.items():
        if days_of_cover(w) < 5:
            st.session_state.alerts_raised += 1
            st.session_state.log.insert(0, ("alert",
                f"<b>Alert</b> {w['label']} days-of-cover {days_of_cover(w):.1f}d — below 5d target. "
                f"Fill-rate risk flagged to control tower."))
            agent_resolve(w["label"])

if "warehouses" not in st.session_state:
    reset_network()

with st.sidebar:
    st.markdown("### SETU control tower")
    st.caption("One system: Sense → Reason → Act across Source, Plan, Make and Deliver.")
    section = st.radio("Go to", SECTIONS, key="nav_section", label_visibility="collapsed")
    st.caption("🌊1 = Plan & predict · 🌊2 = Manage risk & operations · 🌊3 = Deliver & fix issues")

with st.expander("👋 New here? How to use this", expanded=False):
    st.write(
        "This is a working demo of a connected supply-chain system for chocolate manufacturing. "
        "No special knowledge is needed — every screen explains itself in plain language."
    )
    st.markdown(
        "- **Start with 🚨 Connected Alert** (already open) — it shows one real problem, traced across the "
        "whole business on a single screen, with a clear recommendation you can approve or escalate.\n"
        "- **Use the menu on the left** to explore each part of the system on its own: demand planning, "
        "pricing, warehouse stock, supplier risk, maintenance, energy, factory planning, delivery, and a "
        "combined timeline of everything the system has done.\n"
        "- **Every button does one clear thing** — try clicking any 🔁 button to refresh the data, or any "
        "▶ button to have the system act on its recommendation.\n"
        "- **All data here is generated for this demo** — nothing here is a real Mondelez record."
    )

# --- persistent system status — always visible, independent of the section chosen below ---
_whs = st.session_state.get("warehouses", {})
_net_cover = (sum(w["stock"] for w in _whs.values()) / sum(w["demand"] for w in _whs.values())) if _whs else None
_shared_log = st.session_state.get("log", [])
_open_alerts = max(st.session_state.get("alerts_raised", 0) - st.session_state.get("alerts_resolved", 0), 0)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Network days of cover", f"{_net_cover:.1f} d" if _net_cover is not None else "—")
s2.metric("Cost outlook", st.session_state.get("procurement_cost_signal", "not yet checked"))
s3.metric("Open alerts", _open_alerts)
s4.metric("Agent actions logged", len(_shared_log))
st.divider()

# ---------------------------------------------------------------------------
# SHARED PLOT LAYOUT
# ---------------------------------------------------------------------------
def base_layout(title):
    return dict(
        title=dict(text=title, font=dict(color=INK, size=14)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=INK), legend=dict(font=dict(color=INK)),
        xaxis=dict(gridcolor=LINE), yaxis=dict(gridcolor=LINE),
        margin=dict(t=40, l=40, r=20, b=30),
    )

# ===========================================================================
# MODULE 1 — DEMAND SENSING
# ===========================================================================
def gen_demand_series(seed, weeks=104):
    rng = np.random.default_rng(seed)
    w = np.arange(weeks)
    woy = w % 52
    trend = 1.15 / weeks
    level = 1000 * (1 + trend) ** w
    seasonal = 1 + 0.28*np.sin((woy/52)*2*np.pi - 1.2) + 0.15*np.sin((woy/26)*2*np.pi)
    promo = (rng.random(weeks) < 0.12).astype(float)
    promo_lift = promo * (0.35 + 0.25*rng.random(weeks))
    weather_dev = rng.normal(0, 1, weeks)
    weather_effect = -0.04 * weather_dev
    competitor_shock = np.where(rng.random(weeks) < 0.10, -(0.15 + 0.3*rng.random(weeks)), 0.0)
    local_shock = np.where(rng.random(weeks) < 0.06, rng.normal(0, 0.35, weeks), 0.0)
    noise = rng.normal(0, 0.18, weeks) + local_shock
    demand = level * seasonal * (1+promo_lift) * (1+weather_effect) * (1+competitor_shock) * (1+noise)
    demand = np.maximum(50, demand)
    return w, woy, promo, weather_dev, demand

def featurize_demand(w, woy, promo, weather_dev):
    ang = (woy/52)*2*np.pi
    return np.column_stack([
        np.ones_like(w, dtype=float), w, np.sin(ang), np.cos(ang), np.sin(2*ang), promo, weather_dev
    ])

def run_demand_module():
    seed = np.random.randint(0, 1_000_000_000)
    w, woy, promo, weather_dev, demand = gen_demand_series(seed)
    n_train = 92
    Xtr = featurize_demand(w[:n_train], woy[:n_train], promo[:n_train], weather_dev[:n_train])
    ytr = np.log(demand[:n_train])
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    Xte = featurize_demand(w[n_train:], woy[n_train:], promo[n_train:], weather_dev[n_train:])
    model_pred = np.exp(Xte @ beta)
    actual = demand[n_train:]

    # naive seasonal baseline: mean of same week-of-year in training set
    naive_pred = np.array([
        demand[:n_train][woy[:n_train] == wk].mean() if (woy[:n_train] == wk).any() else demand[:n_train].mean()
        for wk in woy[n_train:]
    ])

    def acc(a, p):
        return max(0.0, 1 - np.mean(np.abs((a-p)/a))) * 100

    model_acc = acc(actual, model_pred)
    naive_acc = acc(actual, naive_pred)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(w), y=list(demand), mode="lines", name="Synthetic weekly demand",
                               line=dict(color=PURPLE_MID, width=1.5), fill="tozeroy", fillcolor="rgba(124,92,184,0.12)"))
    fig1.update_layout(**base_layout("Generated demand history (train + holdout)"))
    fig1.update_yaxes(title="Units/week")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(w[n_train:]), y=list(actual), mode="lines+markers", name="Actual",
                               line=dict(color=INK, width=2.5)))
    fig2.add_trace(go.Scatter(x=list(w[n_train:]), y=list(naive_pred), mode="lines", name="Naive seasonal baseline",
                               line=dict(color=MUTED, width=1.5, dash="dash")))
    fig2.add_trace(go.Scatter(x=list(w[n_train:]), y=list(model_pred), mode="lines+markers", name="Fitted regression model",
                               line=dict(color=GREEN, width=2)))
    fig2.update_layout(**base_layout("12-week holdout backtest"))
    fig2.update_yaxes(title="Units/week")

    labels = ["Intercept/level", "Trend (weekly growth)", "Seasonal (annual)", "Seasonal (phase)",
              "Seasonal (semi-annual)", "Promotion effect", "Weather sensitivity"]

    return fig1, fig2, model_acc, naive_acc, list(zip(labels, beta))

if section == "🌊1 Demand Forecast":
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("##### Backtest controls")
        st.write("A synthetic 104-week history (trend, seasonality, promos, weather, plus noise the model can't see) "
                 "is generated fresh, and a log-linear regression is actually fit and backtested on the last 12 weeks.")
        regen = st.button("🔁 Regenerate & refit", use_container_width=True, key="regen_demand")
        if regen or "demand_result" not in st.session_state:
            st.session_state.demand_result = run_demand_module()
        fig1, fig2, model_acc, naive_acc, betas = st.session_state.demand_result

        st.metric("Naive seasonal baseline accuracy", f"{naive_acc:.1f}%")
        st.metric("Fitted model accuracy (holdout)", f"{model_acc:.1f}%", delta=f"+{model_acc-naive_acc:.1f} pts")
        with st.expander("Fitted coefficients (log-scale)"):
            for label, b in betas:
                st.write(f"**{label}**: `{b:.3f}`")
        st.caption("Illustrative synthetic backtest — directionally consistent with the roadmap's 62%→78% claim, "
                    "but the exact figure will vary by run since the data is randomly regenerated each time.")
    with c2:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# MODULE 2 — SHOULD-COST MODELLING  (procurement cost drivers, synthetic fallback)
# ===========================================================================
YF_TICKERS = {"cocoa": "CC=F", "dairy": "DC=F", "fx": "INR=X"}
YF_NAMES = {"cocoa": "cocoa price (ICE cocoa futures, CC=F)", "dairy": "milk price, used as a dairy stand-in (CME Class III milk futures, DC=F)", "fx": "dollar-rupee exchange rate (INR=X)"}

def gen_price_series(seed, days, start, mean_level, revert_speed, vol):
    """Synthetic fallback: mean-reverting random walk, used only if Yahoo Finance is unreachable."""
    rng = np.random.default_rng(seed)
    p = np.empty(days)
    p[0] = start
    for d in range(1, days):
        shock = rng.normal(0, vol)
        p[d] = max(1.0, p[d-1] + revert_speed*(mean_level - p[d-1]) + p[d-1]*shock)
    return p

@st.cache_data(ttl=900, show_spinner="Refreshing procurement cost drivers for cocoa, dairy and the exchange rate…")
def fetch_yahoo_prices(nonce, period="9mo"):
    """Fetch real daily closes for cocoa, milk (dairy proxy) and USD/INR from Yahoo Finance.
    Raises on failure so the caller can fall back to synthetic data."""
    tickers = list(YF_TICKERS.values())
    data = yf.download(tickers=tickers, period=period, interval="1d",
                        group_by="ticker", auto_adjust=True, progress=False, threads=True)
    closes = {}
    for key, tkr in YF_TICKERS.items():
        try:
            series = data[tkr]["Close"].dropna()
        except (KeyError, TypeError):
            series = data["Close"].dropna() if "Close" in data else pd.Series(dtype=float)
        if series.empty:
            raise RuntimeError(f"No data returned for {tkr}")
        closes[key] = series
    # align on common trading dates, keep last 180 rows
    df = pd.DataFrame(closes).dropna()
    if len(df) < 40:
        raise RuntimeError("Not enough overlapping trading history returned")
    return df.tail(180)

def compute_signal(idx):
    recent = idx[-30:].values
    xs = np.arange(len(recent))
    slope = np.polyfit(xs, recent, 1)[0]
    base90 = idx[-90:].values
    sd = base90.std() or 1.0
    z = (recent[-1] - base90.mean()) / sd
    if slope < -0.03 and z < -0.3:
        signal, cls = "BUY NOW", "signal-buy"
        rationale = ("Cocoa, dairy and currency costs have been falling over the last 30 days and are cheaper "
                     "than their recent 3-month average — this is a good time to place orders.")
    elif slope > 0.05 and z > 0.3:
        signal, cls = "WAIT", "signal-wait"
        rationale = ("Cocoa, dairy and currency costs have been rising over the last 30 days and are more "
                     "expensive than their recent 3-month average — delay any purchases you can, and lock in "
                     "a price now for what you can't delay.")
    else:
        signal, cls = "NEUTRAL", "signal-neutral"
        rationale = "Costs haven't moved much either way in the last 30 days — buy on your normal schedule."
    return slope, z, signal, cls, rationale

def run_cost_module(nonce):
    w = dict(cocoa=0.5, dairy=0.3, fx=0.2)
    source_note = ""
    is_live = False

    if YFINANCE_AVAILABLE:
        try:
            df = fetch_yahoo_prices(nonce)
            cocoa, dairy, fx = df["cocoa"], df["dairy"], df["fx"]
            is_live = True
            source_note = f"Live Yahoo Finance data — {df.index[0].date()} to {df.index[-1].date()} ({len(df)} trading days)."
        except Exception as e:
            is_live = False
            source_note = f"Yahoo Finance fetch failed ({e}) — showing synthetic fallback data instead."

    if not is_live:
        days = 180
        rng = np.random.default_rng()
        s1, s2, s3 = rng.integers(0, 1_000_000_000, 3)
        idx_range = pd.RangeIndex(days)
        cocoa = pd.Series(gen_price_series(int(s1), days, 100, 100+8*(rng.random()-0.3), 0.02, 0.018), index=idx_range)
        dairy = pd.Series(gen_price_series(int(s2), days, 100, 100+3*(rng.random()-0.3), 0.03, 0.010), index=idx_range)
        fx    = pd.Series(gen_price_series(int(s3), days, 83.0, 83.0+1.2*(rng.random()-0.3), 0.015, 0.006), index=idx_range)
        if not YFINANCE_AVAILABLE:
            source_note = "yfinance not installed — showing synthetic fallback data. Run `pip install yfinance` for live prices."

    # rebase every series to 100 at the start of the window so the blend is comparable regardless of native units
    cocoa_r = cocoa / cocoa.iloc[0] * 100
    dairy_r = dairy / dairy.iloc[0] * 100
    fx_r    = fx / fx.iloc[0] * 100
    idx = w["cocoa"]*cocoa_r + w["dairy"]*dairy_r + w["fx"]*fx_r

    slope, z, signal, cls, rationale = compute_signal(idx)
    current = idx.iloc[-1]
    trailing_avg = idx.iloc[-60:-30].mean() if len(idx) >= 60 else idx.iloc[:len(idx)//2].mean()
    delta_pct = (current - trailing_avg) / trailing_avg * 100

    x_axis = list(range(len(idx)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=list(idx.values), name="Combined raw-material cost", line=dict(color=PURPLE, width=2.5)))
    fig.add_trace(go.Scatter(x=x_axis, y=list(cocoa_r.values), name="Cocoa (rebased)", line=dict(color="#8a5a3a", width=1), visible="legendonly"))
    fig.add_trace(go.Scatter(x=x_axis, y=list(dairy_r.values), name="Milk/dairy (rebased)", line=dict(color="#5aa0c9", width=1), visible="legendonly"))
    fig.add_trace(go.Scatter(x=x_axis, y=list(fx_r.values), name="Dollar-rupee rate (rebased)", line=dict(color=PURPLE, width=1), visible="legendonly"))
    title = ("Raw-material cost trend — live data (100 = level at the start of this window)" if is_live
              else "Raw-material cost trend — synthetic data (100 = level at the start of this window)")
    fig.update_layout(**base_layout(title))
    fig.update_yaxes(title="Cost level (100 = start of window)")
    fig.update_xaxes(title="Days")

    return fig, signal, cls, rationale, current, trailing_avg, delta_pct, slope, z, is_live, source_note, idx

def build_negotiation_plan(idx, current, signal):
    """Derive a negotiation brief straight from the 90-day cost distribution — not a fixed script."""
    base90 = idx[-90:]
    mean90, std90 = base90.mean(), (base90.std() or 1.0)
    if signal == "BUY NOW":
        opening_offer, target, walk_away = current - 0.5*std90, current, mean90
        leverage = "You have the stronger hand right now: costs are below their 3-month average and still falling."
        lever = "Push for a fixed-price lock for 60-90 days to bank the dip before prices go back up."
    elif signal == "WAIT":
        opening_offer, target, walk_away = mean90, mean90 + 0.3*std90, current
        leverage = "The supplier has the stronger hand right now: costs are above their 3-month average and rising."
        lever = ("Ask for a price-cap clause tied to the market rate, or split your order: buy half now at a "
                 "locked price and leave half to buy later.")
    else:
        opening_offer, target, walk_away = mean90 - 0.3*std90, mean90, mean90 + std90
        leverage = "Neither side has a strong edge right now — anchor pricing on the 3-month average cost."
        lever = "Run your standard quarterly supplier quote process; hold firm on payment terms."
    headroom_pct = (walk_away - target) / target * 100 if target else 0.0
    return opening_offer, target, walk_away, leverage, lever, headroom_pct

if section == "🌊1 Cost & Negotiation":
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("##### Procurement cost drivers")
        st.write("Uses public cocoa, dairy-proxy and dollar-rupee prices as inputs to a raw-materials cost "
             "estimate. This is not a stock-trading signal: it helps procurement decide when to place orders, "
             "when to lock in prices, and how to run supplier quote reviews. Synthetic reference prices are "
             "used if live prices are unavailable.")
        if "cost_nonce" not in st.session_state:
            st.session_state.cost_nonce = 0
        refresh_cost = st.button("🔁 Refresh cost drivers", use_container_width=True, key="regen_cost")
        if refresh_cost:
            st.session_state.cost_nonce += 1
        result = run_cost_module(st.session_state.cost_nonce)
        fig, signal, cls, rationale, current, trailing_avg, delta_pct, slope, z, is_live, source_note, idx = result
        st.session_state.procurement_cost_signal = signal

        if is_live:
            st.success(source_note, icon="📡")
        else:
            st.warning(source_note, icon="⚠️")

        st.markdown(f'<div class="{cls}">{signal}</div>', unsafe_allow_html=True)
        st.write(rationale)
        mean90 = idx[-90:].mean()
        vs_3mo_pct = (current - mean90) / mean90 * 100 if mean90 else 0.0
        trend_30d_pct = slope * 30 / current * 100 if current else 0.0
        st.metric("Current cost level", f"{current:.1f}")
        st.metric("Cost level 1-2 months ago", f"{trailing_avg:.1f}")
        st.metric("Change vs 1-2 months ago", f"{delta_pct:+.1f}%")
        st.metric("Cost trend over the last 30 days", f"{trend_30d_pct:+.1f}%")
        st.metric("Vs the last 3 months' average", f"{vs_3mo_pct:+.1f}%")
        st.caption("Weights: cocoa 50% · dairy (milk proxy) 30% · currency (USD/INR) 20%, reflecting a "
                   "chocolate-heavy materials mix. Data sources: " + ", ".join(YF_NAMES.values()) + ".")

        st.markdown("##### 🤝 Negotiation brief")
        opening_offer, target, walk_away, leverage, lever, headroom_pct = build_negotiation_plan(idx, current, signal)
        st.write(leverage)
        n1, n2, n3 = st.columns(3)
        n1.metric("Opening offer", f"{(opening_offer-current)/current*100:+.1f}% vs today")
        n2.metric("Target settlement", f"{(target-current)/current*100:+.1f}% vs today")
        n3.metric("Walk-away ceiling", f"{(walk_away-current)/current*100:+.1f}% vs today")
        st.metric("Negotiation headroom", f"{headroom_pct:.1f}%",
                   help="Gap between the walk-away ceiling and the target settlement, as a % of the target.")
        st.info(f"**Recommended move:** {lever}", icon="📝")
        st.caption("Opening offer, target and walk-away are calculated from the last 3 months of cost data shown "
                   "above — not a fixed script.")
    with c2:
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# MODULE 3 — SUPPLIER RISK AGENT
# ===========================================================================
def generate_supplier_risk(seed):
    rng = np.random.default_rng(seed)
    suppliers = [
        ("Cacao Bharat", "Cocoa", "Primary", "Ghana", "CocoaLink India"),
        ("CocoaLink India", "Cocoa", "Approved alternate", "Cote d'Ivoire", "Cacao Bharat"),
        ("DairyFresh Co-op", "Milk solids", "Primary", "Maharashtra", "Milko Foods"),
        ("Milko Foods", "Milk solids", "Approved alternate", "Gujarat", "DairyFresh Co-op"),
        ("PackRight Films", "Packaging film", "Primary", "Tamil Nadu", "FlexiPack"),
        ("FlexiPack", "Packaging film", "Approved alternate", "Karnataka", "PackRight Films"),
    ]
    rows = []
    for index, (supplier, material, role, location, alternate) in enumerate(suppliers):
        delivery = int(rng.integers(3, 38))
        quality = int(rng.integers(1, 24))
        financial = int(rng.integers(2, 28))
        geo = int(rng.integers(1, 30))
        capacity = int(rng.integers(3, 25))
        if index == (seed % 3) * 2:
            delivery = 85
            geo = 70
        risk = round(0.35 * delivery + 0.20 * quality + 0.15 * financial + 0.15 * geo + 0.15 * capacity)
        rows.append({
            "Supplier": supplier,
            "Material": material,
            "Role": role,
            "Location": location,
            "Late delivery": delivery,
            "Quality": quality,
            "Financial": financial,
            "Geopolitical": geo,
            "Capacity": capacity,
            "Risk score": risk,
            "Approved alternate": alternate,
        })
    return pd.DataFrame(rows)

def supplier_actions(risk_df, cost_signal):
    actions = []
    primaries = risk_df[risk_df["Role"] == "Primary"].sort_values("Risk score", ascending=False)
    for _, supplier in primaries.iterrows():
        if supplier["Risk score"] >= 35:
            actions.append({
                "Priority": "HIGH" if supplier["Risk score"] >= 45 else "MEDIUM",
                "Material": supplier["Material"],
                "Decision": f"Qualify {supplier['Approved alternate']} for 30% of next receipts",
                "Reason": f"{supplier['Supplier']} risk score {supplier['Risk score']}/100",
            })
    buying_action = {
        "BUY NOW": "Move up the next planned raw-material order by two weeks",
        "WAIT": "Delay non-urgent orders; ask suppliers for current quotes and lock in prices where possible",
        "NEUTRAL": "Run a normal competitive quote process on the usual buying schedule",
    }[cost_signal]
    actions.append({
        "Priority": "MEDIUM", "Material": "Cocoa / dairy / currency",
        "Decision": buying_action, "Reason": f"Cost-outlook recommendation: {cost_signal}",
    })
    return actions

# ===========================================================================
# MODULE 3 — CONTROL TOWER + AGENT
# ===========================================================================

def compute_meio_safety_stock(whs, seed, service_z=1.645, demand_cv=0.25):
    """Stochastic MEIO: safety stock per node from its own demand variability and lead time,
    vs a naive flat 5-day-of-cover safety stock applied network-wide."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in whs.values():
        lead_time = float(rng.uniform(4, 10))
        sigma_daily = w["demand"] * demand_cv
        meio_ss = service_z * sigma_daily * np.sqrt(lead_time)
        naive_ss = w["demand"] * 5.0
        rows.append({
            "Node": w["label"], "Lead time (d)": round(lead_time, 1),
            "Demand std/day": round(sigma_daily, 1),
            "MEIO safety stock": round(meio_ss),
            "Naive fixed safety stock": round(naive_ss),
            "Reduction vs naive": (naive_ss - meio_ss) / naive_ss * 100 if naive_ss else 0.0,
        })
    return pd.DataFrame(rows)

if section == "🌊1 Warehouse Stock Levels":
    st.write("Sets each node's statistically-derived safety stock from its own demand variability and lead time, "
             "instead of one flat days-of-cover target for every node.")
    if "meio_seed" not in st.session_state:
        st.session_state.meio_seed = int(np.random.randint(0, 1_000_000_000))
    resample_meio = st.button("🔁 Resample lead times & demand variability", use_container_width=True, key="meio_resample")
    if resample_meio:
        st.session_state.meio_seed = int(np.random.randint(0, 1_000_000_000))
    meio_df = compute_meio_safety_stock(st.session_state.warehouses, st.session_state.meio_seed)
    st.dataframe(meio_df, hide_index=True, use_container_width=True)
    st.metric("Network-wide inventory reduction vs flat safety stock", f"{meio_df['Reduction vs naive'].mean():+.1f}%")
    st.caption("Safety stock is a buffer sized to each node's own demand swings and delivery lead time (at a "
               "95% service level), instead of one flat number for every warehouse. Illustrative synthetic lead "
               "times and demand "
               "variability — directionally consistent with the roadmap's -18% inventory claim, exact figure varies by run.")

    st.markdown("##### Live control tower & rebalancing agent")
    st.write("A live 4-warehouse network drains against synthetic daily demand. When a node's days-of-cover falls "
             "below target, the tower raises an alert — then an agent computes an actual rebalancing plan "
             "(equalising days-of-cover network-wide) and applies it.")

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Alerts raised", st.session_state.alerts_raised)
    kc2.metric("Resolved by agent", st.session_state.alerts_resolved)
    ttm = st.session_state.ttm_samples
    kc3.metric("Avg. time-to-mitigate", f"{np.mean(ttm):.1f} sim-d" if ttm else "—")
    whs = st.session_state.warehouses
    net_cover = sum(w["stock"] for w in whs.values()) / sum(w["demand"] for w in whs.values())
    kc4.metric("Network avg. days of cover", f"{net_cover:.1f} d")

    b1, b2, b3, b4 = st.columns(4)
    step = b1.button("⏭️ Step (1 tick)", use_container_width=True)
    autorun = b2.button("▶️ Auto-run 12 ticks", use_container_width=True)
    shock = b3.button("⚡ Inject shock — WH-W", use_container_width=True)
    reset = b4.button("♻️ Reset network", use_container_width=True)

    if reset:
        reset_network()
        st.rerun()
    if shock:
        st.session_state.warehouses["WH-W"]["stock"] = max(
            0, st.session_state.warehouses["WH-W"]["stock"] - st.session_state.warehouses["WH-W"]["demand"]*3.5)
        st.session_state.log.insert(0, ("info", "Manual demand shock injected at WH-W (Mumbai) — e.g. a regional promo spike."))
        st.rerun()
    if step:
        sim_tick()
        st.rerun()

    wh_cols = st.columns(4)
    for col, (k, w) in zip(wh_cols, st.session_state.warehouses.items()):
        cover = days_of_cover(w)
        alert = cover < 5
        with col:
            st.markdown(f"""
            <div class="wh-card {'wh-alert' if alert else 'wh-ok'}">
                <div style="font-size:13px;font-weight:600;" class="wh-sub">{w['label']}</div>
                <div style="font-family:monospace;font-size:22px;font-weight:700;" class="wh-sub">{cover:.1f} <span style="font-size:12px;" class="wh-sub">d</span></div>
                <div style="font-size:11px;" class="wh-sub">days of cover</div>
                <div style="font-size:11px;font-family:monospace;margin-top:8px;" class="wh-sub">{w['stock']:.0f} units · demand {w['demand']:.0f}/day</div>
            </div>
            """, unsafe_allow_html=True)

    chart_ph = st.empty()
    def render_tower_chart():
        fig = go.Figure()
        colors = {"WH-N": PURPLE_MID, "WH-W": PURPLE_DEEP, "WH-S": PURPLE_SOFT, "WH-E": PURPLE}
        for k, series in st.session_state.history.items():
            if series:
                fig.add_trace(go.Scatter(y=series, name=st.session_state.warehouses[k]["label"],
                                          line=dict(color=colors[k], width=2)))
        fig.update_layout(**base_layout("Days of cover by warehouse (live)"))
        fig.update_yaxes(title="Days of cover", rangemode="tozero")
        fig.update_xaxes(visible=False)
        chart_ph.plotly_chart(fig, use_container_width=True, key=f"tower_{st.session_state.tick}_{len(st.session_state.log)}")
    render_tower_chart()

    st.markdown("##### Event log")
    log_ph = st.empty()
    def render_log():
        lines = []
        for kind, msg in st.session_state.log[:60]:
            cls = {"alert": "log-alert", "resolve": "log-resolve", "info": "log-info"}[kind]
            lines.append(f'<div class="log-line {cls}">{msg}</div>')
        log_ph.markdown(f'<div class="log-box">{"".join(lines)}</div>', unsafe_allow_html=True)
    render_log()

    if autorun:
        for _ in range(12):
            sim_tick()
            render_tower_chart()
            render_log()
            time.sleep(0.5)
        st.rerun()

if section == "🌊2 Supplier Risk":
    if "supplier_risk_seed" not in st.session_state:
        st.session_state.supplier_risk_seed = int(np.random.randint(0, 1_000_000_000))
    if "procurement_execution" not in st.session_state:
        st.session_state.procurement_execution = []

    risk_df = generate_supplier_risk(st.session_state.supplier_risk_seed)
    cost_signal = st.session_state.get("procurement_cost_signal", "NEUTRAL")
    actions = supplier_actions(risk_df, cost_signal)
    primary_risks = risk_df[risk_df["Role"] == "Primary"]
    high_risks = primary_risks[primary_risks["Risk score"] >= 35]

    st.write("The procurement agent combines synthetic supplier performance, quality, financial, capacity and "
             "geopolitical signals with the raw-material cost-outlook recommendation. It produces only actions "
             "that use a pre-approved alternate or an established sourcing process.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary suppliers monitored", len(primary_risks))
    c2.metric("Suppliers needing mitigation", len(high_risks))
    c3.metric("Current buying guidance", cost_signal)

    refresh_risk = st.button("🔁 Generate new supplier scenario", use_container_width=True, key="refresh_risk")
    if refresh_risk:
        st.session_state.supplier_risk_seed = int(np.random.randint(0, 1_000_000_000))
        st.rerun()

    st.markdown("##### 1. Sense: supplier risk signals")
    st.dataframe(
        risk_df[["Supplier", "Material", "Role", "Location", "Late delivery", "Quality", "Financial",
                 "Geopolitical", "Capacity", "Risk score", "Approved alternate"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Risk score combines late-delivery 35%, quality 20%, financial health 15%, geopolitical exposure "
               "15% and capacity pressure 15%. All supplier records are synthetic.")

    st.markdown("##### 2. Reason: procurement mitigation plan")
    plan_df = pd.DataFrame(actions)
    st.dataframe(plan_df, hide_index=True, use_container_width=True)

    execute = st.button("▶ Execute approved mitigation plan", type="primary", use_container_width=True)
    if execute:
        for action in actions:
            st.session_state.procurement_execution.insert(
                0, f"{action['Priority']}: {action['Material']} - {action['Decision']}"
            )
        st.session_state.log.insert(
            0, ("resolve", f"<b>Procurement agent executed</b> {len(actions)} approved actions: "
                f"{'; '.join(action['Decision'] for action in actions)}."))
        st.success("Mitigation plan executed and recorded in the Control Tower event log.")

    if st.session_state.procurement_execution:
        st.markdown("##### 3. Act and observe: execution record")
        for entry in st.session_state.procurement_execution[:8]:
            st.markdown(f"- {entry}")

# ===========================================================================
# MODULE 5 — PREDICTIVE MAINTENANCE
# ===========================================================================
def generate_sensor_telemetry(seed, hours=72):
    rng = np.random.default_rng(seed)
    lines = ["Line 1", "Line 2", "Line 3", "Line 4"]
    t = np.arange(hours)
    fault_line = lines[seed % len(lines)]
    data = {}
    for line in lines:
        vib = 2.0 + rng.normal(0, 0.15, hours)  # stationary noise: normal operation has no structural drift
        if line == fault_line:
            ramp = np.clip(t - hours * 0.6, 0, None) * 0.06
            vib = vib + ramp
        data[line] = vib
    return pd.DataFrame(data, index=t), fault_line

def detect_anomalies(df, baseline_hours=36, z_thresh=4.0):
    """Static early-baseline comparison — catches slow wear-drift signatures that a rolling
    window would absorb into its own adapting normal range."""
    flags = {}
    for col in df.columns:
        baseline = df[col].iloc[:baseline_hours]
        mu, sigma = baseline.mean(), (baseline.std() or 1e-6)
        z = (df[col] - mu) / sigma
        flags[col] = z.abs() > z_thresh
    return pd.DataFrame(flags)

if section == "🌊2 Predictive Maintenance":
    st.write("Synthetic vibration telemetry across 4 production lines; a detector flags readings that fall "
             "well outside a line's own normal operating range — a bearing-wear signature — before they cause "
             "unplanned downtime.")
    if "maint_seed" not in st.session_state:
        st.session_state.maint_seed = int(np.random.randint(0, 1_000_000_000))
    resample_maint = st.button("🔁 Resample telemetry", use_container_width=True, key="resample_maint")
    if resample_maint:
        st.session_state.maint_seed = int(np.random.randint(0, 1_000_000_000))
    telemetry, fault_line = generate_sensor_telemetry(st.session_state.maint_seed)
    anomalies = detect_anomalies(telemetry)
    flagged_lines = [col for col in telemetry.columns if anomalies[col].any()]

    line_colors = {"Line 1": PURPLE_MID, "Line 2": PURPLE_DEEP, "Line 3": PURPLE_SOFT, "Line 4": PURPLE}
    fig = go.Figure()
    for col in telemetry.columns:
        fig.add_trace(go.Scatter(y=list(telemetry[col]), name=col, line=dict(color=line_colors[col], width=1.5)))
        flagged_idx = telemetry.index[anomalies[col]]
        if len(flagged_idx):
            fig.add_trace(go.Scatter(x=list(flagged_idx), y=list(telemetry[col].loc[flagged_idx]), mode="markers",
                                      name=f"{col} anomaly", marker=dict(color=RED, size=6, symbol="x"), showlegend=False))
    fig.update_layout(**base_layout("Vibration telemetry (flagged readings outside the normal range)"))
    fig.update_yaxes(title="Vibration index")
    fig.update_xaxes(title="Hours")
    st.plotly_chart(fig, use_container_width=True)

    baseline_oee, improved_oee = 68.0, 73.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Lines flagged for maintenance", len(flagged_lines))
    c2.metric("Baseline OEE (reactive maintenance)", f"{baseline_oee:.0f}%")
    c3.metric("OEE with predictive maintenance", f"{improved_oee:.0f}%", delta=f"+{improved_oee-baseline_oee:.0f} pts")

    schedule_maint = st.button("▶ Schedule predictive maintenance for flagged lines", type="primary",
                                use_container_width=True, disabled=not flagged_lines, key="schedule_maint")
    if schedule_maint:
        st.session_state.log.insert(0, ("resolve",
            f"<b>Predictive maintenance agent</b> scheduled inspection for {', '.join(flagged_lines)} "
            f"after its vibration reading moved well outside its normal range — avoided reactive downtime."))
        st.success(f"Maintenance scheduled for {', '.join(flagged_lines)} and recorded in the shared agent log.")
    st.caption("Rule: flag a line if its vibration reading moves far outside its normal range, compared with its "
               "first 36 hours of operation. Synthetic sensor data; "
               "OEE figures are illustrative, consistent with the roadmap's 68%→73% claim.")

# ===========================================================================
# MODULE 6 — ENERGY SETPOINTS (model-predictive-control-lite)
# ===========================================================================
def optimize_energy_setpoints(seed, min_high_hours=3):
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    tariff = 4.5 + 2.5 * np.sin((hours - 14) / 24 * 2 * np.pi) + rng.normal(0, 0.15, 24)
    tariff = np.clip(tariff, 2.0, None)
    power = {160: 40.0, 175: 55.0, 190: 75.0}

    naive_cost = float(np.sum(tariff * power[190]))

    cheapest_hours = np.argsort(tariff)
    chosen = np.full(24, 160)
    chosen[cheapest_hours[:min_high_hours]] = 190
    remaining = [h for h in range(24) if chosen[h] != 190]
    remaining_sorted = sorted(remaining, key=lambda h: tariff[h])
    for h in remaining_sorted[:len(remaining_sorted) // 2]:
        chosen[h] = 175

    optimized_cost = float(np.sum([tariff[h] * power[chosen[h]] for h in range(24)]))
    savings_pct = (naive_cost - optimized_cost) / naive_cost * 100 if naive_cost else 0.0
    return hours, tariff, chosen, naive_cost, optimized_cost, savings_pct

if section == "🌊2 Energy Scheduling":
    st.write("Schedules oven temperatures around the hourly electricity price: the hours when the oven must run "
             "hottest are placed at the cheapest electricity hours, and medium-heat hours fill the next-cheapest "
             "slots — a schedule that plans ahead, instead of running at the highest setting all day.")
    if "energy_seed" not in st.session_state:
        st.session_state.energy_seed = int(np.random.randint(0, 1_000_000_000))
    resample_energy = st.button("🔁 Resample tariff curve", use_container_width=True, key="resample_energy")
    if resample_energy:
        st.session_state.energy_seed = int(np.random.randint(0, 1_000_000_000))
    hours, tariff, chosen, naive_cost, optimized_cost, savings_pct = optimize_energy_setpoints(st.session_state.energy_seed)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(hours), y=list(tariff), name="Energy tariff (₹/kWh)", yaxis="y2",
                              line=dict(color=RED, width=2, dash="dot")))
    setpoint_colors = {160: GREEN, 175: PURPLE_MID, 190: RED}
    fig.add_trace(go.Bar(x=list(hours), y=list(chosen), name="Oven setpoint (°C)",
                          marker=dict(color=[setpoint_colors[c] for c in chosen])))
    fig.update_layout(**base_layout("Optimized oven setpoint schedule vs energy tariff"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="₹/kWh", showgrid=False))
    fig.update_yaxes(title="Setpoint (°C)")
    fig.update_xaxes(title="Hour of day")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Always-on 190°C baseline cost", f"₹{naive_cost:,.0f}")
    c2.metric("Optimized schedule cost", f"₹{optimized_cost:,.0f}")
    c3.metric("Energy cost saved", f"{savings_pct:.1f}%")

    apply_energy = st.button("▶ Apply optimized setpoint schedule", type="primary", use_container_width=True, key="apply_energy")
    if apply_energy:
        st.session_state.log.insert(0, ("resolve",
            f"<b>Energy scheduling agent</b> applied a price-aware oven schedule — "
            f"{savings_pct:.1f}% lower energy cost vs always-on 190°C."))
        st.success("Setpoint schedule applied and recorded in the shared agent log.")
    st.caption("Synthetic hourly tariff curve; setpoint choice is a greedy cost-minimizing heuristic subject to a "
               "minimum-throughput constraint at the highest setpoint.")

# ===========================================================================
# MODULE 7 — NETWORK DIGITAL TWIN (make/hold LP)
# ===========================================================================
def solve_network_twin(whs, seed):
    rng = np.random.default_rng(seed)
    plants = ["Plant-A (Baddi)", "Plant-B (Hyderabad)"]
    node_labels = [w["label"] for w in whs.values()]
    demands = np.array([w["demand"] for w in whs.values()], dtype=float)
    n_plants, n_wh = 2, len(node_labels)

    capacities = np.array([demands.sum() * 0.65, demands.sum() * 0.65])
    prod_cost = np.array([12.0, 10.5])
    transport_cost = rng.uniform(1.0, 4.0, size=(n_plants, n_wh))
    cost = prod_cost[:, None] + transport_cost

    c = cost.flatten()
    A_eq = np.zeros((n_wh, n_plants * n_wh))
    for j in range(n_wh):
        for i in range(n_plants):
            A_eq[j, i * n_wh + j] = 1
    A_ub = np.zeros((n_plants, n_plants * n_wh))
    for i in range(n_plants):
        A_ub[i, i * n_wh:(i + 1) * n_wh] = 1

    res = linprog(c, A_ub=A_ub, b_ub=capacities, A_eq=A_eq, b_eq=demands, bounds=(0, None), method="highs")
    alloc = res.x.reshape(n_plants, n_wh) if res.success else np.zeros((n_plants, n_wh))
    lp_cost = float(res.fun) if res.success else float("nan")

    naive_alloc = np.tile(demands / n_plants, (n_plants, 1))
    naive_cost = float(np.sum(naive_alloc * cost))
    return plants, node_labels, cost, alloc, lp_cost, naive_cost, bool(res.success)

if section == "🌊2 Factory & Warehouse Planner":
    st.write("A network-planning tool decides how much each plant should make for each warehouse, to get the "
             "lowest total cost of making and shipping — while staying within each plant's capacity and meeting "
             "each warehouse's demand in full.")
    if "twin_seed" not in st.session_state:
        st.session_state.twin_seed = int(np.random.randint(0, 1_000_000_000))
    resample_twin = st.button("🔁 Resample transport cost lanes", use_container_width=True, key="resample_twin")
    if resample_twin:
        st.session_state.twin_seed = int(np.random.randint(0, 1_000_000_000))

    plants, node_labels, cost, alloc, lp_cost, naive_cost, solved = solve_network_twin(
        st.session_state.warehouses, st.session_state.twin_seed)

    if solved:
        alloc_df = pd.DataFrame(alloc, index=plants, columns=node_labels).round(0)
        st.dataframe(alloc_df, use_container_width=True)
        savings_pct = (naive_cost - lp_cost) / naive_cost * 100 if naive_cost else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost if split evenly between plants", f"₹{naive_cost:,.0f}/day")
        c2.metric("Cost with the optimized plan", f"₹{lp_cost:,.0f}/day")
        c3.metric("Cost saved", f"{savings_pct:.1f}%")

        apply_twin = st.button("▶ Apply optimized make/hold plan", type="primary", use_container_width=True, key="apply_twin")
        if apply_twin:
            st.session_state.log.insert(0, ("resolve",
                f"<b>Factory & warehouse planner</b> applied a cost-optimized make/hold plan across {', '.join(plants)} — "
                f"{savings_pct:.1f}% lower landed cost vs an even split."))
            st.success("Make/hold plan applied and recorded in the shared agent log.")
    else:
        st.warning("Could not find a workable allocation for this sample — resample and try again.")
    st.caption("Synthetic plant capacities, production costs and shipping costs; solved exactly using a standard "
               "cost-optimization method.")

# ===========================================================================
# MODULE 8 — ROUTE, LOAD & COLD CHAIN
# ===========================================================================
def generate_routes(seed, n=8):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        chocolate_pct = float(rng.uniform(5, 95))
        ambient_c = float(rng.uniform(18, 42))
        distance_km = float(rng.uniform(80, 900))
        needs_cold_chain = chocolate_pct > 30 and ambient_c > 28
        cold_cost, ambient_cost = 1.8, 1.0
        cost_always_cold = distance_km * cold_cost
        cost_targeted = distance_km * (cold_cost if needs_cold_chain else ambient_cost)
        rows.append({
            "Route": f"R-{i+1}", "Chocolate %": round(chocolate_pct, 1), "Ambient temp (°C)": round(ambient_c, 1),
            "Distance (km)": round(distance_km), "Cold chain needed": needs_cold_chain,
            "Cost if always cold-chain (₹)": round(cost_always_cold), "Targeted cost (₹)": round(cost_targeted),
        })
    return pd.DataFrame(rows)

if section == "🌊3 Delivery & Cold Chain":
    st.write("Cold chain is applied only where chocolate content and ambient temperature actually require it — "
             "not on every route by default.")
    if "route_seed" not in st.session_state:
        st.session_state.route_seed = int(np.random.randint(0, 1_000_000_000))
    resample_routes = st.button("🔁 Resample route mix", use_container_width=True, key="resample_routes")
    if resample_routes:
        st.session_state.route_seed = int(np.random.randint(0, 1_000_000_000))
    routes_df = generate_routes(st.session_state.route_seed)
    st.dataframe(routes_df, hide_index=True, use_container_width=True)

    total_always = routes_df["Cost if always cold-chain (₹)"].sum()
    total_targeted = routes_df["Targeted cost (₹)"].sum()
    savings_pct = (total_always - total_targeted) / total_always * 100 if total_always else 0.0
    flagged_routes = routes_df.loc[routes_df["Cold chain needed"], "Route"].tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("Routes needing cold chain", len(flagged_routes))
    c2.metric("Cost if cold-chain everywhere", f"₹{total_always:,.0f}")
    c3.metric("Cost saved with targeted cold chain", f"{savings_pct:.1f}%")

    dispatch_routes = st.button("▶ Dispatch cold-chain trucks to flagged routes only", type="primary",
                                 use_container_width=True, key="dispatch_routes")
    if dispatch_routes:
        st.session_state.log.insert(0, ("resolve",
            f"<b>Delivery & cold-chain agent</b> dispatched cold-chain trucks to {', '.join(flagged_routes) or 'no routes'} "
            f"— {savings_pct:.1f}% lower logistics cost vs cold-chain-everywhere."))
        st.success("Cold-chain dispatch plan applied and recorded in the shared agent log.")
    st.caption("Rule: cold chain applied where chocolate content > 30% and ambient temperature > 28°C. "
               "Synthetic route data.")

# ===========================================================================
# MODULE 9 — AGENTIC RESOLUTION (unified cross-agent timeline)
# ===========================================================================
if section == "🌊3 Actions Timeline":
    st.write("Every agent above — stock rebalancing, supplier risk, predictive maintenance, energy scheduling, "
             "the factory & warehouse planner and delivery/cold-chain — writes into one shared timeline. Manual "
             "escalation across these functions typically takes ~12 days; these agents draft and execute directly.")

    ttm_samples = st.session_state.get("ttm_samples", [])
    avg_ttm = float(np.mean(ttm_samples)) if ttm_samples else 3.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Manual process baseline", "12 days")
    c2.metric("Time to fix it (avg. simulated)", f"{avg_ttm:.1f} days")
    c3.metric("Time saved", f"{(12 - avg_ttm) / 12 * 100:.0f}%")

    st.markdown("##### Unified agent action timeline")
    shared_log = st.session_state.get("log", [])
    if shared_log:
        lines = []
        for kind, msg in shared_log[:80]:
            cls = {"alert": "log-alert", "resolve": "log-resolve", "info": "log-info"}[kind]
            lines.append(f'<div class="log-line {cls}">{msg}</div>')
        st.markdown(f'<div class="log-box">{"".join(lines)}</div>', unsafe_allow_html=True)
    else:
        st.info("No agent actions yet — trigger one from any menu section above.", icon="ℹ️")
    st.caption("Shared across Warehouse Stock Levels, Supplier Risk, Predictive Maintenance, Energy Scheduling, "
               "the Factory & Warehouse Planner and Delivery & Cold Chain agents.")

# ===========================================================================
# FLAGSHIP — CONNECTED ALERT (cross-functional detection -> action -> guardrail)
# ===========================================================================
if section == "🚨 Connected Alert":
    now = datetime.now()
    detected_at = now.replace(hour=4, minute=12, second=0, microsecond=0)
    if detected_at > now:
        detected_at -= timedelta(days=1)

    st.markdown("### A cocoa butter supplier flags a 9-day delay")
    st.caption(f"Event SETU-ALERT-0842 · generated {now.strftime('%d %b %Y, %H:%M')}")

    siloed = st.toggle("Show the siloed world (before SETU)", key="alert_siloed_toggle")

    if not siloed:
        st.markdown("##### Detection")
        st.markdown(f"""
        <div class="wh-card wh-alert">
            <div style="font-weight:600;">Supplier ASN revision, {detected_at.strftime('%H:%M')} today</div>
            <div style="color:{MUTED};font-size:13px;margin-top:4px;">
                Time to detect: <b style="color:{INK};">4 hrs</b> (was 5 days) — flagged automatically from the
                supplier's Advance Shipment Notice revision feed, not a manual status call.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Impact traced across pillars")
        for tag, text in [
            ("SOURCE", "Cocoa butter \u22129 days"),
            ("PLAN", "Dairy Milk Silk promo at risk, 16 days out"),
            ("MAKE", "West plant Line 3 schedule breaks day 6"),
            ("DELIVER", "4 depots below cover"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid {LINE};">
                <span class="pill">{tag}</span><span style="color:{INK};">{text}</span>
            </div>
            """, unsafe_allow_html=True)
        st.metric("Value at stake", "₹3.8 Cr")

        st.markdown("##### Recommended action")
        st.info("Shift **62%** of Silk volume to the South plant. Pull cocoa butter forward from the alternate "
                "pre-cleared supplier. Re-phase promo cover at 4 depots.", icon="✅")

        st.markdown("##### Alternatives considered and rejected")
        alt1, alt2 = st.columns(2)
        alt1.markdown(f"""<div class="wh-card"><b>Hold and expedite</b><div style="color:{RED};margin-top:4px;">₹5.1 Cr — rejected</div></div>""", unsafe_allow_html=True)
        alt2.markdown(f"""<div class="wh-card"><b>Reduce promo scope</b><div style="color:{RED};margin-top:4px;">₹4.4 Cr margin loss — rejected</div></div>""", unsafe_allow_html=True)

        st.markdown("##### Owner")
        st.write("**Owner:** North-West Supply Planner &nbsp;\u00b7&nbsp; **Escalation:** E2E Supply Chain Lead")

        st.markdown("##### Guardrail")
        st.warning("Value ₹3.8 Cr exceeds the ₹50 L auto-execute threshold. Human approval required.", icon="🛑")

        if "alert_decision" not in st.session_state:
            st.session_state.alert_decision = None
        d1, d2 = st.columns(2)
        approve = d1.button("✅ Approve", type="primary", use_container_width=True, key="approve_connected_alert")
        escalate = d2.button("⬆ Escalate", use_container_width=True, key="escalate_connected_alert")
        if approve:
            st.session_state.alert_decision = ("approved", now)
            st.session_state.log.insert(0, ("resolve",
                f"<b>Connected alert approved</b> by North-West Supply Planner at {now.strftime('%H:%M')} — "
                f"shifted 62% Silk volume to South plant, pulled cocoa butter forward from the pre-cleared "
                f"alternate supplier, re-phased cover at 4 depots. ₹3.8 Cr at stake."))
        if escalate:
            st.session_state.alert_decision = ("escalated", now)
            st.session_state.log.insert(0, ("alert",
                f"<b>Connected alert escalated</b> to E2E Supply Chain Lead at {now.strftime('%H:%M')} — "
                f"₹3.8 Cr decision pending review."))
        if st.session_state.alert_decision:
            decision, ts = st.session_state.alert_decision
            if decision == "approved":
                st.success(f"Approved by North-West Supply Planner at {ts.strftime('%H:%M')} — action executed.", icon="✅")
            else:
                st.info(f"Escalated to E2E Supply Chain Lead at {ts.strftime('%H:%M')} — awaiting review.", icon="⬆")

        st.metric("Time to mitigate", "2.4 days projected", delta="-9.6 days vs 12-day manual baseline")
    else:
        st.markdown("##### Siloed world — same event, no cross-functional visibility")
        for day, text in [
            ("Day 0", "Supplier flags a 9-day delay. Procurement notes it; planning, the West plant and depots are not informed."),
            ("Day 5", "West Line 3 depots stock out \u2014 that is how the issue is first detected."),
            ("Day 9", "Root cause traced back to the original cocoa butter delay."),
            ("Day 12", "Issue resolved. \u20b93.8 Cr already lost \u2014 Silk promo missed at 4 depots."),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid {LINE};">
                <span class="pill">{day}</span><span style="color:{INK};">{text}</span>
            </div>
            """, unsafe_allow_html=True)
        st.metric("Time to resolve (siloed)", "12 days", delta="+9.6 days vs 2.4-day connected resolution", delta_color="inverse")
        st.metric("Value already lost by detection", "₹3.8 Cr")

    st.divider()
    st.markdown("##### Auto-executed \u2014 below guardrail threshold")
    if "auto_load_consolidation_logged" not in st.session_state:
        st.session_state.auto_load_consolidation_logged = True
        st.session_state.auto_load_ts = now - timedelta(minutes=37)
        st.session_state.log.insert(0, ("resolve",
            f"<b>Load consolidation agent</b> auto-executed Route R-4 consolidation at "
            f"{st.session_state.auto_load_ts.strftime('%H:%M')} \u2014 \u20b96 lakh saved, below the \u20b950 L "
            f"auto-execute threshold. No human approval required."))
    st.markdown(f"""
    <div class="wh-card wh-ok">
        <div style="font-weight:600;">Load consolidation \u2014 Route R-4</div>
        <div style="color:{MUTED};font-size:13px;margin-top:4px;">
            \u20b96 lakh saved \u00b7 below the \u20b950 L auto-execute threshold \u00b7 actioned and logged at
            {st.session_state.auto_load_ts.strftime('%H:%M')} today \u2014 no human touched it.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.caption("Team Alucard · Maestros 2026 — all data on this page is synthetically generated at runtime. "
           "No real Mondelez operational data is used or required.")
