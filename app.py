"""SETU - connected supply-chain response workspace."""

from datetime import datetime, timedelta

import streamlit as st

st.set_page_config(page_title="SETU | Connected supply response", page_icon="S", layout="wide")

PURPLE = "#4B2E83"
PURPLE_DARK = "#29154E"
PURPLE_TEXT = "#34205E"
PURPLE_MUTED = "#6E5C90"
PURPLE_LINE = "#E4DEF2"
WHITE = "#FFFFFF"
OFF_WHITE = "#FAF9FE"

st.markdown(f"""
<style>
.stApp {{ background: {OFF_WHITE}; color: {PURPLE_TEXT}; }}
h1, h2, h3, h4, p, label, .stMarkdown {{ color: {PURPLE_TEXT}; }}
[data-testid="stHeader"] {{ background: {OFF_WHITE}; }}
[data-testid="stMetric"] {{ background: {WHITE}; border: 1px solid {PURPLE_LINE}; border-radius: 6px; padding: 12px; }}
[data-testid="stMetricLabel"] {{ color: {PURPLE_MUTED}; }}
[data-testid="stMetricValue"] {{ color: {PURPLE_TEXT}; }}
.stButton button {{ background: {PURPLE}; color: {WHITE}; border: 0; border-radius: 6px; min-height: 42px; font-weight: 650; }}
.stButton button:hover {{ background: {PURPLE_DARK}; color: {WHITE}; }}
.stButton button p, .stButton button span, .stButton button div {{ color: {WHITE} !important; }}
.eyebrow {{ color: {PURPLE_MUTED}; font-size: .78rem; font-weight: 700; letter-spacing: .08em; }}
.system-note, .small {{ color: {PURPLE_MUTED}; font-size: .9rem; }}
.card {{ background: {WHITE}; border: 1px solid {PURPLE_LINE}; border-radius: 6px; padding: 16px; height: 100%; }}
.impact {{ display: grid; grid-template-columns: 80px 1fr; gap: 12px; align-items: center; padding: 11px 0; border-bottom: 1px solid {PURPLE_LINE}; }}
.impact:last-child {{ border-bottom: 0; }}
.tag {{ background: #F0ECF8; border: 1px solid {PURPLE_LINE}; color: {PURPLE_TEXT}; border-radius: 4px; padding: 3px 7px; font: 700 .72rem monospace; text-align: center; }}
.decision {{ background: {PURPLE_DARK}; color: {WHITE}; border-radius: 6px; padding: 16px; }}
.decision * {{ color: {WHITE} !important; }}
.flow-step {{ border-left: 3px solid {PURPLE}; padding: 10px 14px; margin: 8px 0; background: {WHITE}; border-radius: 0 6px 6px 0; }}
.timeline {{ background: {PURPLE_DARK}; border-radius: 6px; padding: 12px 16px; }}
.event {{ color: #F2EEFA; padding: 9px 0 9px 12px; border-left: 2px solid #B9A8D6; font-size: .9rem; }}
.event.done {{ border-left-color: #7FD1A8; }}
.event.alert {{ border-left-color: #F1B3AE; }}
</style>
""", unsafe_allow_html=True)

if "response_state" not in st.session_state:
    st.session_state.response_state = "waiting"
if "show_siloed" not in st.session_state:
    st.session_state.show_siloed = False
if "events" not in st.session_state:
    st.session_state.events = []

now = datetime.now().replace(second=0, microsecond=0)
detected_at = now.replace(hour=4, minute=12)
if detected_at > now:
    detected_at -= timedelta(days=1)
if not st.session_state.events:
    st.session_state.events.append(("done", now - timedelta(minutes=37), "Load consolidation completed on Route R-4. ₹6 lakh saved; this was below the ₹50 L limit, so SETU completed it automatically."))

st.markdown("<div class='eyebrow'>SETU CONTROL TOWER · ONE CONNECTED SUPPLY-CHAIN SYSTEM</div>", unsafe_allow_html=True)
st.title("One disruption. One coordinated response.")
st.markdown("<div class='system-note'>SETU connects supplier, planning, factory and depot signals so people can see the full impact and make one clear decision.</div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current alert", "Cocoa butter delayed")
m2.metric("Impact", "₹3.8 Cr at risk")
m3.metric("Response time", "2.4 days")
m4.metric("Automatic actions today", "1")
st.divider()

with st.expander("How this screen works", expanded=False):
    st.write("Read from top to bottom. SETU detects a change, connects the impact across the business, compares options, checks whether a person must approve, and records what happened. There is nothing to configure.")

left, right = st.columns([1.45, 1])
with left:
    st.markdown("### 1. What SETU detected")
    st.markdown(f"""<div class="card"><b>Supplier delivery update, {detected_at.strftime('%H:%M')} today</b><br><span class="small">Cocoa butter supplier changed its promised delivery date to 9 days later.</span><br><br><span class="small">Detected from: supplier delivery update · Time to detect: <b>4 hours</b> (previously 5 days)</span></div>""", unsafe_allow_html=True)

    st.markdown("### 2. What this changes across the business")
    impacts = [
        ("SOURCE", "Cocoa butter arrives 9 days late."),
        ("PLAN", "Dairy Milk Silk promotion is at risk in 16 days."),
        ("MAKE", "West plant Line 3 cannot keep its planned schedule from day 6."),
        ("DELIVER", "Four depots would fall below their minimum stock level."),
    ]
    impact_html = "".join(f"<div class='impact'><span class='tag'>{tag}</span><span>{message}</span></div>" for tag, message in impacts)
    st.markdown(f"<div class='card'>{impact_html}</div>", unsafe_allow_html=True)

    st.markdown("### 3. SETU's recommended response")
    st.markdown("""<div class="decision"><b>Keep the promotion running and protect depot stock.</b><br><br>1. Move <b>62%</b> of Dairy Milk Silk production from West to South plant.<br>2. Pull cocoa butter forward from the pre-approved alternate supplier.<br>3. Send extra promotional stock to the 4 affected depots first.</div>""", unsafe_allow_html=True)

with right:
    st.markdown("### Why this is the best option")
    st.markdown("<div class='card'><b>Options SETU considered</b><br><br><b>Recommended: coordinated response</b><br><span class='small'>₹3.8 Cr protected · promotion continues · 2.4 days to complete</span><hr><b>Not chosen: hold and expedite</b><br><span class='small'>₹5.1 Cr cost · too expensive</span><hr><b>Not chosen: reduce promotion scope</b><br><span class='small'>₹4.4 Cr margin loss · loses sales</span></div>", unsafe_allow_html=True)
    st.markdown("### Who decides")
    st.markdown("<div class='card'><b>Owner</b><br>North-West Supply Planner<br><br><b>Escalation</b><br>E2E Supply Chain Lead<br><br><b>Decision rule</b><br><span class='small'>SETU may act on items below ₹50 L. This ₹3.8 Cr decision needs a human approval.</span></div>", unsafe_allow_html=True)

st.markdown("### 4. Approve, escalate, or compare with the old way")
a1, a2, a3 = st.columns([1, 1, 1.3])
with a1:
    if st.button("Approve recommended response", type="primary", use_container_width=True):
        st.session_state.response_state = "approved"
        st.session_state.events.insert(0, ("done", now, "North-West Supply Planner approved the coordinated response. South plant capacity reserved, alternate cocoa butter requested, and depot cover re-phased."))
with a2:
    if st.button("Escalate for review", use_container_width=True):
        st.session_state.response_state = "escalated"
        st.session_state.events.insert(0, ("alert", now, "Decision sent to the E2E Supply Chain Lead for review. The recommended response remains ready to approve."))
with a3:
    st.session_state.show_siloed = st.toggle("Compare with the siloed world", value=st.session_state.show_siloed)

if st.session_state.response_state == "approved":
    st.success("Response approved and actions started. SETU will keep watching supplier delivery, plant capacity and depot stock until the risk is closed.")
elif st.session_state.response_state == "escalated":
    st.warning("Sent to the E2E Supply Chain Lead. No high-value action will be taken until approval is received.")

if st.session_state.show_siloed:
    st.markdown("### The same event without SETU")
    st.markdown("<div class='card'><b>Day 0:</b> Procurement sees the supplier delay, but planning, factory and depot teams do not.<br><br><b>Day 5:</b> A depot stockout reveals the problem.<br><br><b>Day 9:</b> Teams trace the stockout back to the delayed cocoa butter.<br><br><b>Day 12:</b> The issue is finally resolved; ₹3.8 Cr has already been lost.</div>", unsafe_allow_html=True)
else:
    st.markdown("### 5. Autonomous flow: what SETU did and what it is waiting for")
    for line in [
        "Detected · Supplier delivery update changed.",
        "Connected the impact · Checked promotion plan, West plant schedule and depot stock together.",
        "Compared choices · Selected the lowest-cost response that keeps the promotion running.",
        "Applied the safety rule · ₹3.8 Cr is above the ₹50 L limit, so SETU is waiting for a person.",
        "Acts automatically when safe · Route R-4 was already consolidated because it saved ₹6 lakh, below the approval limit.",
    ]:
        st.markdown(f"<div class='flow-step'>{line}</div>", unsafe_allow_html=True)

st.markdown("### Activity log")
event_html = "".join(f"<div class='event {kind}'><b>{when.strftime('%H:%M')}</b> · {message}</div>" for kind, when, message in st.session_state.events[:8])
st.markdown(f"<div class='timeline'>{event_html}</div>", unsafe_allow_html=True)
st.caption("Demo data only. The screen uses named products, sites and roles to demonstrate a connected end-to-end supply-chain response.")
