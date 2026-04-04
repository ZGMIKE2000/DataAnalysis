"""
IMC Prosperity Market Replay — Desktop Research Tool
Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import (
    discover_data_dirs,
    load_round_data,
    build_instrument_index,
    get_instruments,
    get_step_count,
    get_orderbook_at_step,
    get_trades_at_step,
    get_price_context_by_step,
    get_trades_in_step_range,
    format_step_label,
    compute_imbalance,
    build_ofi_series,
    run_event_study,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prosperity Market Replay",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    .block-container {padding-top: 2.5rem; padding-bottom: 0rem;}
    div[data-testid="stMetric"] {background: #1e1e2f; border-radius: 8px; padding: 12px 16px;}
    /* Hide the default Streamlit colored header bar / toolbar decoration */
    header[data-testid="stHeader"] {background: transparent !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    </style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Order book ladder HTML builder
# ---------------------------------------------------------------------------
def build_ob_ladder_html(ob: dict) -> str:
    """Build a full self-contained HTML page for the order book ladder."""
    bids = ob["bids"]  # [(price, vol), ...] best first
    asks = ob["asks"]  # [(price, vol), ...] best first

    if not bids and not asks:
        return '<div style="color:#6b7280;font-family:sans-serif;padding:20px;">No order book data</div>'

    all_vols = [v for _, v in bids] + [v for _, v in asks]
    max_vol = max(all_vols) if all_vols else 1

    rows_html = ""

    # Asks: highest price first, best ask at bottom (closest to spread)
    asks_desc = sorted(asks, key=lambda x: x[0], reverse=True)
    for i, (price, vol) in enumerate(asks_desc):
        bar_pct = int(vol / max_vol * 100) if max_vol > 0 else 0
        rows_html += f"""<tr class="ask-row">
            <td class="vol-cell"></td>
            <td class="price-cell">{price:,.1f}</td>
            <td class="vol-cell">
                <div class="bar-container"><div class="bar ask-bar" style="width:{bar_pct}%"></div></div>
                <span class="vol-text ask-vol">{vol:,}</span>
            </td>
        </tr>\n"""
        if i < len(asks_desc) - 1:
            gap = abs(asks_desc[i][0] - asks_desc[i + 1][0])
            if gap > 0:
                rows_html += f'<tr class="gap-row"><td></td><td class="gap-text">↕ {gap:,.1f}</td><td></td></tr>\n'

    # Spread
    if bids and asks:
        spread = asks[0][0] - bids[0][0]
        rows_html += f'<tr class="spread-row"><td></td><td class="spread-text">SPREAD  {spread:,.1f}</td><td></td></tr>\n'

    # Bids: highest (best) first
    bids_desc = sorted(bids, key=lambda x: x[0], reverse=True)
    for i, (price, vol) in enumerate(bids_desc):
        bar_pct = int(vol / max_vol * 100) if max_vol > 0 else 0
        rows_html += f"""<tr class="bid-row">
            <td class="vol-cell">
                <div class="bar-container"><div class="bar bid-bar" style="width:{bar_pct}%"></div></div>
                <span class="vol-text bid-vol">{vol:,}</span>
            </td>
            <td class="price-cell">{price:,.1f}</td>
            <td class="vol-cell"></td>
        </tr>\n"""
        if i < len(bids_desc) - 1:
            gap = abs(bids_desc[i][0] - bids_desc[i + 1][0])
            if gap > 0:
                rows_html += f'<tr class="gap-row"><td></td><td class="gap-text">↕ {gap:,.1f}</td><td></td></tr>\n'

    n_rows = len(asks) + len(bids) + (len(asks) - 1) + (len(bids) - 1) + (1 if bids and asks else 0)

    return f"""<!DOCTYPE html>
<html><head><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background: transparent; font-family: 'SF Mono','Fira Code','Consolas','Courier New',monospace; }}
table {{ width:100%; border-collapse:collapse; }}
th {{
    padding: 8px 12px;
    text-align: center;
    color: #9ca3af;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid #374151;
}}
td {{ padding: 6px 12px; text-align: center; vertical-align: middle; }}
.ask-row td {{ background: rgba(239, 68, 68, 0.08); }}
.bid-row td {{ background: rgba(34, 197, 94, 0.08); }}
.gap-row td {{ padding: 1px 12px; }}
.gap-text {{ color: #4b5563; font-size: 10px; }}
.spread-row td {{
    padding: 6px 12px;
    border-top: 1px solid #374151;
    border-bottom: 1px solid #374151;
    background: rgba(251, 191, 36, 0.06);
}}
.spread-text {{ color: #fbbf24; font-weight: 700; font-size: 12px; letter-spacing: 1px; }}
.price-cell {{ color: #e5e7eb; font-weight: 500; font-size: 14px; width: 40%; }}
.vol-cell {{ width: 30%; position: relative; }}
.vol-text {{ position: relative; z-index: 1; font-weight: 600; font-size: 14px; }}
.bid-vol {{ color: #4ade80; }}
.ask-vol {{ color: #f87171; }}
.bar-container {{ position: absolute; top: 2px; bottom: 2px; left: 0; right: 0; }}
.bar {{ height: 100%; border-radius: 2px; }}
.bid-bar {{ background: rgba(34, 197, 94, 0.18); float: right; }}
.ask-bar {{ background: rgba(239, 68, 68, 0.18); float: left; }}
</style></head><body>
<table>
    <thead><tr>
        <th style="width:30%">Bid Vol</th>
        <th style="width:40%">Price</th>
        <th style="width:30%">Ask Vol</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
</table>
</body></html>"""


def compute_ob_height(ob: dict) -> int:
    """Estimate iframe height for the order book."""
    n_levels = len(ob["bids"]) + len(ob["asks"])
    n_gaps = max(0, len(ob["bids"]) - 1) + max(0, len(ob["asks"]) - 1)
    has_spread = 1 if ob["bids"] and ob["asks"] else 0
    # header + levels*row_height + gaps*gap_height + spread + padding
    return 38 + n_levels * 34 + n_gaps * 16 + has_spread * 34 + 10


# ---------------------------------------------------------------------------
# Trades table HTML builder
# ---------------------------------------------------------------------------
def build_trades_html(current_trades: pd.DataFrame) -> str:
    """Build self-contained HTML for the trades table."""
    if current_trades.empty:
        return '<div style="color:#6b7280;font-family:sans-serif;font-size:13px;padding:8px;">No trades at this timestamp.</div>'

    has_parties = "buyer" in current_trades.columns

    header = "<tr><th>Price</th><th>Qty</th>"
    if has_parties:
        header += "<th>Buyer</th><th>Seller</th>"
    header += "</tr>"

    rows = ""
    for _, r in current_trades.iterrows():
        price = r.get("price", "")
        qty = r.get("quantity", "")
        price_str = f"{float(price):,.1f}" if pd.notna(price) else "—"
        qty_str = f"{int(qty):,}" if pd.notna(qty) else "—"
        rows += f'<tr><td class="t-price">{price_str}</td><td class="t-qty">{qty_str}</td>'
        if has_parties:
            buyer = r.get("buyer", "")
            seller = r.get("seller", "")
            buyer_str = str(buyer) if pd.notna(buyer) and buyer != "" else ""
            seller_str = str(seller) if pd.notna(seller) and seller != "" else ""
            rows += f'<td class="t-buyer">{buyer_str}</td><td class="t-seller">{seller_str}</td>'
        rows += "</tr>\n"

    return f"""<!DOCTYPE html>
<html><head><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background: transparent; font-family: 'SF Mono','Fira Code','Consolas','Courier New',monospace; }}
table {{ width:100%; border-collapse:collapse; }}
th {{
    padding: 6px 10px;
    text-align: left;
    color: #9ca3af;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #374151;
}}
td {{ padding: 5px 10px; color: #e5e7eb; font-size: 13px; }}
tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
.t-price {{ color: #fbbf24; font-weight: 600; }}
.t-qty {{ color: #93c5fd; }}
.t-buyer {{ color: #4ade80; }}
.t-seller {{ color: #f87171; }}
</style></head><body>
<table>
    <thead>{header}</thead>
    <tbody>{rows}</tbody>
</table>
</body></html>"""


def compute_trades_height(current_trades: pd.DataFrame) -> int:
    """Estimate iframe height for the trades table."""
    if current_trades.empty:
        return 36
    return 32 + len(current_trades) * 28 + 4


# ---------------------------------------------------------------------------
# Sidebar — Data source selection
# ---------------------------------------------------------------------------
DATA_ROOT = st.sidebar.text_input(
    "Data root directory",
    # Default to the `data` directory next to this script so the repo is portable
    value=str(Path(__file__).parent / "data"),
    help="Root folder to scan for Prosperity CSV data",
)

if "data_dirs" not in st.session_state or st.sidebar.button("Rescan"):
    with st.spinner("Scanning for data..."):
        st.session_state.data_dirs = discover_data_dirs(DATA_ROOT)

data_dirs = st.session_state.get("data_dirs", [])
if not data_dirs:
    st.warning("No Prosperity data found. Check the data root path.")
    st.stop()

labels = [d["label"] for d in data_dirs]
selected_idx = st.sidebar.selectbox(
    "Round", range(len(labels)), format_func=lambda i: labels[i]
)
selected_dir = data_dirs[selected_idx]

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading market data...")
def cached_load(folder: str):
    return load_round_data(folder)


prices, trades = cached_load(selected_dir["path"])

if prices.empty:
    st.error("No price data found in selected folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Instrument selector
# ---------------------------------------------------------------------------
instruments = get_instruments(prices)
selected_instrument = st.sidebar.selectbox("Instrument", instruments)

# ---------------------------------------------------------------------------
# Pre-build index for fast scrubbing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_index(_prices, instrument, _folder_path):
    return build_instrument_index(_prices, instrument)


idx = cached_index(prices, selected_instrument, selected_dir["path"])
if idx is None:
    st.error("No data for this instrument.")
    st.stop()

total_steps = len(idx["step_idxs"])

# Pre-compute OFI series (cached)
@st.cache_data(show_spinner=False)
def cached_ofi(_trades, _prices, instrument, _idx, _folder_path):
    return build_ofi_series(_trades, _prices, instrument, _idx)

ofi_series = cached_ofi(trades, prices, selected_instrument, idx, selected_dir["path"])

# Context window slider — max = total_steps so user can view the full series
context_window = st.sidebar.slider(
    "Price context window (steps)",
    min_value=10,
    max_value=max(10, total_steps),
    value=min(50, total_steps),
    step=10,
    help="Number of steps shown each side of the cursor. Set to max for the full series.",
)

# Order book imbalance controls
st.sidebar.markdown("---")
st.sidebar.markdown("**Order Book Imbalance**")
imbalance_level = st.sidebar.radio(
    "Depth level",
    ("Level 1 (BBO)", "Level 3 (Full book)"),
    index=0,
    help="Level 1 uses best bid/ask only.  Level 3 aggregates all three levels.",
)
imbalance_col = "imbalance_l1" if "Level 1" in imbalance_level else "imbalance_l3"
imbalance_threshold = st.sidebar.slider(
    "Extreme zone threshold", 0.3, 0.95, 0.6, step=0.05,
    help="Highlight regions where |imbalance| exceeds this value.",
)

# Order Flow Imbalance (OFI) controls
st.sidebar.markdown("---")
st.sidebar.markdown("**Order Flow Imbalance (OFI)**")
OFI_WINDOWS = (5, 10, 20)
ofi_window = st.sidebar.select_slider(
    "Rolling window (steps)",
    options=list(OFI_WINDOWS),
    value=10,
    help="Number of steps over which to sum net signed trade volume.",
)
ofi_col = f"ofi_{ofi_window}"

# Pre-compute OFI for the entire instrument (cached for scrubbing speed)
@st.cache_data(show_spinner=False)
def cached_ofi(_trades, _prices, instrument, _idx, _folder_path):
    return build_ofi_series(_trades, _prices, instrument, _idx, windows=OFI_WINDOWS)

ofi_series = cached_ofi(trades, prices, selected_instrument, idx, selected_dir["path"])

st.sidebar.markdown("---")
st.sidebar.caption(f"**{selected_dir['label']}**")
st.sidebar.caption(f"{len(instruments)} instruments  ·  {total_steps} timestamps")
if not trades.empty:
    n_trades = len(trades[trades["instrument"] == selected_instrument])
    st.sidebar.caption(f"{n_trades} trades for {selected_instrument}")

if idx["day_boundaries"]:
    days = [b["day"] for b in idx["day_boundaries"]]
    st.sidebar.caption(f"Days: {min(days)} to {max(days)}")

# ---------------------------------------------------------------------------
# Timestamp slider
# ---------------------------------------------------------------------------
step = st.slider(
    "Timestamp",
    min_value=0,
    max_value=total_steps - 1,
    value=0,
    key="ts_slider",
)

# Current state
ob = get_orderbook_at_step(prices, selected_instrument, step)
current_ts_key = int(idx["step_to_ts"].get(step, 0))
current_day = ob["day"]
current_timestamp = ob["timestamp"]

# Header
day_str = f"Day {current_day}" if current_day is not None else ""
ts_str = f"t = {current_timestamp}" if current_timestamp is not None else f"step {step}"
st.markdown(f"### {selected_instrument}  ·  {day_str}  ·  {ts_str}")

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
current_trades = get_trades_at_step(trades, selected_instrument, current_ts_key)
price_ctx = get_price_context_by_step(prices, selected_instrument, step, context_window)

if not price_ctx.empty:
    ts_lo = int(price_ctx["ts_key"].min())
    ts_hi = int(price_ctx["ts_key"].max())
    trades_ctx = get_trades_in_step_range(trades, selected_instrument, ts_lo, ts_hi)
    if not trades_ctx.empty:
        trades_ctx = trades_ctx.copy()
        trades_ctx["step_idx"] = trades_ctx["ts_key"].map(idx["ts_to_step"])
        trades_ctx = trades_ctx.dropna(subset=["step_idx"])
else:
    trades_ctx = pd.DataFrame()

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

# Compute imbalance at current step
def _current_imbalance(ob: dict, level: str) -> float | None:
    """Compute imbalance from the current order book snapshot."""
    bids = ob["bids"]  # [(price, vol), ...]
    asks = ob["asks"]
    if not bids and not asks:
        return None
    if level == "imbalance_l1":
        bv = bids[0][1] if bids else 0
        av = asks[0][1] if asks else 0
    else:
        bv = sum(v for _, v in bids)
        av = sum(v for _, v in asks)
    total = bv + av
    return (bv - av) / total if total > 0 else 0.0

_cur_imb = _current_imbalance(ob, imbalance_col)

# Current OFI value at this step
_cur_ofi_row = ofi_series.loc[ofi_series["step_idx"] == step, ofi_col]
_cur_ofi = float(_cur_ofi_row.iloc[0]) if not _cur_ofi_row.empty else 0.0

mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
mcol1.metric("Mid Price", f"{ob['mid_price']:.2f}" if ob["mid_price"] else "—")

if ob["bids"] and ob["asks"]:
    best_bid = ob["bids"][0][0]
    best_ask = ob["asks"][0][0]
    spread = best_ask - best_bid
    mcol2.metric("Spread", f"{spread:.2f}")
    mcol3.metric("Best Bid", f"{best_bid:.2f}")
    mcol4.metric("Best Ask", f"{best_ask:.2f}")
else:
    mcol2.metric("Spread", "—")
    mcol3.metric("Best Bid", "—")
    mcol4.metric("Best Ask", "—")

if _cur_imb is not None:
    _imb_label = "Imbalance L1" if imbalance_col == "imbalance_l1" else "Imbalance L3"
    mcol5.metric(_imb_label, f"{_cur_imb:+.3f}")
else:
    mcol5.metric("Imbalance", "—")

mcol6.metric(f"OFI ({ofi_window})", f"{_cur_ofi:+.0f}")

# ---------------------------------------------------------------------------
# Layout: Price chart (full width) then Order Book + Trades side-by-side
# ---------------------------------------------------------------------------

# ---- FULL WIDTH: Price Chart ----
st.subheader("Price Context")

if not price_ctx.empty:
    # Compute imbalance for the visible window
    price_ctx_imb = compute_imbalance(price_ctx)
    imb_vals = price_ctx_imb[imbalance_col].values

    ctx_steps = price_ctx_imb["step_idx"].values

    # Slice pre-computed OFI to the visible window
    ofi_mask = (ofi_series["step_idx"] >= ctx_steps[0]) & (ofi_series["step_idx"] <= ctx_steps[-1])
    ofi_ctx = ofi_series.loc[ofi_mask]
    ofi_steps = ofi_ctx["step_idx"].values
    ofi_vals = ofi_ctx[ofi_col].values

    # Compute extreme OFI thresholds (top/bottom 10% of the *full* series)
    _full_ofi = ofi_series[ofi_col].values
    _nonzero = _full_ofi[_full_ofi != 0]
    if len(_nonzero) > 2:
        ofi_p90 = float(np.percentile(_nonzero, 90))
        ofi_p10 = float(np.percentile(_nonzero, 10))
    else:
        ofi_p90 = 1.0
        ofi_p10 = -1.0

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.15, 0.22, 0.23],
        vertical_spacing=0.035,
    )

    n_ticks = min(12, len(ctx_steps))
    tick_positions = np.linspace(0, len(ctx_steps) - 1, n_ticks, dtype=int)
    tick_step_vals = [int(ctx_steps[i]) for i in tick_positions]
    tick_labels_list = [idx["labels"].get(sv, str(sv)) for sv in tick_step_vals]

    day_boundary_steps = [
        b for b in idx["day_boundaries"]
        if ctx_steps[0] <= b["step_idx"] <= ctx_steps[-1]
    ]

    # ----- Row 1: Price chart -----
    fig.add_trace(
        go.Scatter(
            x=ctx_steps, y=price_ctx_imb["mid_price"].values,
            mode="lines", name="Mid Price",
            line=dict(color="#5C9DFF", width=2),
            hovertemplate="%{y:.2f}<extra>Mid</extra>",
        ),
        row=1, col=1,
    )

    if "bid_price_1" in price_ctx_imb.columns and "ask_price_1" in price_ctx_imb.columns:
        fig.add_trace(
            go.Scatter(
                x=ctx_steps, y=price_ctx_imb["bid_price_1"].values,
                mode="lines", name="Best Bid",
                line=dict(color="rgba(38,166,91,0.5)", width=1),
                hovertemplate="%{y:.2f}<extra>Bid</extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ctx_steps, y=price_ctx_imb["ask_price_1"].values,
                mode="lines", name="Best Ask",
                line=dict(color="rgba(239,83,80,0.5)", width=1),
                fill="tonexty", fillcolor="rgba(200,200,200,0.08)",
                hovertemplate="%{y:.2f}<extra>Ask</extra>",
            ),
            row=1, col=1,
        )

    if not trades_ctx.empty and "step_idx" in trades_ctx.columns:
        fig.add_trace(
            go.Scatter(
                x=trades_ctx["step_idx"].values,
                y=trades_ctx["price"].values,
                mode="markers", name="Trades",
                marker=dict(
                    color="#FFA726",
                    size=trades_ctx["quantity"].clip(upper=20).values + 4,
                    opacity=0.7, line=dict(width=1, color="#333"),
                ),
                hovertemplate="P=%{y:.1f}<br>Q=%{text}<extra>Trade</extra>",
                text=trades_ctx["quantity"].values,
            ),
            row=1, col=1,
        )

    # Current position marker
    fig.add_vline(x=step, line_dash="dash", line_color="rgba(255,255,255,0.6)", line_width=1.5)
    if ob["mid_price"]:
        fig.add_trace(
            go.Scatter(
                x=[step], y=[ob["mid_price"]],
                mode="markers", name="Now",
                marker=dict(color="white", size=10, symbol="diamond"),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Extreme OBI zones on price panel
    _in_zone = False
    _zone_start = None
    _zone_sign = 0
    for k, (sx, iv) in enumerate(zip(ctx_steps, imb_vals)):
        if abs(iv) >= imbalance_threshold:
            if not _in_zone:
                _in_zone = True
                _zone_start = sx
                _zone_sign = 1 if iv > 0 else -1
        else:
            if _in_zone:
                _color = "rgba(34,197,94,0.12)" if _zone_sign > 0 else "rgba(239,68,68,0.12)"
                fig.add_vrect(
                    x0=_zone_start, x1=ctx_steps[k - 1],
                    fillcolor=_color, line_width=0,
                    layer="below", row=1, col=1,
                )
                _in_zone = False
    if _in_zone:
        _color = "rgba(34,197,94,0.12)" if _zone_sign > 0 else "rgba(239,68,68,0.12)"
        fig.add_vrect(
            x0=_zone_start, x1=ctx_steps[-1],
            fillcolor=_color, line_width=0,
            layer="below", row=1, col=1,
        )

    # Day boundaries (all rows)
    for b in day_boundary_steps:
        fig.add_vline(x=b["step_idx"], line_dash="dot",
                      line_color="rgba(255,215,0,0.4)", line_width=1)
        fig.add_annotation(
            x=b["step_idx"], y=1.0, yref="y domain",
            text=f"Day {b['day']}", showarrow=False,
            font=dict(size=10, color="rgba(255,215,0,0.7)"),
            yanchor="bottom", xanchor="left", xshift=4,
        )

    # ----- Row 2: OBI subplot -----
    imb_colors = np.where(
        imb_vals >= 0,
        "rgba(34,197,94,0.7)",
        "rgba(239,68,68,0.7)",
    )
    fig.add_trace(
        go.Bar(
            x=ctx_steps, y=imb_vals,
            name="Imbalance",
            marker_color=imb_colors.tolist(),
            hovertemplate="%{y:+.3f}<extra>Imbalance</extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=imbalance_threshold, line_dash="dot",
                  line_color="rgba(34,197,94,0.4)", line_width=1, row=2, col=1)
    fig.add_hline(y=-imbalance_threshold, line_dash="dot",
                  line_color="rgba(239,68,68,0.4)", line_width=1, row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=0.5, row=2, col=1)

    # ----- Row 3: OFI subplot -----
    ofi_bar_colors = np.where(
        ofi_vals >= 0,
        "rgba(100,181,246,0.7)",   # blue for net buy flow
        "rgba(239,154,154,0.7)",   # light red for net sell flow
    )
    extreme_mask = (ofi_vals >= ofi_p90) | (ofi_vals <= ofi_p10)
    ofi_bar_colors = np.where(
        extreme_mask & (ofi_vals >= 0),
        "rgba(33,150,243,1.0)",    # bright blue – extreme buy
        np.where(
            extreme_mask & (ofi_vals < 0),
            "rgba(244,67,54,1.0)",     # bright red – extreme sell
            ofi_bar_colors,
        ),
    )

    fig.add_trace(
        go.Bar(
            x=ofi_steps, y=ofi_vals,
            name=f"OFI ({ofi_window})",
            marker_color=ofi_bar_colors.tolist(),
            hovertemplate="%{y:+.0f}<extra>OFI</extra>",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=ofi_p90, line_dash="dot",
                  line_color="rgba(33,150,243,0.45)", line_width=1, row=3, col=1)
    fig.add_hline(y=ofi_p10, line_dash="dot",
                  line_color="rgba(244,67,54,0.45)", line_width=1, row=3, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=0.5, row=3, col=1)

    # ----- Row 4: Volume subplot -----
    if "bid_volume_1" in price_ctx_imb.columns:
        bid_vol_cols = [c for c in price_ctx_imb.columns if c.startswith("bid_volume")]
        ask_vol_cols = [c for c in price_ctx_imb.columns if c.startswith("ask_volume")]
        fig.add_trace(
            go.Bar(x=ctx_steps, y=price_ctx_imb[bid_vol_cols].sum(axis=1).values,
                   name="Bid Vol", marker_color="rgba(38,166,91,0.5)"),
            row=4, col=1,
        )
        fig.add_trace(
            go.Bar(x=ctx_steps, y=price_ctx_imb[ask_vol_cols].sum(axis=1).values,
                   name="Ask Vol", marker_color="rgba(239,83,80,0.5)"),
            row=4, col=1,
        )

    # ----- Layout -----
    fig.update_layout(
        height=820,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        barmode="group", hovermode="x unified",
    )
    for r in (1, 2, 3):
        fig.update_xaxes(tickvals=tick_step_vals, ticktext=tick_labels_list,
                         showticklabels=False, row=r, col=1)
    fig.update_xaxes(tickvals=tick_step_vals, ticktext=tick_labels_list,
                     tickangle=-30, row=4, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    _imb_axis_label = "Imb L1" if imbalance_col == "imbalance_l1" else "Imb L3"
    fig.update_yaxes(title_text=_imb_axis_label, range=[-1.05, 1.05], row=2, col=1)
    fig.update_yaxes(title_text=f"OFI({ofi_window})", row=3, col=1)
    fig.update_yaxes(title_text="Book Vol", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True, key="price_chart")
else:
    st.info("No price context available.")

# ---------------------------------------------------------------------------
# Order Book + Trades (side by side, below chart)
# ---------------------------------------------------------------------------
ob_col, trades_col = st.columns(2)

with ob_col:
    st.subheader("Order Book Depth")
    ob_html = build_ob_ladder_html(ob)
    ob_height = compute_ob_height(ob)
    components.html(ob_html, height=ob_height, scrolling=False)

with trades_col:
    trade_count = len(current_trades) if not current_trades.empty else 0
    st.subheader(f"Trades at Timestamp ({trade_count})")
    trades_html = build_trades_html(current_trades)
    trades_height = compute_trades_height(current_trades)
    components.html(trades_html, height=trades_height, scrolling=False)

# ---------------------------------------------------------------------------
# Event Study
# ---------------------------------------------------------------------------
with st.expander("Event Study", expanded=False):
    ev_cols = st.columns([1, 1, 1, 1, 1])
    with ev_cols[0]:
        ev_type = st.selectbox(
            "Signal type",
            ["OFI Spike", "Imbalance Spike", "OFI + Imbalance"],
            key="ev_type",
        )
    with ev_cols[1]:
        ev_threshold = st.slider(
            "Threshold",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="ev_threshold",
            help="OFI: number of σ from mean.  Imbalance: value / 10 (e.g. 6 → |imb| > 0.6).",
        )
    with ev_cols[2]:
        ev_ofi_window = st.selectbox(
            "OFI window",
            [5, 10, 20],
            index=1,
            key="ev_ofi_window",
        )
    with ev_cols[3]:
        ev_horizon = st.selectbox(
            "Horizon (steps)",
            [10, 20, 50, 100],
            index=1,
            key="ev_horizon",
        )
    with ev_cols[4]:
        st.markdown("<br>", unsafe_allow_html=True)
        run_study = st.button("Run Event Study", key="run_ev_study", use_container_width=True)

    if run_study:
        ev_ofi_col = f"ofi_{ev_ofi_window}"
        result = run_event_study(
            inst_prices=idx["inst_prices"],
            ofi_df=ofi_series,
            event_type=ev_type,
            threshold=ev_threshold,
            ofi_col=ev_ofi_col,
            imbalance_col=imbalance_col,
            horizon=ev_horizon,
        )
        st.session_state["ev_result"] = result

    ev_result = st.session_state.get("ev_result")
    if ev_result and ev_result["event_count"] > 0:
        st.caption(f"**{ev_result['event_count']} events detected** — average post-event return path (bps)")
        ev_fig = go.Figure()
        ev_fig.add_trace(go.Scatter(
            x=list(range(len(ev_result["avg_path"]))),
            y=ev_result["avg_path"],
            mode="lines+markers",
            name="Avg Return",
            line=dict(color="#5C9DFF", width=2),
            marker=dict(size=4),
            hovertemplate="t+%{x}: %{y:+.2f} bps<extra></extra>",
        ))
        ev_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        ev_fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Steps after event",
            yaxis_title="Return (bps)",
            hovermode="x unified",
        )
        st.plotly_chart(ev_fig, use_container_width=True, key="ev_chart")
    elif ev_result is not None and ev_result["event_count"] == 0:
        st.info("No events detected with current settings. Try lowering the threshold.")

# ---------------------------------------------------------------------------
# Jump-to-timestamp
# ---------------------------------------------------------------------------
with st.expander("Jump to timestamp"):
    jump_cols = st.columns(3)
    with jump_cols[0]:
        jump_day = st.number_input("Day", value=current_day if current_day else 0, step=1)
    with jump_cols[1]:
        jump_ts = st.number_input("Timestamp", value=0, min_value=0, step=100)
    with jump_cols[2]:
        target_key = int(jump_day) * 1_000_000 + int(jump_ts)
        target_step = idx["ts_to_step"].get(target_key)
        if target_step is not None:
            st.caption(f"Exact match at step {target_step}")
        else:
            arr = np.array(list(idx["ts_to_step"].keys()))
            if len(arr) > 0:
                nearest_key = arr[np.abs(arr - target_key).argmin()]
                nearest_step = idx["ts_to_step"][nearest_key]
                day_part = int(nearest_key // 1_000_000)
                ts_part = int(nearest_key % 1_000_000)
                st.caption(f"Nearest: step {nearest_step} (Day {day_part}, t={ts_part})")
