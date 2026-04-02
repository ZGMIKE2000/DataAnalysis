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
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    div[data-testid="stMetric"] {background: #1e1e2f; border-radius: 8px; padding: 12px 16px;}
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
    value="/Users/zhilialexanderli/Python_projects",
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

# Context window slider
context_window = st.sidebar.slider(
    "Price context window (steps)", 10, 200, 50, step=10
)

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
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
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

# ---------------------------------------------------------------------------
# Layout: Order book + Trades  |  Price chart
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([2, 3])

# ---- LEFT: Order Book Ladder + Trades ----
with left_col:
    st.subheader("Order Book Depth")
    ob_html = build_ob_ladder_html(ob)
    ob_height = compute_ob_height(ob)
    components.html(ob_html, height=ob_height, scrolling=False)

    st.markdown("---")
    trade_count = len(current_trades) if not current_trades.empty else 0
    st.subheader(f"Trades at Timestamp ({trade_count})")
    trades_html = build_trades_html(current_trades)
    trades_height = compute_trades_height(current_trades)
    components.html(trades_html, height=trades_height, scrolling=False)

# ---- RIGHT: Price Chart ----
with right_col:
    st.subheader("Price Context")

    if not price_ctx.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.05,
        )

        ctx_steps = price_ctx["step_idx"].values

        n_ticks = min(12, len(ctx_steps))
        tick_positions = np.linspace(0, len(ctx_steps) - 1, n_ticks, dtype=int)
        tick_step_vals = [int(ctx_steps[i]) for i in tick_positions]
        tick_labels_list = [idx["labels"].get(sv, str(sv)) for sv in tick_step_vals]

        day_boundary_steps = [
            b for b in idx["day_boundaries"]
            if ctx_steps[0] <= b["step_idx"] <= ctx_steps[-1]
        ]

        # Mid price
        fig.add_trace(
            go.Scatter(
                x=ctx_steps, y=price_ctx["mid_price"].values,
                mode="lines", name="Mid Price",
                line=dict(color="#5C9DFF", width=2),
                hovertemplate="%{y:.2f}<extra>Mid</extra>",
            ),
            row=1, col=1,
        )

        # Bid/Ask bands
        if "bid_price_1" in price_ctx.columns and "ask_price_1" in price_ctx.columns:
            fig.add_trace(
                go.Scatter(
                    x=ctx_steps, y=price_ctx["bid_price_1"].values,
                    mode="lines", name="Best Bid",
                    line=dict(color="rgba(38,166,91,0.5)", width=1),
                    hovertemplate="%{y:.2f}<extra>Bid</extra>",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ctx_steps, y=price_ctx["ask_price_1"].values,
                    mode="lines", name="Best Ask",
                    line=dict(color="rgba(239,83,80,0.5)", width=1),
                    fill="tonexty", fillcolor="rgba(200,200,200,0.08)",
                    hovertemplate="%{y:.2f}<extra>Ask</extra>",
                ),
                row=1, col=1,
            )

        # Trade markers
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

        # Current position
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

        # Day boundaries
        for b in day_boundary_steps:
            fig.add_vline(x=b["step_idx"], line_dash="dot",
                          line_color="rgba(255,215,0,0.4)", line_width=1)
            fig.add_annotation(
                x=b["step_idx"], y=1.0, yref="y domain",
                text=f"Day {b['day']}", showarrow=False,
                font=dict(size=10, color="rgba(255,215,0,0.7)"),
                yanchor="bottom", xanchor="left", xshift=4,
            )

        # Volume subplot
        if "bid_volume_1" in price_ctx.columns:
            bid_vol_cols = [c for c in price_ctx.columns if c.startswith("bid_volume")]
            ask_vol_cols = [c for c in price_ctx.columns if c.startswith("ask_volume")]
            fig.add_trace(
                go.Bar(x=ctx_steps, y=price_ctx[bid_vol_cols].sum(axis=1).values,
                       name="Bid Vol", marker_color="rgba(38,166,91,0.5)"),
                row=2, col=1,
            )
            fig.add_trace(
                go.Bar(x=ctx_steps, y=price_ctx[ask_vol_cols].sum(axis=1).values,
                       name="Ask Vol", marker_color="rgba(239,83,80,0.5)"),
                row=2, col=1,
            )

        fig.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            barmode="group", hovermode="x unified",
        )
        fig.update_xaxes(tickvals=tick_step_vals, ticktext=tick_labels_list,
                         tickangle=-30, row=2, col=1)
        fig.update_xaxes(tickvals=tick_step_vals, ticktext=tick_labels_list,
                         showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Book Vol", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True, key="price_chart")
    else:
        st.info("No price context available.")

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
