from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from monitor_state import UltraNestLiveMonitor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_LABEL_RE = re.compile(r"\$a_\{(?P<b>[0-9.]+)\}\$")

# Colour palette
COL_BG       = "#f3f4f8"
COL_CARD     = "#ffffff"
COL_BORDER   = "#e8eaef"
COL_TEXT     = "#1a1d26"
COL_TEXT_SEC = "#6b7280"
COL_TEXT_MUT = "#9ca3af"
COL_ACCENT   = "#4f6ef7"
COL_ACC_LT   = "rgba(79,110,247,0.08)"
COL_TEAL     = "#10b981"
COL_ORANGE   = "#f59e0b"
COL_RED      = "#ef4444"
COL_PURPLE   = "#8b5cf6"

_CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=12, color=COL_TEXT_SEC),
    title_font=dict(size=14, color=COL_TEXT),
    margin=dict(l=48, r=16, t=40, b=40),
    xaxis=dict(gridcolor="#f0f1f5", zerolinecolor="#e8eaef", showline=False),
    yaxis=dict(gridcolor="#f0f1f5", zerolinecolor="#e8eaef", showline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif"),
    dragmode="zoom",
)

def _cl(**kw: Any) -> dict:
    d = dict(_CHART_LAYOUT); d.update(kw); return d

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live UltraNest monitor for ASAP runs")
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8062)
    p.add_argument("--interval-ms", type=int, default=3000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Figures – existing panels
# ---------------------------------------------------------------------------

def _empty(title="", yt="", note=None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_cl(title=title, yaxis_title=yt, height=280))
    if note:
        fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=13, color=COL_TEXT_MUT))
    return fig

def _make_logz_figure(pts, evts):
    if not pts: return _empty("Evidence trajectory", "logZ")
    x = [p["sample_index"] for p in pts]; y = [p["logz"] for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="logZ",
        line=dict(width=2, color=COL_ACCENT),
        customdata=[[p["iteration"], p["ncalls"]] for p in pts],
        hovertemplate="sample=%{x}<br>logZ=%{y:.3f}<br>iter=%{customdata[0]}<br>ncalls=%{customdata[1]}<extra></extra>"))
    for e in evts[-80:]:
        mx = e.get("sample_index")
        if mx is not None and e.get("kind") == "will_add_live_points":
            fig.add_vline(x=mx, line_width=1, line_dash="dot", line_color=COL_RED, opacity=0.5)
    fig.update_layout(**_cl(title="Evidence trajectory (logZ)", yaxis_title="logZ", height=300))
    return fig

def _make_remainder_figure(pts):
    if not pts: return _empty("Remaining evidence fraction", "percent")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["sample_index"] for p in pts], y=[p["remainder_fraction"] for p in pts],
        mode="lines", name="remainder", line=dict(width=2, color=COL_TEAL),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="sample=%{x}<br>remainder=%{y:.4f}%<extra></extra>"))
    fig.update_layout(**_cl(title="Remaining evidence fraction", yaxis_title="percent", height=300))
    return fig

def _make_lbounds_figure(pts):
    if not pts: return _empty("Likelihood bounds", "log-likelihood")
    x = [p["sample_index"] for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[p["lmin"] for p in pts], mode="lines", name="Lmin",
        line=dict(width=1.5, color=COL_ACCENT)))
    fig.add_trace(go.Scatter(x=x, y=[p["lmax"] for p in pts], mode="lines", name="Lmax",
        line=dict(width=1.5, color=COL_ORANGE)))
    fig.update_layout(**_cl(title="Likelihood bounds", yaxis_title="log-likelihood", height=300))
    return fig

def _make_throughput_figure(pts):
    if len(pts) < 3: return _empty("Throughput", "ncalls / s")
    xs, ys = [], []
    for i in range(1, len(pts)):
        prev, cur = pts[i-1], pts[i]
        dt = cur["clock_seconds"] - prev["clock_seconds"]
        if dt <= 0: dt += 86400
        if dt <= 0: continue
        dc = cur["ncalls"] - prev["ncalls"]
        if dc < 0: continue
        xs.append(cur["sample_index"]); ys.append(dc / dt)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="ncalls/s",
        line=dict(width=2, color=COL_PURPLE),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.06)"))
    fig.update_layout(**_cl(title="Likelihood call throughput", yaxis_title="ncalls / second", height=300))
    return fig

# ---------------------------------------------------------------------------
# Figures – NEW panels
# ---------------------------------------------------------------------------

def _make_ndraw_figure(pts):
    """Proposals-per-acceptance (ndraw) trend — sampler health indicator."""
    if len(pts) < 3: return _empty("Proposals per acceptance (ndraw)", "ndraw")
    x = [p["sample_index"] for p in pts]
    y = [p["ndraw"] for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="ndraw",
        line=dict(width=1.5, color=COL_ORANGE),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.06)",
        hovertemplate="sample=%{x}<br>ndraw=%{y:,}<extra></extra>"))
    fig.update_layout(**_cl(title="Proposals per acceptance (ndraw)", yaxis_title="ndraw",
                            height=300, yaxis_type="log"))
    return fig

def _make_correlation_heatmap(weighted_summary):
    """Weighted parameter correlation matrix."""
    if not weighted_summary:
        return _empty("Parameter correlations", note="Waiting for posterior …")
    corr = weighted_summary.get("correlation_matrix")
    labels = weighted_summary.get("correlation_labels", [])
    if corr is None or len(labels) < 2:
        return _empty("Parameter correlations", note="Need ≥ 2 parameters")

    # Shorten labels for display
    short = []
    for lb in labels:
        lb = lb.replace("$", "").replace("\\rm ", "").replace("\\,", " ")
        lb = re.sub(r"\{([^}]*)\}", r"\1", lb)
        if len(lb) > 18:
            lb = lb[:16] + "…"
        short.append(lb)

    fig = go.Figure(go.Heatmap(
        z=corr, x=short, y=short,
        colorscale=[
            [0.0, "#4f6ef7"],   # -1 strong blue
            [0.25, "#b5c7fb"],
            [0.5, "#f8f8fa"],   #  0 near-white
            [0.75, "#f9b89a"],
            [1.0, "#d85a30"],   # +1 strong coral
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", thickness=12, len=0.6),
    ))
    n = len(short)
    fig.update_layout(**_cl(
        title="Parameter correlations",
        height=max(320, 50 + 28 * n),
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=120, b=100, t=40, r=16),
    ))
    return fig

# ---------------------------------------------------------------------------
# Figures – B-distribution & posterior evolution (unchanged logic)
# ---------------------------------------------------------------------------

def _extract_b_dist(ws):
    if not ws: return []
    labels, q16, q50, q84 = ws.get("labels",[]), ws.get("q16",[]), ws.get("q50",[]), ws.get("q84",[])
    n = min(len(labels), len(q16), len(q50), len(q84))
    pts = []
    for i in range(n):
        lb = labels[i]
        if "derived" in lb and "a_{0.0}" in lb:
            bv = 0.0
        else:
            m = FIELD_LABEL_RE.fullmatch(lb)
            if not m: continue
            bv = float(m.group("b"))
        pts.append(dict(label=lb, b=bv, q16=float(q16[i]), q50=float(q50[i]), q84=float(q84[i])))
    pts.sort(key=lambda d: d["b"])
    return pts

def _make_b_dist_figure(ws):
    src = (ws or {}).get("source")
    sub = " (live approx)" if src == "points_hdf5_approx" else ""
    dist = _extract_b_dist(ws)
    if not dist:
        return _empty(f"B-distribution{sub}", "filling factor", note="Waiting for posterior …")
    bv = [d["b"] for d in dist]; med = [d["q50"] for d in dist]
    up = [d["q84"]-d["q50"] for d in dist]; lo = [d["q50"]-d["q16"] for d in dist]
    hd = [[d["label"],d["q16"],d["q84"]] for d in dist]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bv, y=med,
        marker=dict(color=COL_TEAL, line=dict(color="rgba(0,0,0,0.06)", width=1), cornerradius=4),
        error_y=dict(type="data", array=up, arrayminus=lo, thickness=1.4, color=COL_TEXT_SEC),
        customdata=hd, name="median ± 68 %",
        hovertemplate="B=%{x:.1f} kG<br>%{customdata[0]}<br>median=%{y:.5g}<br>68%=[%{customdata[1]:.5g}, %{customdata[2]:.5g}]<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=bv, ticktext=[f"{b:.1f}" for b in bv])
    fig.update_layout(**_cl(title=f"B-distribution{sub}", xaxis_title="B (kG)", yaxis_title="filling factor", height=340))
    return fig

def _post_opts(snap):
    evo = snap.get("weighted_evolution") or {}
    np_ = int(evo.get("num_params", 0))
    if np_ <= 0: return []
    nm = evo.get("labels", [])
    return [{"label": (nm[i] if i < len(nm) else f"param_{i}"), "value": i} for i in range(np_)]

def _resolve_sel(val, opts):
    if not opts: return None
    vs = {o["value"] for o in opts}
    return val if val in vs else opts[0]["value"]

def _make_post_evo_figure(we, idx, lbl):
    if not we or idx is None:
        return _empty("Posterior evolution", "value", note="Waiting for checkpoints …")
    cp = we.get("checkpoints",[]); q16a=we.get("q16",[]); q50a=we.get("q50",[]); q84a=we.get("q84",[])
    if idx >= len(q50a) or idx >= len(q16a) or idx >= len(q84a):
        return _empty("Posterior evolution", "value", note="Parameter unavailable")
    q16,q50,q84 = q16a[idx], q50a[idx], q84a[idx]
    if not cp or not q50:
        return _empty("Posterior evolution", "value", note="No data yet")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cp,y=q84,mode="lines",line=dict(width=0,color="rgba(79,110,247,0.15)"),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cp,y=q16,mode="lines",fill="tonexty",fillcolor="rgba(79,110,247,0.10)",
        line=dict(width=0,color="rgba(79,110,247,0.15)"),name="68 % interval",
        hovertemplate="rows=%{x}<br>q16=%{y:.5g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=cp,y=q50,mode="lines+markers",marker=dict(size=4,color=COL_ACCENT),
        line=dict(width=2,color=COL_ACCENT),name="median",hovertemplate="rows=%{x}<br>median=%{y:.5g}<extra></extra>"))
    src = we.get("source")
    t = f"Posterior evolution (live): {lbl}" if src == "points_hdf5_approx" else f"Posterior evolution: {lbl}"
    xt = "rows from points.hdf5" if src == "points_hdf5_approx" else "rows from weighted_post.txt"
    fig.update_layout(**_cl(title=t, xaxis_title=xt, yaxis_title="value", height=340))
    return fig

# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def _post_stats_text(snap):
    ws = snap.get("weighted_summary") or {}
    dist = _extract_b_dist(ws)
    if not dist: return "Waiting for filling-factor posterior …"
    tag = "(live approx)" if ws.get("source") == "points_hdf5_approx" else "(median ± 68 %)"
    lines = [f"Filling-factor snapshot {tag}:"]
    for p in dist:
        lines.append(f"  B = {p['b']:5.1f} kG  →  {p['q50']:.6g}  (+{(p['q84']-p['q50']):.3g} / −{(p['q50']-p['q16']):.3g})")
    return "\n".join(lines)

def _summary_payload(snap, log_dir):
    last = snap.get("last_point"); r = snap.get("results_json") or {}; evo = snap.get("weighted_evolution") or {}
    return {
        "log_dir": str(log_dir), "status": snap.get("status"),
        "num_points": len(snap.get("points",[])), "num_events": len(snap.get("events",[])),
        "throughput_ncalls_per_sec": snap.get("throughput_ncalls_per_sec"),
        "elapsed_seconds": snap.get("elapsed_seconds"),
        "eta_seconds": snap.get("eta_seconds"),
        "sampling_efficiency": snap.get("sampling_efficiency"),
        "dlogz_remaining": snap.get("dlogz_remaining"),
        "ess": (snap.get("weighted_summary") or {}).get("ess"),
        "mww_test": snap.get("mww_test"),
        "posterior_evolution": {"num_params": evo.get("num_params"), "num_checkpoints": len(evo.get("checkpoints",[]))},
        "latest": {
            "clock": last.get("clock") if last else None,
            "iteration": last.get("iteration") if last else None,
            "ncalls": last.get("ncalls") if last else None,
            "logz": last.get("logz") if last else None,
            "remainder_fraction": last.get("remainder_fraction") if last else None,
            "lmin": last.get("lmin") if last else None,
            "lmax": last.get("lmax") if last else None,
        },
        "results_json": {
            "niter": r.get("niter"), "ncall": r.get("ncall"),
            "logz": r.get("logz"), "logzerr": r.get("logzerr"),
        },
    }

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds):
    """Seconds → human-readable '2h 14m 08s' string."""
    if seconds is None: return "—"
    s = int(seconds)
    if s < 0: return "—"
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0: return f"{h}h {m:02d}m {s:02d}s"
    if m > 0: return f"{m}m {s:02d}s"
    return f"{s}s"

def _mww_badge(mww):
    """Return (text, colour_class) for the MWW insertion-order test."""
    if mww is None: return "—", "muted"
    pv = mww.get("p_value")
    if pv is None: return "—", "muted"
    pv = float(pv)
    if pv >= 0.05:  return f"p = {pv:.3f}", "good"     # pass
    if pv >= 0.01:  return f"p = {pv:.3f}", "warn"     # borderline
    return f"p = {pv:.4f}", "bad"                       # fail

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',system-ui,-apple-system,sans-serif;background:{COL_BG};color:{COL_TEXT};-webkit-font-smoothing:antialiased}}
#_dash-app-content{{padding:0!important}}
::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:#d1d5db;border-radius:3px}}

/* KPI cards */
.kpi-card{{background:{COL_CARD};border:1px solid {COL_BORDER};border-radius:12px;padding:18px 20px 14px;flex:1 1 0;min-width:130px;transition:box-shadow .15s ease}}
.kpi-card:hover{{box-shadow:0 2px 12px rgba(0,0,0,0.04)}}
.kpi-label{{font-size:11px;font-weight:500;color:{COL_TEXT_SEC};letter-spacing:.03em;text-transform:uppercase;margin-bottom:5px}}
.kpi-value{{font-size:24px;font-weight:700;color:{COL_TEXT};line-height:1.15;font-variant-numeric:tabular-nums}}
.kpi-sub{{font-size:11px;color:{COL_TEXT_MUT};margin-top:3px;font-variant-numeric:tabular-nums}}

/* Chart card */
.chart-card{{background:{COL_CARD};border:1px solid {COL_BORDER};border-radius:12px;padding:4px 6px 0;transition:box-shadow .15s ease}}
.chart-card:hover{{box-shadow:0 2px 12px rgba(0,0,0,0.04)}}

/* Section title */
.section-title{{font-size:15px;font-weight:600;color:{COL_TEXT};margin-bottom:12px;padding-left:2px}}

/* Dropdown */
.Select-control{{border-color:{COL_BORDER}!important;border-radius:8px!important}}

/* Text block */
.text-block{{background:{COL_CARD};border:1px solid {COL_BORDER};border-radius:12px;padding:18px 22px;
  font-family:'JetBrains Mono','SF Mono','Fira Code',monospace;font-size:12px;line-height:1.6;
  color:{COL_TEXT_SEC};white-space:pre-wrap;max-height:260px;overflow-y:auto}}

/* Status badge */
.status-badge{{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:500;padding:4px 14px;border-radius:20px;background:{COL_ACC_LT};color:{COL_ACCENT}}}
.status-dot{{width:7px;height:7px;border-radius:50%;background:{COL_ACCENT}}}
.status-badge.running .status-dot{{background:{COL_TEAL};animation:pulse-dot 1.8s ease-in-out infinite}}
.status-badge.running{{background:rgba(16,185,129,0.08);color:{COL_TEAL}}}
.status-badge.completed{{background:rgba(16,185,129,0.08);color:{COL_TEAL}}}
.status-badge.waiting .status-dot{{background:{COL_ORANGE}}}
.status-badge.waiting{{background:rgba(245,158,11,0.08);color:{COL_ORANGE}}}
@keyframes pulse-dot{{0%,100%{{opacity:1}}50%{{opacity:.35}}}}

/* MWW badge */
.mww-badge{{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;padding:3px 12px;border-radius:16px;font-variant-numeric:tabular-nums}}
.mww-badge.good{{background:rgba(16,185,129,0.10);color:#059669}}
.mww-badge.warn{{background:rgba(245,158,11,0.10);color:#b45309}}
.mww-badge.bad{{background:rgba(239,68,68,0.10);color:#dc2626}}
.mww-badge.muted{{background:rgba(156,163,175,0.10);color:{COL_TEXT_MUT}}}

/* Plotly modebar */
.modebar{{opacity:0!important;transition:opacity .2s}}
.chart-card:hover .modebar{{opacity:.7!important}}
"""

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _kpi(cid, label, sid=None):
    ch = [html.Div(label, className="kpi-label"), html.Div("—", id=cid, className="kpi-value")]
    if sid: ch.append(html.Div("", id=sid, className="kpi-sub"))
    return html.Div(ch, className="kpi-card")

def _cc(*ch, style=None):
    return html.Div(list(ch), className="chart-card", style=style or {})

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(log_dir: Path, monitor: UltraNestLiveMonitor, interval_ms: int) -> Dash:
    app = Dash(__name__, title="UltraNest Live Monitor", update_title=None)

    app.index_string = '<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>' + CUSTOM_CSS + '</style><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'

    gcfg = dict(displayModeBar=True, displaylogo=False,
                modeBarButtonsToRemove=["lasso2d","select2d","autoScale2d"])

    app.layout = html.Div([
        dcc.Interval(id="refresh", interval=interval_ms, n_intervals=0),

        # ─── Header ───────────────────────────────────────────────
        html.Div([html.Div([
            html.Div([
                html.Span("◉", style={"fontSize":"20px","color":COL_ACCENT,"marginRight":"10px"}),
                html.Span("UltraNest", style={"fontWeight":"700","fontSize":"18px"}),
                html.Span(" Live Monitor", style={"fontWeight":"400","fontSize":"18px","color":COL_TEXT_SEC}),
            ], style={"display":"flex","alignItems":"center"}),
            html.Div([
                html.Div(id="mww-badge-el", style={"marginRight":"12px"}),
                html.Div(id="status-badge"),
            ], style={"display":"flex","alignItems":"center","gap":"8px"}),
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"center",
                  "maxWidth":"1440px","margin":"0 auto","padding":"0 32px"})],
        style={"background":COL_CARD,"borderBottom":f"1px solid {COL_BORDER}",
               "padding":"14px 0","position":"sticky","top":"0","zIndex":"100"}),

        # ─── Body ─────────────────────────────────────────────────
        html.Div([
            html.Div(f"Watching {log_dir}", style={"fontSize":"12px","color":COL_TEXT_MUT,"marginBottom":"20px","wordBreak":"break-all"}),

            # ─── KPI row 1: core sampler stats ────────────────────
            html.Div([
                _kpi("kpi-iteration","Iteration"),
                _kpi("kpi-ncalls","Likelihood Calls"),
                _kpi("kpi-logz","log Z", sid="kpi-logz-sub"),
                _kpi("kpi-remainder","Remainder"),
                _kpi("kpi-dlogz","Δlog Z remaining"),
                _kpi("kpi-samples","Samples"),
            ], style={"display":"flex","gap":"14px","marginBottom":"12px","flexWrap":"wrap"}),

            # ─── KPI row 2: timing & health ──────────────────────
            html.Div([
                _kpi("kpi-elapsed","Elapsed"),
                _kpi("kpi-eta","Est. Time Remaining"),
                _kpi("kpi-throughput","Throughput"),
                _kpi("kpi-efficiency","Sampling Efficiency"),
                _kpi("kpi-ess","Effective Sample Size", sid="kpi-ess-sub"),
            ], style={"display":"flex","gap":"14px","marginBottom":"20px","flexWrap":"wrap"}),

            # ─── Chart row 1 ─────────────────────────────────────
            html.Div([
                _cc(dcc.Graph(id="fig-logz", config=gcfg, style={"height":"300px"}), style={"flex":"1 1 0","minWidth":"400px"}),
                _cc(dcc.Graph(id="fig-remainder", config=gcfg, style={"height":"300px"}), style={"flex":"1 1 0","minWidth":"400px"}),
            ], style={"display":"flex","gap":"14px","marginBottom":"14px","flexWrap":"wrap"}),

            # ─── Chart row 2 ─────────────────────────────────────
            html.Div([
                _cc(dcc.Graph(id="fig-lbounds", config=gcfg, style={"height":"300px"}), style={"flex":"1 1 0","minWidth":"400px"}),
                _cc(dcc.Graph(id="fig-throughput", config=gcfg, style={"height":"300px"}), style={"flex":"1 1 0","minWidth":"400px"}),
            ], style={"display":"flex","gap":"14px","marginBottom":"14px","flexWrap":"wrap"}),

            # ─── Chart row 3: ndraw ──────────────────────────────
            html.Div([
                _cc(dcc.Graph(id="fig-ndraw", config=gcfg, style={"height":"300px"}), style={"flex":"1 1 0","minWidth":"400px"}),
                _cc(dcc.Graph(id="fig-correlation", config=gcfg, style={"height":"auto"}), style={"flex":"1 1 0","minWidth":"400px"}),
            ], style={"display":"flex","gap":"14px","marginBottom":"28px","flexWrap":"wrap"}),

            # ─── Posterior section ────────────────────────────────
            html.Div("Phase 2 — Posterior analysis", className="section-title"),
            html.Div([dcc.Dropdown(id="posterior-param-dropdown", clearable=False,
                                   placeholder="Select parameter …", style={"fontSize":"13px"})],
                     style={"maxWidth":"360px","marginBottom":"14px"}),
            html.Div([
                _cc(dcc.Graph(id="fig-post-evo", config=gcfg, style={"height":"340px"}), style={"flex":"1 1 0","minWidth":"400px"}),
                _cc(dcc.Graph(id="fig-b-dist", config=gcfg, style={"height":"340px"}), style={"flex":"1 1 0","minWidth":"400px"}),
            ], style={"display":"flex","gap":"14px","marginBottom":"20px","flexWrap":"wrap"}),

            # ─── Text panels ─────────────────────────────────────
            html.Div([
                html.Div([html.Div("Filling-factor summary",className="section-title"),
                          html.Div(id="post-stats",className="text-block")],style={"flex":"1 1 0","minWidth":"380px"}),
                html.Div([html.Div("Live summary",className="section-title"),
                          html.Div(id="summary-text",className="text-block",style={"maxHeight":"280px"})],style={"flex":"1 1 0","minWidth":"380px"}),
            ], style={"display":"flex","gap":"14px","marginBottom":"40px","flexWrap":"wrap"}),

        ], style={"maxWidth":"1440px","margin":"0 auto","padding":"24px 32px"}),
    ], style={"minHeight":"100vh","background":COL_BG})

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    @app.callback(
        Output("status-badge","children"),
        Output("mww-badge-el","children"),
        # KPI row 1
        Output("kpi-iteration","children"), Output("kpi-ncalls","children"),
        Output("kpi-logz","children"), Output("kpi-logz-sub","children"),
        Output("kpi-remainder","children"), Output("kpi-dlogz","children"),
        Output("kpi-samples","children"),
        # KPI row 2
        Output("kpi-elapsed","children"), Output("kpi-eta","children"),
        Output("kpi-throughput","children"), Output("kpi-efficiency","children"),
        Output("kpi-ess","children"), Output("kpi-ess-sub","children"),
        # Charts
        Output("fig-logz","figure"), Output("fig-remainder","figure"),
        Output("fig-lbounds","figure"), Output("fig-throughput","figure"),
        Output("fig-ndraw","figure"), Output("fig-correlation","figure"),
        # Posterior
        Output("posterior-param-dropdown","options"), Output("posterior-param-dropdown","value"),
        Output("fig-post-evo","figure"), Output("fig-b-dist","figure"),
        # Text
        Output("post-stats","children"), Output("summary-text","children"),
        Input("refresh","n_intervals"),
        State("posterior-param-dropdown","value"),
    )
    def refresh(_, sel_param):
        snap = monitor.update()
        status = snap.get("status","unknown")
        last = snap.get("last_point")
        pts = snap.get("points",[])
        evts = snap.get("events",[])
        ws = snap.get("weighted_summary") or {}

        # ── Status badge ──
        badge = html.Div([html.Span(className="status-dot"), html.Span(status.capitalize())],
                         className=f"status-badge {status}")

        # ── MWW badge ──
        mww = snap.get("mww_test")
        mww_text, mww_cls = _mww_badge(mww)
        mww_el = html.Div([
            html.Span("MWW ", style={"fontWeight":"400","fontSize":"11px"}),
            html.Span(mww_text),
        ], className=f"mww-badge {mww_cls}")

        # ── KPI row 1 ──
        if last:
            ki = f"{last['iteration']:,}"
            kn = f"{last['ncalls']:,}"
            kz = f"{last['logz']:.3f}"
            res = snap.get("results_json") or {}
            kz_sub = f"± {res['logzerr']:.3f}" if res.get("logzerr") else ""
            kr = f"{last['remainder_fraction']:.3f} %"
        else:
            ki = kn = kz = kr = "—"; kz_sub = ""

        dlogz = snap.get("dlogz_remaining")
        k_dlogz = f"{dlogz:.2f}" if dlogz is not None else "—"
        k_samples = f"{len(pts):,}"

        # ── KPI row 2 ──
        k_elapsed = _fmt_duration(snap.get("elapsed_seconds"))
        k_eta = _fmt_duration(snap.get("eta_seconds"))
        tp = snap.get("throughput_ncalls_per_sec")
        k_tp = f"{tp:,.1f} / s" if tp else "—"
        eff = snap.get("sampling_efficiency")
        k_eff = f"{eff*100:.1f} %" if eff is not None else "—"
        ess = ws.get("ess")
        k_ess = f"{ess:,.0f}" if ess is not None else "—"
        ess_sub = ""
        if ess is not None:
            if ess < 100: ess_sub = "low — interpret with caution"
            elif ess < 500: ess_sub = "moderate"

        # ── Figures ──
        f_logz = _make_logz_figure(pts, evts)
        f_rem  = _make_remainder_figure(pts)
        f_lb   = _make_lbounds_figure(pts)
        f_tp   = _make_throughput_figure(pts)
        f_nd   = _make_ndraw_figure(pts)
        f_corr = _make_correlation_heatmap(snap.get("weighted_summary"))

        # ── Posterior ──
        popts = _post_opts(snap)
        sel = _resolve_sel(sel_param, popts)
        sel_lbl = "parameter"
        if sel is not None:
            sel_lbl = next((o["label"] for o in popts if o["value"]==sel), f"param_{sel}")
        f_pe = _make_post_evo_figure(snap.get("weighted_evolution"), sel, sel_lbl)
        f_bd = _make_b_dist_figure(snap.get("weighted_summary"))

        # ── Text ──
        t_ps = _post_stats_text(snap)
        t_sm = json.dumps(_summary_payload(snap, log_dir), indent=2, sort_keys=False)

        return (
            badge, mww_el,
            ki, kn, kz, kz_sub, kr, k_dlogz, k_samples,
            k_elapsed, k_eta, k_tp, k_eff, k_ess, ess_sub,
            f_logz, f_rem, f_lb, f_tp, f_nd, f_corr,
            popts, sel, f_pe, f_bd,
            t_ps, t_sm,
        )

    return app

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    monitor = UltraNestLiveMonitor(log_dir=log_dir)
    app = create_app(log_dir=log_dir, monitor=monitor, interval_ms=args.interval_ms)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()