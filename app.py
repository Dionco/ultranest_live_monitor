from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from monitor_state import UltraNestLiveMonitor


FIELD_LABEL_RE = re.compile(r"\$a_\{(?P<b>[0-9.]+)\}\$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live UltraNest monitor for ASAP runs")
    parser.add_argument("--log-dir", type=Path, required=True, help="Path to ultranest_logdir")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8062, help="Dash port")
    parser.add_argument("--interval-ms", type=int, default=3000, help="Refresh interval in milliseconds")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    return parser.parse_args()


def _empty_figure(title: str, yaxis_title: str = "", note: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white", xaxis_title="sample index", yaxis_title=yaxis_title)
    if note:
        fig.add_annotation(
            text=note,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": "dimgray"},
        )
    return fig


def _make_logz_figure(points: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> go.Figure:
    if not points:
        return _empty_figure("Evidence Trajectory", "logZ")

    x = [p["sample_index"] for p in points]
    y = [p["logz"] for p in points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="logZ",
            line={"width": 2},
            customdata=[[p["iteration"], p["ncalls"]] for p in points],
            hovertemplate="sample=%{x}<br>logZ=%{y:.3f}<br>iter=%{customdata[0]}<br>ncalls=%{customdata[1]}<extra></extra>",
        )
    )

    for evt in events[-80:]:
        marker_x = evt.get("sample_index")
        if marker_x is None:
            continue
        if evt.get("kind") == "will_add_live_points":
            fig.add_vline(x=marker_x, line_width=1, line_dash="dot", line_color="firebrick")

    fig.update_layout(
        title="Evidence Trajectory (logZ)",
        template="plotly_white",
        xaxis_title="sample index",
        yaxis_title="logZ",
        legend={"orientation": "h"},
    )
    return fig


def _make_remainder_figure(points: List[Dict[str, Any]]) -> go.Figure:
    if not points:
        return _empty_figure("Remainder Fraction", "percent")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p["sample_index"] for p in points],
            y=[p["remainder_fraction"] for p in points],
            mode="lines",
            name="remainder_fraction",
            line={"width": 2, "color": "teal"},
            hovertemplate="sample=%{x}<br>remainder=%{y:.4f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Remaining Evidence Fraction",
        template="plotly_white",
        xaxis_title="sample index",
        yaxis_title="percent",
    )
    return fig


def _make_likelihood_bounds_figure(points: List[Dict[str, Any]]) -> go.Figure:
    if not points:
        return _empty_figure("Likelihood Bounds", "log-likelihood")

    x = [p["sample_index"] for p in points]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[p["lmin"] for p in points],
            mode="lines",
            name="Lmin",
            line={"width": 1.5, "color": "royalblue"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[p["lmax"] for p in points],
            mode="lines",
            name="Lmax",
            line={"width": 1.5, "color": "darkorange"},
        )
    )
    fig.update_layout(
        title="Likelihood Bounds",
        template="plotly_white",
        xaxis_title="sample index",
        yaxis_title="log-likelihood",
        legend={"orientation": "h"},
    )
    return fig


def _make_throughput_figure(points: List[Dict[str, Any]]) -> go.Figure:
    if len(points) < 3:
        return _empty_figure("Sampler Throughput", "ncalls / s")

    xs = []
    ys = []
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        cur = points[idx]
        dt = cur["clock_seconds"] - prev["clock_seconds"]
        if dt <= 0:
            dt += 24 * 3600
        if dt <= 0:
            continue
        dcall = cur["ncalls"] - prev["ncalls"]
        if dcall < 0:
            continue
        xs.append(cur["sample_index"])
        ys.append(dcall / dt)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="ncalls/s",
            line={"width": 2, "color": "purple"},
        )
    )
    fig.update_layout(
        title="Likelihood Call Throughput",
        template="plotly_white",
        xaxis_title="sample index",
        yaxis_title="ncalls / second",
    )
    return fig


def _extract_b_distribution(weighted_summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not weighted_summary:
        return []

    labels = weighted_summary.get("labels", [])
    q16 = weighted_summary.get("q16", [])
    q50 = weighted_summary.get("q50", [])
    q84 = weighted_summary.get("q84", [])
    n = min(len(labels), len(q16), len(q50), len(q84))
    if n == 0:
        return []

    points: List[Dict[str, Any]] = []
    for idx in range(n):
        label = labels[idx]
        if "derived" in label and "a_{0.0}" in label:
            b_val = 0.0
        else:
            m = FIELD_LABEL_RE.fullmatch(label)
            if not m:
                continue
            b_val = float(m.group("b"))

        points.append(
            {
                "label": label,
                "b": b_val,
                "q16": float(q16[idx]),
                "q50": float(q50[idx]),
                "q84": float(q84[idx]),
            }
        )

    points.sort(key=lambda d: d["b"])
    return points


def _make_b_distribution_figure(
    weighted_summary: Optional[Dict[str, Any]],
) -> go.Figure:
    source = (weighted_summary or {}).get("source")
    title = "B Distribution Snapshot (Filling Factors)"
    if source == "points_hdf5_approx":
        title = "B Distribution Snapshot (Filling Factors, live approx from points.hdf5)"

    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="B (kG)",
        yaxis_title="filling factor",
    )

    distribution = _extract_b_distribution(weighted_summary)
    if not distribution:
        fig.add_annotation(
            text="Waiting for filling-factor posterior (chains/weighted_post.txt or results/points.hdf5)",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": "dimgray"},
        )
        return fig

    b_vals = [d["b"] for d in distribution]
    medians = [d["q50"] for d in distribution]
    upper = [d["q84"] - d["q50"] for d in distribution]
    lower = [d["q50"] - d["q16"] for d in distribution]
    hover_data = [[d["label"], d["q16"], d["q84"]] for d in distribution]

    fig.add_trace(
        go.Bar(
            x=b_vals,
            y=medians,
            marker={"color": "seagreen", "line": {"color": "darkgreen", "width": 0.8}},
            error_y={"type": "data", "array": upper, "arrayminus": lower, "thickness": 1.2},
            customdata=hover_data,
            name="filling factor median +/- 68%",
            hovertemplate="B=%{x:.1f} kG<br>%{customdata[0]}<br>median=%{y:.5g}<br>68%=[%{customdata[1]:.5g}, %{customdata[2]:.5g}]<extra></extra>",
        )
    )

    fig.update_xaxes(tickmode="array", tickvals=b_vals, ticktext=[f"{b:.1f}" for b in b_vals])
    return fig


def _posterior_param_options(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    evo = snapshot.get("weighted_evolution") or {}
    n_params = int(evo.get("num_params", 0))
    if n_params <= 0:
        return []

    names = evo.get("labels", [])
    options = []
    for idx in range(n_params):
        label = names[idx] if idx < len(names) else f"param_{idx}"
        options.append({"label": label, "value": idx})
    return options


def _resolve_selected_param(selected_value: Optional[int], options: List[Dict[str, Any]]) -> Optional[int]:
    if not options:
        return None
    option_values = {opt["value"] for opt in options}
    if selected_value in option_values:
        return selected_value
    return options[0]["value"]


def _make_posterior_evolution_figure(
    weighted_evolution: Optional[Dict[str, Any]],
    selected_param_idx: Optional[int],
    param_label: str,
) -> go.Figure:
    if not weighted_evolution or selected_param_idx is None:
        return _empty_figure(
            "Posterior Evolution (Phase 2)",
            "parameter value",
            note="Waiting for posterior checkpoints (weighted_post.txt or points.hdf5)",
        )

    checkpoints = weighted_evolution.get("checkpoints", [])
    q16_all = weighted_evolution.get("q16", [])
    q50_all = weighted_evolution.get("q50", [])
    q84_all = weighted_evolution.get("q84", [])

    if selected_param_idx >= len(q50_all) or selected_param_idx >= len(q16_all) or selected_param_idx >= len(q84_all):
        return _empty_figure(
            "Posterior Evolution (Phase 2)",
            "parameter value",
            note="Selected parameter is unavailable",
        )

    q16 = q16_all[selected_param_idx]
    q50 = q50_all[selected_param_idx]
    q84 = q84_all[selected_param_idx]

    if not checkpoints or not q50:
        return _empty_figure(
            "Posterior Evolution (Phase 2)",
            "parameter value",
            note="No posterior checkpoint data yet",
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=q84,
            mode="lines",
            line={"width": 0.5, "color": "rgba(0,128,96,0.2)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=q16,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0,128,96,0.2)",
            line={"width": 0.5, "color": "rgba(0,128,96,0.2)"},
            name="68% interval",
            hovertemplate="rows=%{x}<br>q16=%{y:.5g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=q50,
            mode="lines+markers",
            marker={"size": 5},
            line={"width": 2, "color": "seagreen"},
            name="median",
            hovertemplate="rows=%{x}<br>median=%{y:.5g}<extra></extra>",
        )
    )

    source = weighted_evolution.get("source")
    if source == "points_hdf5_approx":
        title = f"Posterior Evolution (live points.hdf5 approx): {param_label}"
        x_title = "rows processed from results/points.hdf5"
    else:
        title = f"Posterior Evolution (weighted_post): {param_label}"
        x_title = "rows processed from weighted_post.txt"

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=x_title,
        yaxis_title="parameter value",
        legend={"orientation": "h"},
    )
    return fig


def _posterior_stats_text(snapshot: Dict[str, Any]) -> str:
    summary = snapshot.get("weighted_summary") or {}
    source = summary.get("source")
    distribution = _extract_b_distribution(summary)
    if not distribution:
        return "Filling-factor summary unavailable yet (waiting for chains/weighted_post.txt or results/points.hdf5)."

    if source == "points_hdf5_approx":
        lines = ["Latest filling-factor snapshot (live approximation from points.hdf5):"]
    else:
        lines = ["Latest filling-factor snapshot (median and 68% interval):"]
    for point in distribution:
        lines.append(
            f"- B={point['b']:.1f} kG ({point['label']}): "
            f"{point['q50']:.6g} (+{(point['q84']-point['q50']):.3g} / -{(point['q50']-point['q16']):.3g})"
        )
    return "\n".join(lines)


def _summary_text(snapshot: Dict[str, Any], log_dir: Path) -> str:
    last = snapshot.get("last_point")
    results = snapshot.get("results_json") or {}
    weighted_evolution = snapshot.get("weighted_evolution") or {}

    payload: Dict[str, Any] = {
        "log_dir": str(log_dir),
        "status": snapshot.get("status"),
        "num_points": len(snapshot.get("points", [])),
        "num_events": len(snapshot.get("events", [])),
        "throughput_ncalls_per_sec": snapshot.get("throughput_ncalls_per_sec"),
        "posterior_evolution": {
            "num_params": weighted_evolution.get("num_params"),
            "num_checkpoints": len(weighted_evolution.get("checkpoints", [])),
            "n_rows": weighted_evolution.get("n_rows"),
        },
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
            "niter": results.get("niter"),
            "ncall": results.get("ncall"),
            "logz": results.get("logz"),
            "logzerr": results.get("logzerr"),
            "insertion_order_MWW_test": results.get("insertion_order_MWW_test"),
        },
    }

    return json.dumps(payload, indent=2, sort_keys=False)


def create_app(log_dir: Path, monitor: UltraNestLiveMonitor, interval_ms: int) -> Dash:
    app = Dash(__name__)
    app.title = "UltraNest Live Monitor"

    app.layout = html.Div(
        [
            html.H2("UltraNest Live Monitor"),
            html.Div(f"Watching: {log_dir}", id="watching-path"),
            html.Div(id="status-line", style={"marginBottom": "12px", "marginTop": "8px"}),
            dcc.Interval(id="refresh", interval=interval_ms, n_intervals=0),
            dcc.Graph(id="fig-logz"),
            dcc.Graph(id="fig-remainder"),
            dcc.Graph(id="fig-lbounds"),
            dcc.Graph(id="fig-throughput"),
            html.H4("Phase 2: Posterior Evolution"),
            dcc.Dropdown(id="posterior-param-dropdown", clearable=False, placeholder="Select parameter"),
            dcc.Graph(id="fig-posterior-evolution"),
            dcc.Graph(id="fig-b-distribution"),
            html.Pre(id="posterior-stats-text", style={"maxHeight": "220px", "overflowY": "auto"}),
            html.H4("Live Summary"),
            html.Pre(id="summary-text", style={"maxHeight": "320px", "overflowY": "auto"}),
        ],
        style={"maxWidth": "1400px", "margin": "0 auto", "padding": "12px"},
    )

    @app.callback(
        Output("status-line", "children"),
        Output("fig-logz", "figure"),
        Output("fig-remainder", "figure"),
        Output("fig-lbounds", "figure"),
        Output("fig-throughput", "figure"),
        Output("posterior-param-dropdown", "options"),
        Output("posterior-param-dropdown", "value"),
        Output("fig-posterior-evolution", "figure"),
        Output("fig-b-distribution", "figure"),
        Output("posterior-stats-text", "children"),
        Output("summary-text", "children"),
        Input("refresh", "n_intervals"),
        State("posterior-param-dropdown", "value"),
    )
    def refresh(_: int, selected_param_value: Optional[int]):
        snapshot = monitor.update()

        status = snapshot.get("status", "unknown")
        last = snapshot.get("last_point")
        status_line = f"Status: {status}"
        if last:
            status_line += (
                f" | iteration={last['iteration']}"
                f" | ncalls={last['ncalls']}"
                f" | logz={last['logz']:.3f}"
                f" | remainder={last['remainder_fraction']:.4f}%"
            )

        points = snapshot.get("points", [])
        events = snapshot.get("events", [])

        fig_logz = _make_logz_figure(points, events)
        fig_remainder = _make_remainder_figure(points)
        fig_lbounds = _make_likelihood_bounds_figure(points)
        fig_throughput = _make_throughput_figure(points)

        param_options = _posterior_param_options(snapshot)
        selected_param = _resolve_selected_param(selected_param_value, param_options)
        selected_label = "parameter"
        if selected_param is not None:
            selected_label = next(
                (opt["label"] for opt in param_options if opt["value"] == selected_param),
                f"param_{selected_param}",
            )
        fig_posterior_evolution = _make_posterior_evolution_figure(
            snapshot.get("weighted_evolution"), selected_param, selected_label
        )

        fig_b_distribution = _make_b_distribution_figure(snapshot.get("weighted_summary"))
        posterior_stats = _posterior_stats_text(snapshot)

        summary = _summary_text(snapshot, log_dir)
        return (
            status_line,
            fig_logz,
            fig_remainder,
            fig_lbounds,
            fig_throughput,
            param_options,
            selected_param,
            fig_posterior_evolution,
            fig_b_distribution,
            posterior_stats,
            summary,
        )

    return app


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    monitor = UltraNestLiveMonitor(log_dir=log_dir)
    app = create_app(log_dir=log_dir, monitor=monitor, interval_ms=args.interval_ms)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
