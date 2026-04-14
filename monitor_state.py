from __future__ import annotations

import configparser
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ITERATION_RE = re.compile(
    r"^(?P<clock>\d{2}:\d{2}:\d{2}) \[ultranest\] \[DEBUG\] "
    r"iteration=(?P<iteration>-?\d+), ncalls=(?P<ncalls>-?\d+), regioncalls=(?P<regioncalls>-?\d+), "
    r"ndraw=(?P<ndraw>-?\d+), logz=(?P<logz>[-+0-9.eEinfINF]+), "
    r"remainder_fraction=(?P<remainder>[0-9.]+)%, Lmin=(?P<lmin>[-+0-9.eE]+), Lmax=(?P<lmax>[-+0-9.eE]+)$"
)
WILL_ADD_RE = re.compile(r"Will add (?P<nlive>\d+) live points \(x(?P<mult>\d+)\) at L=(?P<level>.+?) \.\.\.")
EXPLORED_RE = re.compile(r"Explored until L=(?P<level>.+?)\s*$")
FIELD_NAME_RE = re.compile(r"\$a_\{(?P<kg>[0-9.]+)\}\$")


@dataclass
class UltraNestLiveMonitor:
    """Incrementally parses UltraNest logdir outputs for live monitoring."""

    log_dir: Path
    keep_last_points: int = 12000
    posterior_max_params: int = 16
    posterior_max_checkpoints: int = 30
    points: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    _debug_offset: int = 0
    _debug_partial: str = ""
    _next_index: int = 0

    _results_json: Optional[Dict[str, Any]] = None
    _results_mtime: Optional[float] = None

    _config_param_names: Optional[List[str]] = None
    _config_mtime: Optional[float] = None

    _weighted_summary: Optional[Dict[str, Any]] = None
    _weighted_evolution: Optional[Dict[str, Any]] = None
    _weighted_mtime: Optional[float] = None
    _weighted_source: Optional[str] = None

    _points_mtime: Optional[float] = None
    _points_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, Any]:
        self._consume_debug_log()
        self._load_results_json()
        self._load_config_param_names()
        self._load_weighted_summary()
        return self._snapshot()

    # ------------------------------------------------------------------
    # debug.log parsing
    # ------------------------------------------------------------------

    def _consume_debug_log(self) -> None:
        debug_file = self.log_dir / "debug.log"
        if not debug_file.exists():
            return
        size = debug_file.stat().st_size
        if size < self._debug_offset:
            self._debug_offset = 0
            self._debug_partial = ""
            self.points.clear()
            self.events.clear()
            self._next_index = 0
        with debug_file.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._debug_offset)
            chunk = handle.read()
            self._debug_offset = handle.tell()
        if not chunk:
            return
        data = self._debug_partial + chunk
        lines = data.splitlines()
        if data and not data.endswith("\n"):
            self._debug_partial = lines.pop() if lines else data
        else:
            self._debug_partial = ""
        for line in lines:
            self._parse_debug_line(line.strip())
        if len(self.points) > self.keep_last_points:
            self.points = self.points[-self.keep_last_points:]

    def _parse_debug_line(self, line: str) -> None:
        match = ITERATION_RE.match(line)
        if match:
            point = {
                "sample_index": self._next_index,
                "clock": match.group("clock"),
                "clock_seconds": self._clock_to_seconds(match.group("clock")),
                "iteration": int(match.group("iteration")),
                "ncalls": int(match.group("ncalls")),
                "regioncalls": int(match.group("regioncalls")),
                "ndraw": int(match.group("ndraw")),
                "logz": self._safe_float(match.group("logz")),
                "remainder_fraction": float(match.group("remainder")),
                "lmin": self._safe_float(match.group("lmin")),
                "lmax": self._safe_float(match.group("lmax")),
            }
            self.points.append(point)
            self._next_index += 1
            return
        if "Writing samples and results to disk ... done" in line:
            self._append_event(line, "write_done"); return
        if "Writing samples and results to disk" in line:
            self._append_event(line, "write_start"); return
        will_add = WILL_ADD_RE.search(line)
        if will_add:
            self._append_event(line, "will_add_live_points", {
                "nlive": int(will_add.group("nlive")),
                "multiple": int(will_add.group("mult")),
                "level": will_add.group("level"),
            }); return
        explored = EXPLORED_RE.search(line)
        if explored:
            self._append_event(line, "explored_until", {"level": explored.group("level")}); return

    def _append_event(self, raw_line: str, kind: str, extras: Optional[Dict[str, Any]] = None) -> None:
        clock = self._extract_clock(raw_line)
        event: Dict[str, Any] = {
            "sample_index": self.points[-1]["sample_index"] if self.points else None,
            "clock": clock, "kind": kind, "message": raw_line,
        }
        if extras:
            event.update(extras)
        self.events.append(event)
        if len(self.events) > 3000:
            self.events = self.events[-3000:]

    # ------------------------------------------------------------------
    # results.json / config
    # ------------------------------------------------------------------

    def _load_results_json(self) -> None:
        path = self.log_dir / "info" / "results.json"
        if not path.exists():
            return
        mtime = path.stat().st_mtime
        if self._results_mtime == mtime:
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                self._results_json = json.load(handle)
            self._results_mtime = mtime
        except json.JSONDecodeError:
            return

    def _load_config_param_names(self) -> None:
        path = self.log_dir.parent / "config_copy.ini"
        if not path.exists():
            return
        mtime = path.stat().st_mtime
        if self._config_mtime == mtime:
            return
        cfg = configparser.ConfigParser()
        try:
            cfg.read(path, encoding="utf-8")
        except Exception:
            return
        names = self._param_names_from_config(cfg)
        if names:
            self._config_param_names = names
            self._config_mtime = mtime

    # ------------------------------------------------------------------
    # Weighted posterior
    # ------------------------------------------------------------------

    def _load_weighted_summary(self) -> None:
        path = self.log_dir / "chains" / "weighted_post.txt"
        if path.exists():
            mtime = path.stat().st_mtime
            if self._weighted_source == "weighted_post" and self._weighted_mtime == mtime:
                return
            data = self._read_weighted_post_table(path)
            if data is not None and data.size > 0:
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] >= 4:
                    weights = np.asarray(data[:, 0], dtype=float)
                    params = np.asarray(data[:, 2:], dtype=float)
                    if self._update_weighted_products(
                        params=params, weights=weights,
                        param_names=self._resolved_param_names(), source="weighted_post",
                    ):
                        self._weighted_mtime = mtime
                        self._weighted_source = "weighted_post"
                        return
        self._load_points_hdf5_summary()

    def _load_points_hdf5_summary(self) -> None:
        path = self.log_dir / "results" / "points.hdf5"
        if not path.exists():
            return
        stat = path.stat()
        if (self._weighted_source == "points_hdf5_approx"
                and self._points_mtime == stat.st_mtime
                and self._points_size == stat.st_size):
            return
        table = self._read_points_hdf5_table(path)
        if table is None or table.ndim != 2 or table.shape[0] == 0 or table.shape[1] < 7:
            return
        ncols = table.shape[1]
        param_names = self._resolved_param_names()
        if param_names:
            num_params = min(len(param_names), ncols - 3)
        else:
            tail = ncols - 3
            num_params = tail // 2 if tail % 2 == 0 else tail
        x_dim = ncols - 3 - num_params
        if x_dim < 0:
            return
        params = np.asarray(table[:, 3 + x_dim: 3 + x_dim + num_params], dtype=float)
        logl = np.asarray(table[:, 1], dtype=float)
        nlive = np.asarray(table[:, 2], dtype=float)
        finite_rows = np.isfinite(logl) & np.isfinite(params).all(axis=1)
        if finite_rows.sum() < max(5, num_params):
            return
        params = params[finite_rows]; logl = logl[finite_rows]; nlive = nlive[finite_rows]
        finite_nlive = nlive[np.isfinite(nlive) & (nlive > 0)]
        nlive_default = float(np.median(finite_nlive)) if finite_nlive.size else 1.0
        nlive = np.where(np.isfinite(nlive) & (nlive > 0), nlive, nlive_default)
        order = np.argsort(logl)
        params = params[order]; logl = logl[order]; nlive = nlive[order]
        weights = self._approx_nested_weights(logl, nlive)
        if self._update_weighted_products(params=params, weights=weights,
                                          param_names=param_names, source="points_hdf5_approx"):
            self._points_mtime = stat.st_mtime
            self._points_size = stat.st_size
            self._weighted_source = "points_hdf5_approx"

    def _read_points_hdf5_table(self, path: Path) -> Optional[np.ndarray]:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        try:
            import h5py
        except Exception:
            return None
        try:
            with h5py.File(path, "r") as f:
                if "points" not in f:
                    return None
                return np.asarray(f["points"][:], dtype=float)
        except Exception:
            return None

    def _update_weighted_products(self, params: np.ndarray, weights: np.ndarray,
                                  param_names: List[str], source: str) -> bool:
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if params.ndim != 2 or params.shape[0] == 0 or params.shape[1] == 0:
            return False
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != params.shape[0]:
            return False
        good_w = np.isfinite(weights) & (weights >= 0)
        weights = np.where(good_w, weights, 0.0)
        wsum = float(weights.sum())
        if (not np.isfinite(wsum)) or wsum <= 0:
            weights = np.ones(params.shape[0], dtype=float) / params.shape[0]
        else:
            weights = weights / wsum

        labels, series = self._build_posterior_series(params=params, param_names=param_names)
        if not series:
            return False

        max_params = len(series)
        q16: List[float] = []
        q50: List[float] = []
        q84: List[float] = []
        for idx in range(max_params):
            quant = self._weighted_quantile(series[idx], weights, [0.16, 0.50, 0.84])
            q16.append(float(quant[0])); q50.append(float(quant[1])); q84.append(float(quant[2]))

        # ── New: ESS ─────────────────────────────────────────────
        ess = self._compute_ess(weights)

        # ── New: Correlation matrix ──────────────────────────────
        corr_matrix, corr_labels = self._compute_weighted_correlation(series, labels, weights)

        # ── Posterior evolution checkpoints ───────────────────────
        checkpoints = self._posterior_checkpoints(n_rows=params.shape[0])
        evo_q16: List[List[float]] = [[] for _ in range(max_params)]
        evo_q50: List[List[float]] = [[] for _ in range(max_params)]
        evo_q84: List[List[float]] = [[] for _ in range(max_params)]

        for end_row in checkpoints:
            w_slice = np.asarray(weights[:end_row], dtype=float)
            w_sum = float(w_slice.sum())
            if (not np.isfinite(w_sum)) or w_sum <= 0:
                w_slice = np.ones_like(w_slice) / len(w_slice)
            else:
                w_slice = w_slice / w_sum
            for idx in range(max_params):
                quant = self._weighted_quantile(series[idx][:end_row], w_slice, [0.16, 0.50, 0.84])
                evo_q16[idx].append(float(quant[0]))
                evo_q50[idx].append(float(quant[1]))
                evo_q84[idx].append(float(quant[2]))

        self._weighted_summary = {
            "n_rows": int(params.shape[0]),
            "labels": labels, "q16": q16, "q50": q50, "q84": q84,
            "ess": ess,
            "correlation_matrix": corr_matrix,
            "correlation_labels": corr_labels,
            "source": source,
        }
        self._weighted_evolution = {
            "n_rows": int(params.shape[0]),
            "checkpoints": checkpoints, "labels": labels,
            "q16": evo_q16, "q50": evo_q50, "q84": evo_q84,
            "num_params": max_params, "source": source,
        }
        return True

    def _read_weighted_post_table(self, path: Path) -> Optional[np.ndarray]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except Exception:
            return None
        if len(lines) <= 1:
            return None
        rows: List[np.ndarray] = []
        expected_cols: Optional[int] = None
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            values = np.fromstring(stripped, sep=" ")
            if values.size < 4:
                continue
            if expected_cols is None:
                expected_cols = int(values.size)
            if int(values.size) != expected_cols:
                continue
            rows.append(values)
        if not rows:
            return None
        try:
            return np.vstack(rows)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _snapshot(self) -> Dict[str, Any]:
        last_point = self.points[-1] if self.points else None
        throughput = self._estimate_throughput_ncalls_per_sec()
        results = self._results_json or {}
        param_names = self._resolved_param_names()

        status = "waiting"
        if last_point is not None:
            status = "running"
            if self._is_completed() and throughput is not None and throughput < 0.5:
                status = "completed"

        # ── New derived statistics ───────────────────────────────
        elapsed_seconds = self._compute_elapsed_seconds()
        eta_seconds = self._estimate_eta_seconds()
        sampling_efficiency = self._compute_sampling_efficiency()
        dlogz_remaining = self._compute_dlogz_remaining()
        mww_test = self._extract_mww_test(results)

        return {
            "status": status,
            "points": self.points,
            "events": self.events,
            "last_point": last_point,
            "throughput_ncalls_per_sec": throughput,
            "results_json": results,
            "weighted_summary": self._weighted_summary,
            "weighted_evolution": self._weighted_evolution,
            "param_names": param_names,
            # New fields
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "sampling_efficiency": sampling_efficiency,
            "dlogz_remaining": dlogz_remaining,
            "mww_test": mww_test,
        }

    # ------------------------------------------------------------------
    # New statistic computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ess(weights: np.ndarray) -> Optional[float]:
        """Effective sample size: 1 / sum(w^2) for normalised weights."""
        if weights.size == 0:
            return None
        w2 = float(np.sum(weights ** 2))
        if w2 <= 0 or not np.isfinite(w2):
            return None
        return 1.0 / w2

    @staticmethod
    def _compute_weighted_correlation(
        series: List[np.ndarray], labels: List[str], weights: np.ndarray,
    ) -> Tuple[Optional[List[List[float]]], List[str]]:
        """Weighted Pearson correlation matrix."""
        n = len(series)
        if n < 2 or weights.size == 0:
            return None, labels
        mat = np.column_stack(series)
        w = weights
        wmean = np.average(mat, axis=0, weights=w)
        centered = mat - wmean
        wcov = np.einsum("k,ki,kj->ij", w, centered, centered)
        diag = np.sqrt(np.diag(wcov))
        diag = np.where(diag > 0, diag, 1.0)
        corr = wcov / np.outer(diag, diag)
        corr = np.clip(corr, -1.0, 1.0)
        return corr.tolist(), labels

    def _compute_elapsed_seconds(self) -> Optional[float]:
        """Wall-clock elapsed from first to last debug.log point."""
        if len(self.points) < 2:
            return None
        first = self.points[0]["clock_seconds"]
        last = self.points[-1]["clock_seconds"]
        dt = last - first
        if dt <= 0:
            dt += 86400
        return float(dt) if dt > 0 else None

    def _estimate_eta_seconds(self, convergence_threshold: float = 0.01) -> Optional[float]:
        """Estimate seconds until remainder_fraction hits threshold via log-linear extrapolation."""
        if len(self.points) < 20:
            return None
        n_window = max(20, min(500, len(self.points) // 4))
        window = self.points[-n_window:]

        remainders = np.array([p["remainder_fraction"] for p in window], dtype=float)
        clocks = np.array([p["clock_seconds"] for p in window], dtype=float)
        # Fix midnight wraps
        for i in range(1, len(clocks)):
            if clocks[i] < clocks[i - 1]:
                clocks[i:] += 86400

        mask = remainders > 0
        if mask.sum() < 10:
            return None
        log_rem = np.log(remainders[mask])
        t = clocks[mask] - clocks[mask][0]
        if t[-1] - t[0] < 5:
            return None

        # Linear regression on log(remainder) vs time
        t_mean = t.mean()
        lr_mean = log_rem.mean()
        ss_tt = np.sum((t - t_mean) ** 2)
        if ss_tt < 1e-12:
            return None
        slope = np.sum((t - t_mean) * (log_rem - lr_mean)) / ss_tt
        if slope >= 0:
            return None  # not decreasing

        current_log_rem = log_rem[-1]
        target_log_rem = math.log(max(convergence_threshold, 1e-10))
        if current_log_rem <= target_log_rem:
            return 0.0
        dt = (target_log_rem - current_log_rem) / slope
        return float(dt) if (np.isfinite(dt) and dt > 0) else None

    def _compute_sampling_efficiency(self, window_size: int = 50) -> Optional[float]:
        """Mean 1/ndraw over a recent window. Returns fraction in [0, 1]."""
        if len(self.points) < 5:
            return None
        window = self.points[-window_size:]
        ndraw_vals = [p["ndraw"] for p in window if p["ndraw"] > 0]
        if not ndraw_vals:
            return None
        return float(np.mean([1.0 / nd for nd in ndraw_vals]))

    def _compute_dlogz_remaining(self) -> Optional[float]:
        """Estimated remaining log-evidence contribution: log(remainder/100)."""
        if not self.points:
            return None
        rem = self.points[-1]["remainder_fraction"]
        if rem <= 0:
            return None
        return float(math.log(rem / 100.0))

    @staticmethod
    def _extract_mww_test(results_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract insertion-order MWW test result into a tidy dict."""
        raw = results_json.get("insertion_order_MWW_test")
        if raw is None:
            return None
        if isinstance(raw, dict):
            return {
                "p_value": raw.get("p-value") or raw.get("p_value"),
                "statistic": raw.get("statistic") or raw.get("U"),
                "converged": raw.get("converged"),
            }
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return {"statistic": float(raw[0]), "p_value": float(raw[1]), "converged": None}
        return None

    # ------------------------------------------------------------------
    # Existing helpers
    # ------------------------------------------------------------------

    def _resolved_param_names(self) -> List[str]:
        names = self._param_names(self._results_json or {})
        if names:
            return names
        if self._config_param_names:
            return list(self._config_param_names)
        return []

    def _posterior_checkpoints(self, n_rows: int) -> List[int]:
        if n_rows <= 0:
            return []
        n_points = min(self.posterior_max_checkpoints, n_rows)
        if n_points <= 1:
            return [n_rows]
        checkpoints = np.unique(np.geomspace(1, n_rows, num=n_points).astype(int))
        if checkpoints[-1] != n_rows:
            checkpoints = np.append(checkpoints, n_rows)
        return [int(x) for x in checkpoints.tolist()]

    def _build_posterior_series(self, params: np.ndarray, param_names: List[str]) -> tuple[List[str], List[np.ndarray]]:
        n_cols = params.shape[1]
        names = list(param_names) if param_names else [f"param_{i}" for i in range(n_cols)]
        if len(names) < n_cols:
            names.extend([f"param_{i}" for i in range(len(names), n_cols)])
        field_cols: List[int] = []
        for idx, name in enumerate(names[:n_cols]):
            if FIELD_NAME_RE.fullmatch(name):
                field_cols.append(idx)
            else:
                break
        labels: List[str] = []
        series: List[np.ndarray] = []
        if field_cols:
            field_values = params[:, field_cols]
            a0 = 1.0 - np.sum(field_values, axis=1)
            labels.append("$a_{0.0}$ (derived)")
            series.append(a0)
            for col in field_cols:
                labels.append(names[col])
                series.append(params[:, col])
        non_field_cols = [idx for idx in range(n_cols) if idx not in field_cols]
        for col in non_field_cols:
            if len(series) >= self.posterior_max_params:
                break
            labels.append(names[col])
            series.append(params[:, col])
        if not series:
            limit = min(self.posterior_max_params, n_cols)
            labels = names[:limit]
            series = [params[:, idx] for idx in range(limit)]
        return labels, series

    def _estimate_throughput_ncalls_per_sec(self) -> Optional[float]:
        if len(self.points) < 2:
            return None
        window = self.points[-20:]
        first, last = window[0], window[-1]
        dt = last["clock_seconds"] - first["clock_seconds"]
        if dt <= 0:
            dt += 86400
        if dt <= 0:
            return None
        dcall = last["ncalls"] - first["ncalls"]
        if dcall < 0:
            return None
        return dcall / dt

    def _is_completed(self) -> bool:
        if not self.events:
            return False
        return any(evt["kind"] == "write_done" for evt in self.events[-40:])

    @staticmethod
    def _param_names(results_json: Dict[str, Any]) -> List[str]:
        names = results_json.get("paramnames", [])
        return [str(x) for x in names] if isinstance(names, list) else []

    @staticmethod
    def _param_names_from_config(cfg: configparser.ConfigParser) -> List[str]:
        names: List[str] = []
        fit_fields = cfg.getboolean("MAIN", "fitfields", fallback=False)
        if fit_fields:
            mag_fields = UltraNestLiveMonitor._parse_float_list(cfg.get("MAIN", "magfields", fallback=""))
            if len(mag_fields) > 1:
                for b in mag_fields[1:]:
                    names.append(f"$a_{{{float(b):.1f}}}$")
        if cfg.getboolean("ATMO", "fitteff", fallback=False):
            names.append("$T_{\\rm eff}$ (K)")
        if cfg.getboolean("ATMO", "fitlogg", fallback=False):
            names.append("$\\log{g}$ (dex)")
        if cfg.getboolean("ATMO", "fitmh", fallback=False):
            names.append("$\\rm [M/H]$ (dex)")
        if cfg.getboolean("ATMO", "fitalpha", fallback=False):
            names.append("$\\rm [\\alpha/Fe]$ (dex)")
        if cfg.getboolean("MAIN", "fitbroad", fallback=False):
            names.append("$v_{\\rm broad}\\,(km\\,s^{-1})$")
        if cfg.getboolean("MAIN", "fitrv", fallback=False):
            names.append("$RV\\,(km\\,s^{-1})$")
        if cfg.getboolean("MAIN", "fitrot", fallback=False):
            names.append("$v\\sin i\\,(km\\,s^{-1})$")
        if cfg.getboolean("MAIN", "fitmac", fallback=False):
            names.append("$\\zeta\\,(km\\,s^{-1})$")
        if cfg.getboolean("MAIN", "fitveiling", fallback=False):
            fit_bands = cfg.get("MAIN", "fitbands", fallback="")
            bands = [ch for ch in fit_bands if ch.isalpha()]
            for band in bands:
                names.append(f"veil_{band.upper()}")
        return names

    @staticmethod
    def _parse_float_list(raw: str) -> List[float]:
        cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
        values: List[float] = []
        for token in cleaned.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    @staticmethod
    def _extract_clock(line: str) -> Optional[str]:
        if len(line) >= 8 and line[2] == ":" and line[5] == ":":
            return line[:8]
        return None

    @staticmethod
    def _clock_to_seconds(clock: str) -> int:
        hh, mm, ss = clock.split(":")
        return int(hh) * 3600 + int(mm) * 60 + int(ss)

    @staticmethod
    def _safe_float(raw: str) -> float:
        low = raw.lower()
        if low in {"inf", "+inf"}:
            return float("inf")
        if low == "-inf":
            return float("-inf")
        return float(raw)

    @staticmethod
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> np.ndarray:
        order = np.argsort(values)
        sv = values[order]; sw = weights[order]
        cdf = np.cumsum(sw)
        cdf = (cdf - 0.5 * sw) / cdf[-1]
        return np.interp(np.asarray(quantiles), cdf, sv)

    @staticmethod
    def _approx_nested_weights(logl: np.ndarray, nlive: np.ndarray) -> np.ndarray:
        if logl.size == 0:
            return np.array([], dtype=float)
        logx = 0.0
        logw = np.empty_like(logl, dtype=float)
        for idx, (ll, nl) in enumerate(zip(logl, nlive)):
            nli = max(float(nl), 1.0)
            logw[idx] = float(ll) + logx - math.log(nli)
            logx -= 1.0 / nli
        finite = np.isfinite(logw)
        if not finite.any():
            return np.ones(logl.size, dtype=float) / logl.size
        max_logw = np.max(logw[finite])
        shifted = np.where(finite, np.clip(logw - max_logw, -700.0, 0.0), -700.0)
        weights = np.exp(shifted)
        weights_sum = float(weights.sum())
        if (not np.isfinite(weights_sum)) or weights_sum <= 0:
            return np.ones(logl.size, dtype=float) / logl.size
        return weights / weights_sum