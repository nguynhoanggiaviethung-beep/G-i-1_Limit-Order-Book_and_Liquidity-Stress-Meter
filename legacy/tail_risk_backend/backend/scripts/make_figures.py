"""Generate the nine research figures as portable, dependency-free SVG files."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.evaluation.volatility_metrics import qlike_loss  # noqa: E402


PALETTE = ["#235789", "#f2a541", "#2e8b57", "#b33f62", "#6c5ce7", "#168aad", "#8c564b"]


def _thin(values, labels=None, maximum: int = 500):
    arr = np.asarray(values, dtype=float)
    if len(arr) <= maximum:
        return arr, labels
    step = int(np.ceil(len(arr) / maximum))
    return arr[::step], (labels[::step] if labels is not None else None)


def _line_svg(path: Path, title: str, series: list[tuple[str, np.ndarray]], xlabels=None, y_label: str = "Value") -> None:
    width, height = 1120, 660
    left, right, top, bottom = 88, 38, 78, 108
    plot_w, plot_h = width - left - right, height - top - bottom
    normalized = []
    all_values = []
    max_len = max((len(values) for _, values in series), default=0)
    for name, values in series:
        arr = np.asarray(values, dtype=float).reshape(-1)
        labels = xlabels if xlabels is not None and len(xlabels) == len(arr) else None
        arr, labels = _thin(arr, labels)
        normalized.append((name, arr, labels))
        all_values.extend(arr[np.isfinite(arr)].tolist())
    if not all_values or max_len == 0:
        return
    ymin, ymax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    spread = ymax - ymin
    padding = spread * 0.08 if spread else max(abs(ymax) * 0.08, 1.0)
    ymin, ymax = ymin - padding, ymax + padding
    colors = PALETTE
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="42" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#17212b">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        y = top + plot_h * tick / 5
        value = ymax - (ymax - ymin) * tick / 5
        parts.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e9ed"/>',
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#46515c">{value:.3g}</text>',
        ])
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#65727e"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#65727e"/>',
        f'<text transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#26323d">{html.escape(y_label)}</text>',
    ])
    for series_index, (name, values, labels) in enumerate(normalized):
        points = []
        active = []
        n = len(values)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                if active:
                    points.append(" ".join(active))
                    active = []
                continue
            x = left + (index / max(n - 1, 1)) * plot_w
            y = top + (ymax - value) / (ymax - ymin) * plot_h
            active.append(f"{x:.1f},{y:.1f}")
        if active:
            points.append(" ".join(active))
        for segment in points:
            parts.append(f'<polyline points="{segment}" fill="none" stroke="{colors[series_index % len(colors)]}" stroke-width="2"/>')
        if labels is not None and n:
            for index in sorted(set([0, n // 2, n - 1])):
                label = str(labels[index])
                x = left + (index / max(n - 1, 1)) * plot_w
                parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#46515c">{html.escape(label)}</text>')
        legend_x = left + (series_index % 3) * 325
        legend_y = top + plot_h + 57 + (series_index // 3) * 22
        parts.extend([
            f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 24}" y2="{legend_y - 4}" stroke="{colors[series_index % len(colors)]}" stroke-width="3"/>',
            f'<text x="{legend_x + 31}" y="{legend_y}" font-family="Arial,sans-serif" font-size="13" fill="#26323d">{html.escape(name)}</text>',
        ])
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _heatmap_svg(path: Path, title: str, rows: list[str], columns: list[str], values: np.ndarray, mode: str = "pvalue") -> None:
    width = max(850, 240 + len(columns) * 130)
    height = max(500, 125 + len(rows) * 28)
    left, top, cell_w, cell_h = 210, 100, 122, 25
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{left}" y="40" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#17212b">{html.escape(title)}</text>',
    ]
    for col_index, column in enumerate(columns):
        x = left + col_index * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="82" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#26323d">{html.escape(str(column))}</text>')
    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        parts.append(f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#26323d">{html.escape(str(row))}</text>')
        for col_index in range(len(columns)):
            value = values[row_index, col_index]
            if not np.isfinite(value):
                color, label = "#edf0f2", "–"
            elif mode == "membership":
                color, label = ("#2e8b57" if value >= 0.5 else "#d9e2e8"), ("in" if value >= 0.5 else "out")
            elif mode == "qlike":
                scale = float(np.nanmax(values) - np.nanmin(values)) if np.isfinite(values).any() else 0.0
                normalized = (value - np.nanmin(values)) / scale if scale > 0 else 0.5
                shade = int(235 - 130 * normalized)
                color, label = f"#{shade:02x}{min(245, shade + 12):02x}{min(255, shade + 18):02x}", f"{value:.3g}"
            else:
                p = min(max(float(value), 0.0), 1.0)
                red, green = int(210 - 145 * p), int(70 + 145 * p)
                color, label = f"#{red:02x}{green:02x}75", f"{p:.3g}"
            x = left + col_index * cell_w
            parts.extend([
                f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" rx="2" fill="{color}"/>',
                f'<text x="{x + (cell_w - 3) / 2}" y="{y + 16}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#17212b">{label}</text>',
            ])
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _bar_svg(path: Path, title: str, labels: list[str], values: list[float], x_label: str) -> None:
    width, height = 1120, max(420, 130 + len(labels) * 32)
    left, top, right = 240, 82, 60
    plot_w = width - left - right
    finite = [v for v in values if np.isfinite(v)]
    max_value = max(finite, default=1.0)
    max_value = max(max_value, 1e-12) * 1.12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{left}" y="42" font-family="Arial,sans-serif" font-size="23" font-weight="700" fill="#17212b">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * 30
        safe = max(float(value), 0.0) if np.isfinite(value) else 0.0
        bar_w = safe / max_value * plot_w
        parts.extend([
            f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#26323d">{html.escape(label)}</text>',
            f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="20" fill="#235789"/>',
            f'<text x="{left + bar_w + 6:.1f}" y="{y + 16}" font-family="Arial,sans-serif" font-size="11" fill="#26323d">{safe:.3g}</text>',
        ])
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#26323d">{html.escape(x_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _forecast_frame(symbol: str, horizon: int, model: str) -> pd.DataFrame:
    path = ROOT / "results" / "forecasts" / f"{symbol.lower()}_h{horizon}_walkforward.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    return frame.loc[
        frame["model"].eq(model)
        & frame["status"].eq("ok")
        & frame["fold"].eq("oos")
        & frame["actual_rv_mean"].notna()
    ].sort_values("origin_time").copy()


def main() -> None:
    research = load_research_config()
    settings = load_settings()
    label = pd.Timestamp(research["as_of_utc"]).strftime("%Y-%m-%dT%H-%M-%SZ")
    feature_dir = settings.data_dir / "features" / "snapshots" / label
    figures = ROOT / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    daily_series = []
    daily_labels = None
    for symbol in research["symbols"]:
        path = feature_dir / f"symbol={symbol}" / "daily_rv.parquet"
        if not path.exists():
            continue
        daily = pd.read_parquet(path).sort_values("utc_date")
        daily_series.append((symbol, np.sqrt(pd.to_numeric(daily["rv"], errors="coerce").to_numpy(dtype=float))))
        if daily_labels is None:
            dates = pd.to_datetime(daily["utc_date"], utc=True)
            daily_labels = dates.dt.strftime("%Y-%m").tolist()
    if daily_series:
        _line_svg(figures / "F1_realized_volatility.svg", "F1 Daily realized volatility", daily_series, daily_labels, "Daily volatility, %")

    primary = _forecast_frame("BTCUSDT", 1, "har")
    if not primary.empty:
        primary = primary.tail(365)
        dates = pd.to_datetime(primary["origin_time"], utc=True).dt.strftime("%Y-%m").tolist()
        _line_svg(
            figures / "F2_forecast_vs_actual.svg",
            "F2 BTCUSDT one-day HAR forecast and realized variance",
            [("Realized RV", primary["actual_rv_mean"].to_numpy()), ("HAR forecast", primary["rv_forecast"].to_numpy())],
            dates,
            "Realized variance, percent squared",
        )

    btc_h1 = ROOT / "results" / "forecasts" / "btcusdt_h1_walkforward.parquet"
    if btc_h1.exists():
        forecasts = pd.read_parquet(btc_h1)
        forecasts = forecasts.loc[forecasts["status"].eq("ok") & forecasts["fold"].eq("oos") & forecasts["actual_rv_mean"].notna()]
        rolling_series = []
        dates = None
        for model in ("random_walk", "har", "elasticnet", "random_forest"):
            subset = forecasts.loc[forecasts["model"].eq(model)].sort_values("origin_time").tail(600)
            if subset.empty:
                continue
            losses = qlike_loss(
                subset["actual_rv_mean"].to_numpy(dtype=float),
                subset["rv_forecast"].to_numpy(dtype=float),
                subset["train_forecast_floor"].to_numpy(dtype=float),
            )
            rolling_series.append((model, pd.Series(losses).rolling(90, min_periods=30).mean().to_numpy()))
            dates = pd.to_datetime(subset["origin_time"], utc=True).dt.strftime("%Y-%m").tolist()
        if rolling_series:
            _line_svg(figures / "F3_rolling_qlike.svg", "F3 BTCUSDT 90-origin rolling QLIKE", rolling_series, dates, "QLIKE, lower is better")

    dm = pd.read_csv(ROOT / "results" / "tables" / "dm_hln_tests.csv") if (ROOT / "results" / "tables" / "dm_hln_tests.csv").exists() else pd.DataFrame()
    if not dm.empty:
        p_col = "pvalue_holm_global" if "pvalue_holm_global" in dm else "pvalue"
        dm["comparison"] = dm["symbol"].astype(str) + " h" + dm["horizon_days"].astype(str)
        matrix = dm.pivot_table(index="model", columns="comparison", values=p_col, aggfunc="first")
        _heatmap_svg(figures / "F4_dm_holm_heatmap.svg", "F4 DM/HLN versus random walk, Holm-adjusted p-values", matrix.index.tolist(), matrix.columns.tolist(), matrix.to_numpy(dtype=float), "pvalue")

    mcs_path = ROOT / "results" / "tables" / "mcs_membership.csv"
    if mcs_path.exists():
        mcs = pd.read_csv(mcs_path)
        if not mcs.empty:
            mcs["comparison"] = mcs["symbol"].astype(str) + " h" + mcs["horizon_days"].astype(str)
            mcs["included_num"] = mcs["included"].astype(str).str.lower().isin(["true", "1"]).astype(float)
            matrix = mcs.pivot_table(index="model", columns="comparison", values="included_num", aggfunc="first")
            _heatmap_svg(figures / "F5_mcs_membership.svg", "F5 Model confidence set membership", matrix.index.tolist(), matrix.columns.tolist(), matrix.to_numpy(dtype=float), "membership")

    risk_path = ROOT / "results" / "forecasts" / "risk_forecasts.parquet"
    risk = pd.read_parquet(risk_path) if risk_path.exists() else pd.DataFrame()
    if not risk.empty:
        selected = risk.loc[
            risk["symbol"].eq("BTCUSDT")
            & risk["model"].eq("har")
            & risk["horizon_days"].eq(1)
            & risk["method"].isin(["fhs", "filtered_historical_simulation"])
            & risk["confidence"].eq(0.95)
        ].sort_values("origin_time").tail(365)
        if not selected.empty:
            dates = pd.to_datetime(selected["origin_time"], utc=True).dt.strftime("%Y-%m").tolist()
            _line_svg(
                figures / "F6_var_coverage.svg",
                "F6 BTCUSDT loss and 95% FHS VaR",
                [("Realized loss", selected["actual_loss_percent"].to_numpy()), ("95% VaR", selected["var_loss_percent"].to_numpy())],
                dates,
                "Loss-positive return, %",
            )
            _line_svg(
                figures / "F7_es_backtest.svg",
                "F7 BTCUSDT loss, 95% FHS VaR and ES",
                [
                    ("Realized loss", selected["actual_loss_percent"].to_numpy()),
                    ("95% VaR", selected["var_loss_percent"].to_numpy()),
                    ("95% ES", selected["expected_shortfall_loss_percent"].to_numpy()),
                ],
                dates,
                "Loss-positive return, %",
            )

    regime_path = ROOT / "results" / "tables" / "regime_rv_metrics.csv"
    if regime_path.exists():
        regime = pd.read_csv(regime_path)
        subset = regime.loc[regime["symbol"].eq("BTCUSDT") & regime["horizon_days"].eq(1)]
        if not subset.empty:
            pivot = subset.pivot_table(index="regime", columns="model", values="qlike", aggfunc="first")
            preferred = [name for name in ("random_walk", "har", "elasticnet", "random_forest") if name in pivot.columns]
            if preferred:
                matrix = pivot[preferred]
                _heatmap_svg(figures / "F8_regime_qlike.svg", "F8 BTCUSDT one-day QLIKE by regime", matrix.index.tolist(), matrix.columns.tolist(), matrix.to_numpy(dtype=float), "qlike")

    if not primary.empty:
        portfolio = primary.dropna(subset=["y_cum_return", "sigma_h_approx"]).copy()
        if not portfolio.empty:
            actual = pd.to_numeric(portfolio["y_cum_return"], errors="coerce").to_numpy(dtype=float)
            sigma = pd.to_numeric(portfolio["sigma_h_approx"], errors="coerce").to_numpy(dtype=float)
            target_daily_pct = 10.0 / np.sqrt(365.0)
            weights = np.minimum(2.0, target_daily_pct / np.maximum(sigma, 1e-12))
            turnover = np.abs(weights - np.r_[0.0, weights[:-1]])
            strategy_returns = weights * actual - turnover * 0.1
            hold_returns = actual - np.r_[0.1, np.zeros(max(len(actual) - 1, 0))]
            strategy_wealth = np.cumprod(1.0 + strategy_returns / 100.0)
            hold_wealth = np.cumprod(1.0 + hold_returns / 100.0)
            dates = pd.to_datetime(portfolio["origin_time"], utc=True).dt.strftime("%Y-%m").tolist()
            _line_svg(
                figures / "F9_volatility_target_wealth.svg",
                "F9 BTCUSDT volatility-target strategy wealth",
                [("10% annualized target", strategy_wealth), ("Buy and hold", hold_wealth)],
                dates,
                "Growth of one unit of capital",
            )

    print(f"Wrote SVG figures under {figures}")


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("make_figures.py")
