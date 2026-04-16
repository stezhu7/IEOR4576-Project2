from __future__ import annotations
import base64
import io
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4d",
    "axes.labelcolor":   "#c8cad8",
    "xtick.color":       "#8890a8",
    "ytick.color":       "#8890a8",
    "text.color":        "#e0e2f0",
    "grid.color":        "#2a2d3d",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
}

PALETTE = ["#4f8ef7", "#f7914f", "#4ff798", "#f74f7e", "#b04ff7",
           "#f7e04f", "#4ff7e8", "#f74fb0", "#8ef74f", "#f74f4f"]


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _save_artifact(fig: plt.Figure, name: str) -> str:
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ARTIFACTS_DIR, f"{ts}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def bar_chart(data: dict, title: str, xlabel: str, ylabel: str) -> tuple[str, str]:
    with plt.rc_context(STYLE):
        labels = list(data.keys())
        values = list(data.values())
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
                      edgecolor="none", width=0.6)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.1%}" if abs(x) < 10 else f"{x:.2f}"
        ))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path  = _save_artifact(fig, "bar")
        b64   = _fig_to_b64(fig)
    return b64, path


def line_chart(df: pd.DataFrame, x_col: str, y_cols: list[str],
               title: str, xlabel: str, ylabel: str) -> tuple[str, str]:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, col in enumerate(y_cols):
            if col in df.columns:
                ax.plot(df[x_col], df[col],
                        label=col, color=PALETTE[i % len(PALETTE)],
                        linewidth=1.5, alpha=0.9)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.3)
        ax.grid(True)
        ax.set_axisbelow(True)
        plt.tight_layout()
        path = _save_artifact(fig, "line")
        b64  = _fig_to_b64(fig)
    return b64, path


def scatter_chart(df: pd.DataFrame, x_col: str, y_col: str,
                  label_col: str | None,
                  title: str, xlabel: str, ylabel: str) -> tuple[str, str]:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(df[x_col], df[y_col],
                   color=PALETTE[0], alpha=0.7, s=60, edgecolors="none")
        if label_col and label_col in df.columns:
            for _, row in df.iterrows():
                ax.annotate(str(row[label_col]),
                            (row[x_col], row[y_col]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color="#c8cad8", alpha=0.8)
        x_vals = df[x_col].dropna().values
        y_vals = df[y_col].dropna().values
        if len(x_vals) > 2:
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_range, m * x_range + b,
                    color=PALETTE[1], linewidth=1.5, linestyle="--",
                    label=f"trend (slope={m:.3f})")
            ax.legend(fontsize=9, framealpha=0.3)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True)
        ax.set_axisbelow(True)
        plt.tight_layout()
        path = _save_artifact(fig, "scatter")
        b64  = _fig_to_b64(fig)
    return b64, path


def run_viz(json_data: str, chart_type: str,
            title: str = "Chart", xlabel: str = "", ylabel: str = "",
            label_col: str = "", value_col: str = "",
            x_col: str = "", y_col: str = "", y_cols: str = "",
            top_n: str = "") -> dict:
    config = {
        "title":     title,
        "xlabel":    xlabel,
        "ylabel":    ylabel,
        "label_col": label_col or None,
        "value_col": value_col or None,
        "x_col":     x_col or None,
        "y_col":     y_col or None,
        "y_cols":    [c.strip() for c in y_cols.split(",")] if y_cols else None,
        "top_n":     int(top_n) if top_n else None,
    }
    try:
        records = json.loads(json_data)
        df = pd.DataFrame(records)
        title   = config.get("title", "Chart")
        xlabel  = config.get("xlabel", "")
        ylabel  = config.get("ylabel", "")

        if chart_type == "bar":
            lc    = config.get("label_col")
            vc    = config.get("value_col")
            top_n = config.get("top_n")
            # Auto-detect label/value columns if not specified
            if not lc or not vc:
                cols = df.columns.tolist()
                str_cols = df.select_dtypes(include="object").columns.tolist()
                num_cols = df.select_dtypes(include="number").columns.tolist()
                lc = lc or (str_cols[0] if str_cols else cols[0])
                vc = vc or (num_cols[0] if num_cols else cols[1])
            if lc in df.columns and vc in df.columns:
                df_sorted = df.sort_values(vc, ascending=False)
                if top_n:
                    df_sorted = df_sorted.head(int(top_n))
                data = dict(zip(df_sorted[lc].astype(str), df_sorted[vc]))
            else:
                cols = df.columns.tolist()
                df_sorted = df.sort_values(cols[1], ascending=False) if len(cols) > 1 else df
                data = dict(zip(df_sorted[cols[0]].astype(str), df_sorted[cols[1]]))
            b64, path = bar_chart(data, title, xlabel, ylabel)

        elif chart_type == "line":
            x_col  = config.get("x_col", df.columns[0])
            y_cols = config.get("y_cols", df.columns[1:3].tolist())
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                x_col = "date"
            b64, path = line_chart(df, x_col, y_cols, title, xlabel, ylabel)

        elif chart_type == "scatter":
            x_col     = config.get("x_col", df.columns[0])
            y_col     = config.get("y_col", df.columns[1])
            label_col = config.get("label_col")
            b64, path = scatter_chart(df, x_col, y_col, label_col, title, xlabel, ylabel)

        else:
            return {"success": False, "error": f"Unknown chart_type: {chart_type}"}

        return {
            "success":       True,
            "chart_base64":  b64,
            "chart_title":   title,
            "artifact_path": path,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "chart_base64": None}