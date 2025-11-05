#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === ПУТЬ К ДАННЫМ ===
DATA_PATH = Path("Output_original/F2_trunc_cj15.txt")  # скорректируй при необходимости
OUT_PNG   = Path("resonance_moments_panel.png")

def load_table(path: Path) -> pd.DataFrame:
    """
    Формат (без заголовка, пробельный разделитель):
        1: Q2
        2: M1 (1-я резонансная область)
        3: M2 (2-я резонансная область)
        4: M3 (3-я резонансная область)
        5: Mfull (полная область)
    """
    # Используем sep=r"\s+" вместо устаревшего delim_whitespace
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    df = df.iloc[:, :5].copy()
    df.columns = ["Q2", "M1", "M2", "M3", "Mfull"]

    # Преобразуем к числам и выкинем мусорные строки (если вдруг есть)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("Q2").reset_index(drop=True)
    df = df[df["Q2"] <= 3.5].copy()
    return df

def make_panel_plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    axes = axes.ravel()


    panels = [
        ("M1",   "1st"),
        ("M2",   "2nd"),
        ("M3",   "3rd"),
        ("Mfull","full"),
    ]

    x = df["Q2"].to_numpy()  # <-- ключевое: переводим в numpy, чтобы обойти баг/изменение pandas

    for ax, (col, tag) in zip(axes, panels):
        y = df[col].to_numpy()  # <-- тоже numpy
        ax.plot(x, y, color = "red",  lw=1.8)
        ax.set_ylabel(r"$M_2$")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.97, 0.90, tag,
            transform=ax.transAxes,
            ha="right", va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
            fontsize=10,
        )

    axes[2].set_xlabel(r"$Q^2\ \mathrm{[GeV^2]}$")
    axes[3].set_xlabel(r"$Q^2\ \mathrm{[GeV^2]}$")

    fig.suptitle("Moments in Resonance Regions vs $Q^2$", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png.resolve()}")

if __name__ == "__main__":
    df = load_table(DATA_PATH)
    make_panel_plot(df, OUT_PNG)
