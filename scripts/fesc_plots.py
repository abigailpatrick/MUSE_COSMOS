#!/usr/bin/env python3
"""
fesc_plots.py  –  Lyα escape fraction plots from merged CSV

Usage
-----
  python fesc_plots.py \\
      --merged-csv /path/to/final_merged.csv \\
      --out-dir    /path/to/outputs/ \\issue 
      [--remove-agn] \\
      [--sphinx-table /path/to/sphinx.fits] \\
      [--caseb 8.7] \\
      [--label-ids] \\
      [--dust-tag eta_1]

Marker convention
-----------------
  lya_detect_flag == 1  : filled circle   (strong detection)
  lya_detect_flag == 2  : filled diamond  (weak detection)
  lya_detect_flag == 0  : downward triangle (upper limit)
  agn_flag       == 1   : star (★)  overrides shape above
"""

import argparse
import os

import cmasher as cmr
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from astropy.table import Table as AstropyTable


# ── rcParams ──────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "text.usetex":         False,
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.linewidth":      0.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    4,
    "ytick.major.size":    4,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.top":           True,
    "ytick.right":         True,
    "legend.frameon":      False,
    "legend.fontsize":     9,
    "figure.dpi":          150,
})


# ── colour palette ────────────────────────────────────────────────────────────

THEME = "torch"

def _make_palette(name: str, n: int = 8) -> dict:
    hexes = cmr.take_cmap_colors(
        f"cmr.{name}", n, cmap_range=(0.10, 0.90), return_fmt="hex"
    )
    p = {f"c{i}": h for i, h in enumerate(hexes)}
    p["near_black"]   = mcolors.to_hex((0.10, 0.10, 0.12, 1.0))
    p["neutral_grey"] = "#aab4c8"
    p["white"]        = "#ffffff"
    p["mid_grey"]     = "#888888"
    return p

P = _make_palette(THEME)

# Semantic colour / marker constants
C_DET    = P["near_black"]    # detection edge colour
C_UL     = P["c1"]            # upper-limit triangle
C_AGN    = P["c6"]            # AGN star
C_SPHINX = P["neutral_grey"]  # SPHINX background

M_STRONG = "o"   # strong detection
M_WEAK   = "D"   # weak detection
M_UL     = "v"   # upper limit
M_AGN    = "*"   # AGN

MS       = 35    # base marker size (pts²)
MS_AGN   = 90

# Colourmap axis: Av
AV_COL   = "av_50"
AV_LABEL = r"$A_V$"


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Return a column as a float array; NaN-filled if missing."""
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def pos_finite(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: finite and strictly positive."""
    return np.isfinite(arr) & (arr > 0)


def av_norm(df: pd.DataFrame) -> mcolors.Normalize:
    """Normalise Av to [0, 75th-percentile] so the bulk of the sample
    spreads across the full colourmap range."""
    av = col(df, AV_COL)
    fin = av[np.isfinite(av)]
    if fin.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    vmax = np.nanpercentile(fin, 75)
    if vmax <= 0:
        vmax = fin.max() if fin.max() > 0 else 1.0
    return mcolors.Normalize(vmin=0, vmax=vmax)


def build_masks(df: pd.DataFrame):
    """
    Returns four boolean arrays (aligned with df):
      is_strong – flag == 1, non-AGN
      is_weak   – flag == 2, non-AGN
      is_ul     – flag == 0, non-AGN
      is_agn    – agn_flag == 1 and detected (flag > 0)
    AGN are always excluded from the three detection/UL masks.
    """
    flag = col(df, "lya_detect_flag")
    agn  = col(df, "agn_flag").astype(bool)
    not_agn = ~agn
    is_strong = not_agn & (flag == 1)
    is_weak   = not_agn & (flag == 2)
    is_ul     = not_agn & (flag == 0)
    is_agn    = agn & ((flag == 1) | (flag == 2))
    return is_strong, is_weak, is_ul, is_agn


def get_ew_lya(df: pd.DataFrame) -> np.ndarray:
    """
    Best-available Lyα EW:
      - ew_muv  if beta_err is finite and ≤ 0.5
      - ew_f150 otherwise
    Only for detected sources (lya_detect_flag > 0).
    """
    ew_muv   = col(df, "ew_muv")
    ew_f150  = col(df, "ew_f150")
    beta_err = col(df, "beta_err")
    flag     = col(df, "lya_detect_flag")

    ew       = np.full(len(df), np.nan)
    detected = flag > 0

    use_muv  = detected & np.isfinite(ew_muv) & np.isfinite(beta_err) & (beta_err <= 0.5)
    use_f150 = detected & ~use_muv & np.isfinite(ew_f150)

    ew[use_muv]  = ew_muv[use_muv]
    ew[use_f150] = ew_f150[use_f150]
    return ew


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

from lifelines import KaplanMeierFitter
from scipy.stats import spearmanr, norm
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# KM-CORRECTED CENSORED CORRELATION  (Isobe, Feigelson & Nelson 1986)
# ══════════════════════════════════════════════════════════════════════════════

def km_corrected_ranks(y, censored):
    """
    Use the Kaplan-Meier estimator to assign expected ranks to censored
    and uncensored observations, following Isobe, Feigelson & Nelson (1986).

    For upper limits in y (censored=True), the KM survival function gives
    the probability that the true value exceeds each observed limit, which
    is used to compute expected fractional ranks.

    Parameters
    ----------
    y        : array-like  — observed y values (UL = the limit value)
    censored : bool array  — True where y is an upper limit

    Returns
    -------
    ranks : ndarray, shape (n,)  — KM-corrected expected ranks in [0, 1]
    """
    y        = np.asarray(y, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    n        = len(y)

    # lifelines KaplanMeierFitter treats upper limits as left-censored
    # when we flip the sign: fit on -y, event_observed = ~censored
    # (i.e. a detection in y is an "event" in survival terms)
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations       = -y,           # flip so ULs become left-censored
        event_observed  = ~censored,    # detected = event occurred
        label           = "KM"
    )

    # Evaluate survival function S(t) at each observed -y value
    # S(-y_i) = P(true value > -y_i) = P(true y < y_i)
    # So the expected rank of point i is  n * S(-y_i)
    ranks = np.empty(n)
    for i, yi in enumerate(y):
        # S(-yi) from the KM curve
        s = kmf.survival_function_at_times(-yi).values[0]
        ranks[i] = n * s

    return ranks


def kendall_tau_km(x, y, censored):
    """
    Censored Kendall's τ using KM-corrected ranks for the censored variable y.

    Procedure
    ---------
    1. Fit KM survival function on y (accounting for upper limits).
    2. Replace each y_i with its KM-expected rank E[rank(y_i)].
    3. Compute standard Spearman ρ on (x, km_rank(y)) as a proxy for τ.
       Then compute Kendall τ directly on the KM ranks.

    This follows the Schmitt (1985) / Isobe et al. (1986) approach and is
    the same underlying machinery as the IRAF STSDAS `bhkmethod`.

    Parameters
    ----------
    x        : array-like  — fully observed independent variable
    y        : array-like  — dependent variable, may contain upper limits
    censored : bool array  — True where y[i] is an upper limit

    Returns
    -------
    tau : float   — Kendall's τ on KM-corrected ranks
    p   : float   — two-sided p-value (normal approximation)
    rho : float   — Spearman ρ on KM-corrected ranks (bonus diagnostic)
    p_rho : float
    """
    x        = np.asarray(x,        dtype=float)
    y        = np.asarray(y,        dtype=float)
    censored = np.asarray(censored, dtype=bool)

    finite = np.isfinite(x) & np.isfinite(y)
    x, y, censored = x[finite], y[finite], censored[finite]
    n = len(x)

    if n < 3:
        return np.nan, np.nan, np.nan, np.nan

    # ── Step 1: KM-corrected ranks for y ──────────────────────────────────
    km_ranks = km_corrected_ranks(y, censored)

    # ── Step 2: Kendall τ on (x, km_ranks) ────────────────────────────────
    from scipy.stats import kendalltau as _kendalltau
    tau, _ = _kendalltau(x, km_ranks)

    # Normal approximation for p-value (valid n ≳ 10)
    v0 = 2 * (2 * n + 5) / (9 * n * (n - 1))
    z  = tau / np.sqrt(v0)
    p  = 2 * norm.sf(abs(z))

    # ── Bonus: Spearman on KM ranks ────────────────────────────────────────
    rho, p_rho = spearmanr(x, km_ranks)

    return tau, p, rho, p_rho


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CORRELATION FUNCTION  (drop-in replacement)
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlations(x, y_det, y_ul, det_mask, ul_mask):
    """
    Spearman ρ on detections only.
    KM-corrected Kendall τ on full sample (detections + upper limits).

    Parameters
    ----------
    x        : ndarray  — full array of x values (e.g. log fesc or xi_ion)
    y_det    : ndarray  — y values at detection positions (NaN elsewhere)
    y_ul     : ndarray  — y values at UL positions (NaN elsewhere)
    det_mask : bool arr — True where source is a detection
    ul_mask  : bool arr — True where source is an upper limit
    """

    # ── Spearman — detections only ─────────────────────────────────────────
    xs, ys = x[det_mask], y_det[det_mask]
    m      = np.isfinite(xs) & np.isfinite(ys)
    rho_det, p_rho_det = (
        spearmanr(xs[m], ys[m]) if m.sum() > 2 else (np.nan, np.nan)
    )

    # ── KM-corrected Kendall τ — full sample ──────────────────────────────
    all_mask     = det_mask | ul_mask
    x_all        = x[all_mask]
    y_all        = np.where(det_mask[all_mask], y_det[all_mask], y_ul[all_mask])
    censored_all = ul_mask[all_mask]

    tau, p_tau, rho_km, p_rho_km = kendall_tau_km(x_all, y_all, censored_all)

    n_det = det_mask.sum()
    n_ul  = ul_mask.sum()
    print(
        f"  Spearman (dets only):       ρ = {rho_det:.3f}  (p = {p_rho_det:.3g},  N = {m.sum()})\n"
        f"  KM-corrected Kendall τ:     τ = {tau:.3f}      (p = {p_tau:.3g},  "
        f"N_det = {n_det},  N_UL = {n_ul})\n"
        f"  KM-corrected Spearman ρ:    ρ = {rho_km:.3f}   (p = {p_rho_km:.3g})"
    )

    return rho_det, p_rho_det, tau, p_tau, rho_km, p_rho_km


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: paper-ready text
# ══════════════════════════════════════════════════════════════════════════════

def corr_text(rho, p_rho, tau, p_tau) -> str:
    def fp(p):
        if np.isnan(p):  return r"\text{--}"
        if p < 0.001:    return "<0.001"
        return f"{p:.3f}"
    r = r"\text{--}" if np.isnan(rho) else f"{rho:.2f}"
    t = r"\text{--}" if np.isnan(tau) else f"{tau:.2f}"
    return (
        rf"$\rho={r}$  ($p={fp(p_rho)}$)"
        "\n"
        rf"$\tau_\mathrm{{KM}}={t}$  ($p={fp(p_tau)}$)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_detections(ax, x, y, x_err, y_err, mask_strong, mask_weak,
                    av, norm, logx=False, label_ids=False, ids=None):
    """
    Plot strong (circle) and weak (diamond) detections coloured by Av.
    Both x and y error bars are drawn.
    Returns the scatter artist (for colourbar) or None.
    """
    cmap = f"cmr.{THEME}"
    sc   = None

    for mask, marker, label in [
        (mask_strong, M_STRONG, "Detection (strong)"),
        (mask_weak,   M_WEAK,   "Detection (weak)"),
    ]:
        ok = pos_finite(x[mask]) if logx else np.isfinite(x[mask])
        idx = np.where(mask)[0][ok]
        m2  = np.zeros(len(x), bool)
        m2[idx] = True
        m2 &= np.isfinite(y) & (y > 0)
        if not m2.any():
            continue

        # x errors — only pass if provided and valid
        xe = x_err[m2] if x_err is not None else None
        ye = y_err[m2] if y_err is not None else None

        ax.errorbar(x[m2], y[m2], xerr=xe, yerr=ye,
                    fmt="none", ecolor=P["neutral_grey"],
                    elinewidth=0.8, capsize=1.5, alpha=0.7, zorder=3)
        sc_new = ax.scatter(x[m2], y[m2],
                            c=av[m2], cmap=cmap, norm=norm,
                            s=MS, marker=marker,
                            edgecolor=C_DET, linewidth=0.5,
                            zorder=4, label=label)
        if sc is None:
            sc = sc_new
        if label_ids and ids is not None:
            _annotate_ids(ax, x[m2], y[m2], ids[m2], C_DET, logx)
    return sc


def draw_upper_limits(ax, x, y_ul, mask, logx=False, label_ids=False, ids=None):
    """Downward triangles for upper limits (open, C_UL colour)."""
    ok  = pos_finite(x[mask]) if logx else np.isfinite(x[mask])
    idx = np.where(mask)[0][ok]
    m2  = np.zeros(len(x), bool)
    m2[idx] = True
    m2 &= np.isfinite(y_ul) & (y_ul > 0)
    if not m2.any():
        return
    ax.scatter(x[m2], y_ul[m2],
               s=MS, marker=M_UL,
               facecolor="none", edgecolor=C_UL,
               linewidth=0.8, alpha=0.70, zorder=3,
               label="Upper limit")
    if label_ids and ids is not None:
        _annotate_ids(ax, x[m2], y_ul[m2], ids[m2], C_UL, logx)


def draw_agn(ax, x, y, is_agn, x_ok, label_ids=False, ids=None):
    """Star markers for AGN."""
    m = is_agn & x_ok & np.isfinite(y) & (y > 0)
    if not m.any():
        return
    ax.scatter(x[m], y[m],
               s=MS_AGN, marker=M_AGN,
               facecolor=C_AGN, edgecolor=C_AGN,
               linewidth=0.5, zorder=5, label="AGN")
    if label_ids and ids is not None:
        _annotate_ids(ax, x[m], y[m], ids[m], C_AGN, False)


def _annotate_ids(ax, x, y, ids, color, logx):
    for xi, yi, sid in zip(x, y, ids):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        if logx and xi <= 0:
            continue
        ax.annotate(str(int(sid)), xy=(xi, yi),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6, color=color, alpha=0.75, zorder=5)


def add_legend(ax, show_agn, loc="upper right"):
    handles = [
        Line2D([0], [0], marker=M_STRONG, color="w",
               markerfacecolor=P["c4"], markeredgecolor=C_DET,
               markersize=6, label="Strong det."),
        Line2D([0], [0], marker=M_WEAK, color="w",
               markerfacecolor=P["c4"], markeredgecolor=C_DET,
               markersize=6, label="Weak det."),
        Line2D([0], [0], marker=M_UL, color="w",
               markerfacecolor="none", markeredgecolor=C_UL,
               markersize=6, label="Upper limit"),
    ]
    if show_agn:
        handles.append(
            Line2D([0], [0], marker=M_AGN, color="w",
                   markerfacecolor=C_AGN, markeredgecolor=C_AGN,
                   markersize=8, label="AGN")
        )
    ax.legend(handles=handles, loc=loc, fontsize=7, handletextpad=0.3)

def draw_clipped_points(ax, x, y, mask,
                        xlim=None, ylim=None,
                        color="k",
                        size=45,
                        alpha=0.9,
                        zorder=6):
    """
    Draw directional arrow markers on axis edges for points outside xlim.

    Left-clipped  (x < xmin) → '<' marker pinned to xmin.
    Right-clipped (x > xmax) → '>' marker pinned to xmax.
    y values are preserved in both cases.
    """
    if xlim is None:
        return

    xmin, xmax = xlim

    base_kw = dict(s=size, facecolor=color, edgecolor=color,
                   alpha=alpha, clip_on=False, zorder=zorder)

    m_left  = mask & np.isfinite(x) & np.isfinite(y) & (x < xmin)
    m_right = mask & np.isfinite(x) & np.isfinite(y) & (x > xmax)

    if np.any(m_left):
        ax.scatter(np.full(m_left.sum(), xmin),  y[m_left],  marker="<", **base_kw)

    if np.any(m_right):
        ax.scatter(np.full(m_right.sum(), xmax), y[m_right], marker=">", **base_kw)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — L(Lyα) vs L(Hα): two-panel (uncorrected | dust-corrected)
# ══════════════════════════════════════════════════════════════════════════════

def plot_lya_vs_ha(df, is_strong, is_weak, is_ul, is_agn,
                   out_path, label_ids=False,
                   sphinx_x=None, sphinx_y=None,
                   show_agn=True):
    """
    Two-panel figure:
      Left  – L(Lyα) vs L(Hα) uncorrected
      Right – L(Lyα) vs L(Hα) dust-corrected  (same as previous single panel)
    Both panels share the y-axis.
    """
    lya_L     = col(df, "lya_lumin")
    lya_L_err = col(df, "lya_lumin_err")
    lya_L_ul  = col(df, "lya_lumin_upper_limit")
    ha_uncorr = col(df, "ha_lumin_uncorr")
    ha_uncorr_err = col(df, "ha_lumin_err_uncorr")  # use if present, else None
    ha_corr   = col(df, "ha_lumin_fullcorr")
    ha_corr_err = col(df, "ha_lumin_err_fullcorr")
    ids       = col(df, "ID")

    # If uncorr error column is all NaN, pass None so errorbar silently skips
    if not np.any(np.isfinite(ha_uncorr_err)):
        ha_uncorr_err = None

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9.0, 4.0), sharey=True,
        gridspec_kw={"wspace": 0.05}
    )

    xx = np.logspace(38, 45, 300)

    for ax, ha_L, ha_L_err, title, xlim in [
        (ax_left,  ha_uncorr, ha_uncorr_err, r"$L(\mathrm{H\alpha})_\mathrm{uncorr}\  [\mathrm{{erg\,s^{{-1}}}}]$",(1e41, 8e44),),
        (ax_right, ha_corr,   ha_corr_err,   r"$L(\mathrm{H\alpha})_\mathrm{corr}\  [\mathrm{{erg\,s^{{-1}}}}]$",(1e41, 8e44),),
    ]:
        # SPHINX background (only on corrected panel if provided)
        if ax is ax_right and sphinx_x is not None and sphinx_y is not None:
            m = pos_finite(sphinx_x) & pos_finite(sphinx_y)
            ax.scatter(sphinx_x[m], sphinx_y[m],
                       c=C_SPHINX, s=12, alpha=0.35, edgecolor="none", zorder=0,
                       label="SPHINX")

        # ── Simmonds+ comparison sample ────────────────────────────────
        # Only draw on the LEFT (uncorrected) panel
        if ax is ax_left:
            try:
                sim = pd.read_csv("/home/apatrick/P1/JELSDP/simmonds.csv")

                sim_ha      = pd.to_numeric(sim["Lha_erg_s"], errors="coerce").to_numpy()
                sim_ha_ep   = pd.to_numeric(sim["Lha_err_plus"], errors="coerce").to_numpy()
                sim_ha_em   = pd.to_numeric(sim["Lha_err_minus"], errors="coerce").to_numpy()

                sim_lya     = pd.to_numeric(sim["Llya_erg_s"], errors="coerce").to_numpy()
                sim_lya_ep  = pd.to_numeric(sim["Llya_err_plus"], errors="coerce").to_numpy()
                sim_lya_em  = pd.to_numeric(sim["Llya_err_minus"], errors="coerce").to_numpy()
                sim_z = pd.to_numeric(sim["z_sys"], errors="coerce").to_numpy()

                m_sim = (
                    np.isfinite(sim_ha) &
                    np.isfinite(sim_lya) &
                    (sim_ha > 0) &
                    (sim_lya > 0) &
                    (sim_z > 6.0)
                )

                if np.any(m_sim):

                    ax.errorbar(
                        sim_ha[m_sim],
                        sim_lya[m_sim],
                        xerr=[
                            sim_ha_em[m_sim],
                            sim_ha_ep[m_sim]
                        ],
                        yerr=[
                            sim_lya_em[m_sim],
                            sim_lya_ep[m_sim]
                        ],
                        fmt="none",
                        ecolor=P["mid_grey"],
                        elinewidth=0.5,
                        capsize=1.0,
                        alpha=0.4,
                        zorder=1,
                    )

                    ax.scatter(
                        sim_ha[m_sim],
                        sim_lya[m_sim],
                        s=28,
                        marker="s",
                        facecolor="none",
                        edgecolor=P["mid_grey"],
                        linewidth=0.5,
                        alpha=0.4,
                        zorder=2,
                        label="Simmonds et al. (2023)"
                    )

                    print(f"[INFO] Added {m_sim.sum()} Simmonds+ sources")

            except Exception as e:
                print(f"[WARN] Could not load Simmonds sample: {e}")

        # Case B line
        ax.plot(xx, xx * 8.7, color=P["c5"], lw=1.2, ls="-", alpha=0.85,
                label=r"Case B ($8.7\times$ L(Hα))")

        # Strong + weak detections
        for mask, marker, label in [
            (is_strong, M_STRONG, "Strong det."),
            (is_weak,   M_WEAK,   "Weak det."),
        ]:
            m = mask & pos_finite(ha_L) & pos_finite(lya_L)
            if not m.any():
                continue
            ax.errorbar(ha_L[m], lya_L[m],
                        xerr=ha_L_err[m] if ha_L_err is not None else None,
                        yerr=lya_L_err[m],
                        fmt="none", ecolor=P["neutral_grey"],
                        elinewidth=0.8, capsize=1.5, alpha=0.7, zorder=3)
            ax.scatter(ha_L[m], lya_L[m],
                       s=MS, marker=marker,
                       facecolor=C_DET, edgecolor=C_DET,
                       linewidth=0.6, zorder=4, label=label)
            if label_ids:
                _annotate_ids(ax, ha_L[m], lya_L[m], ids[m], C_DET, True)

        # AGN — star at (ha_L, lya_L), arrow from uncorr to corrected on right panel
        if show_agn:
            m_agn = is_agn & pos_finite(ha_L) & pos_finite(lya_L)
            if m_agn.any():
                if ax is ax_right:
                    ha_uncorr_ = col(df, "ha_lumin_uncorr")
                    for hu, hc, ly in zip(ha_uncorr_[m_agn],
                                          ha_corr[m_agn],
                                          lya_L[m_agn]):
                        ax.annotate("",
                            xy=(hu, ly), xytext=(hc, ly),
                            arrowprops=dict(arrowstyle="-|>", color=C_AGN,
                                            lw=0.8, mutation_scale=10, alpha=0.7),
                            zorder=4)
                ax.scatter(ha_L[m_agn], lya_L[m_agn],
                           s=MS_AGN, marker=M_AGN,
                           facecolor=C_AGN, edgecolor=C_AGN,
                           linewidth=0.5, zorder=5,
                           label="AGN (corr→uncorr)" if ax is ax_right else "AGN")
                if label_ids:
                    _annotate_ids(ax, ha_L[m_agn], lya_L[m_agn],
                                  ids[m_agn], C_AGN, True)

        # Upper limits
        m_ul = is_ul & pos_finite(ha_L) & pos_finite(lya_L_ul)
        if m_ul.any():
            ax.scatter(ha_L[m_ul], lya_L_ul[m_ul],
                       s=MS, marker=M_UL,
                       facecolor="none", edgecolor=C_UL,
                       linewidth=0.8, alpha=0.70, zorder=3,
                       label="Upper limit")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(3e41, 7e42)
        ax.set_xlabel(title, fontsize=12)
        ax.legend(loc="lower right", handletextpad=0.4, fontsize=7)

    ax_left.set_ylabel(r"$L(\mathrm{Ly\alpha})\ [\mathrm{erg\,s^{-1}}]$", fontsize=12)
    ax_right.tick_params(labelleft=False)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        p = out_path.replace(".pdf", ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  [OK] {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — fesc vs single property
# ══════════════════════════════════════════════════════════════════════════════

def plot_fesc_vs_property(df, is_strong, is_weak, is_ul, is_agn,
                          x_col, x_err_col, xlabel,
                          out_path,
                          logx=False, xlim=None, ylim=None,
                          label_ids=False,
                          sphinx_tab=None, sphinx_x_col=None, sphinx_y_col=None,
                          norm=None, show_agn=True):
    """
    f_esc (dust-corrected) vs a single galaxy property.
    Error bars on both axes for detections.
    """
    x      = col(df, x_col)
    x_err  = col(df, x_err_col) if x_err_col and x_err_col in df.columns else None
    y      = col(df, "fesc_lya_dustcorr")
    y_err  = col(df, "fesc_lya_dustcorr_err")
    y_ul   = col(df, "fesc_lya_dustcorr_ul")
    av     = col(df, AV_COL)
    ids    = col(df, "ID")

    if norm is None:
        norm = av_norm(df)

    x_ok     = pos_finite(x) if logx else np.isfinite(x)
    det_mask = (is_strong | is_weak) & x_ok & np.isfinite(y) & (y > 0)
    ul_mask  = is_ul & x_ok & np.isfinite(y_ul) & (y_ul > 0)

    print(f"\n  {x_col}: {det_mask.sum()} det, {ul_mask.sum()} UL")
    rho, p_rho, tau, p_tau, _, _ = compute_correlations(x, y, y_ul, det_mask, ul_mask)


    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    # SPHINX overlay
    if sphinx_tab is not None and sphinx_x_col and sphinx_y_col:
        if sphinx_x_col in sphinx_tab.columns and sphinx_y_col in sphinx_tab.columns:
            sx = pd.to_numeric(sphinx_tab[sphinx_x_col], errors="coerce").to_numpy()
            sy = pd.to_numeric(sphinx_tab[sphinx_y_col], errors="coerce").to_numpy()
            m  = np.isfinite(sx) & np.isfinite(sy) & (sy > 0)
            if logx: m &= sx > 0
            ax.scatter(sx[m], sy[m], c=C_SPHINX, s=12, alpha=0.35,
                       edgecolor="none", zorder=0, label="SPHINX")

    sc = draw_detections(ax, x, y, x_err, y_err,
                         det_mask & is_strong, det_mask & is_weak,
                         av, norm, logx=logx, label_ids=label_ids, ids=ids)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label=AV_LABEL, pad=0.02, shrink=0.85)

    if show_agn:
        draw_agn(ax, x, y, is_agn, x_ok, label_ids=label_ids, ids=ids)

    draw_upper_limits(ax, x, y_ul, ul_mask, logx=logx,
                      label_ids=label_ids, ids=ids)

    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$f^\mathrm{Ly\alpha}_\mathrm{esc}$")
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

    ax.text(0.05, 0.05, corr_text(rho, p_rho, tau, p_tau),
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    add_legend(ax, show_agn, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — 2×3 panel of fesc vs properties
# ══════════════════════════════════════════════════════════════════════════════

# Properties shown in the panel.
# Tuple: (x_col, x_err_col, xlabel, logx, xlim, agn_in_row)
# agn_in_row controls per-panel AGN visibility:
#   True  → show AGN if the global show_agn flag is also True
#   False → never show AGN in this panel
PANEL_PROPS = [
    # ── row 0 (AGN shown if show_agn=True) ──────────────────────────────────
    ("m_uv_ab_uncorr",  "m_uv_ab_err_uncorr", r"$M_\mathrm{UV}$ (AB, uncorr)", False, None,       True),
    ("beta",            "beta_err",            r"$\beta_\mathrm{UV}$",          False, (-3.8, -0.5),       True),
    ("ew_lya",          None,                  r"$\mathrm{EW}_\mathrm{Ly\alpha,rest}\ [\AA]$", False, (0, 150), True),
    # ── row 1 (AGN never shown) ──────────────────────────────────────────────
    ("E_BV",            None,                  r"$E(B{-}V)$",                   False, None,       False),
    ("m_star_50",       None,                  r"$\log\,M_\star\ [M_\odot]$",   False, None,       False),
    ("ssfr_100myr_50",  None,                  r"$\mathrm{sSFR}_{100\,\mathrm{Myr}}$", False, None, False),
]


def plot_fesc_panel(df, is_strong, is_weak, is_ul, is_agn,
                    out_path, label_ids=False,
                    sphinx_tab=None, show_agn=True):
    """
    2×3 panel of fesc vs galaxy properties.
    - Panels in each row share the y-axis (tick labels only on leftmost).
    - One Av colourbar on the right of each row.
    - AGN shown only in top row (controlled via PANEL_PROPS agn_in_row flag
      AND the global show_agn argument).
    - A legend in the first panel of each row.
    - x and y error bars on all detections.
    """
    ncols = 3
    nrows = 2

    norm = av_norm(df)
    av   = col(df, AV_COL)
    ids  = col(df, "ID")
    y      = col(df, "fesc_lya_dustcorr")
    y_err  = col(df, "fesc_lya_dustcorr_err")
    y_ul   = col(df, "fesc_lya_dustcorr_ul")

    # GridSpec: nrows × (ncols panels + 1 colourbar column)
    fig = plt.figure(figsize=(4.0 * ncols, 4.0 * nrows))
    gs  = GridSpec(nrows, ncols + 1,
                   figure=fig,
                   width_ratios=[1, 1, 1, 0.06],
                   wspace=0.0,
                   hspace=0.15)

    # Build axes — share y within each row
    axes      = []
    cbar_axes = []
    for row in range(nrows):
        row_axes = []
        for c in range(ncols):
            if c == 0:
                ax = fig.add_subplot(gs[row, c])
            else:
                ax = fig.add_subplot(gs[row, c], sharey=row_axes[0])
                ax.tick_params(labelleft=False)
            row_axes.append(ax)
        axes.append(row_axes)
        cbar_axes.append(fig.add_subplot(gs[row, ncols]))

    sc_per_row = [None, None]

    for idx, (x_col, x_err_col, xlabel, logx, xlim, agn_in_row) in enumerate(PANEL_PROPS):
        row  = idx // ncols
        cidx = idx % ncols
        ax   = axes[row][cidx]

        x     = col(df, x_col)
        x_err = col(df, x_err_col) if x_err_col and x_err_col in df.columns else None

        x_ok     = pos_finite(x) if logx else np.isfinite(x)
        det_mask = (is_strong | is_weak) & x_ok & np.isfinite(y) & (y > 0)
        ul_mask  = is_ul & x_ok & np.isfinite(y_ul) & (y_ul > 0)

        print(f"  [{x_col}] {det_mask.sum()} det, {ul_mask.sum()} UL")
        rho, p_rho, tau, p_tau, _, _ = compute_correlations(x, y, y_ul, det_mask, ul_mask)

        # SPHINX overlay (E_BV panel only)
        if sphinx_tab is not None and x_col == "E_BV":
            if "E_BV_median" in sphinx_tab.columns and "fesc_lya" in sphinx_tab.columns:
                sx = pd.to_numeric(sphinx_tab["E_BV_median"], errors="coerce").to_numpy()
                sy = pd.to_numeric(sphinx_tab["fesc_lya"],    errors="coerce").to_numpy()
                m  = np.isfinite(sx) & np.isfinite(sy) & (sy > 0)
                ax.scatter(sx[m], sy[m], c=C_SPHINX, s=8, alpha=0.25,
                           edgecolor="none", zorder=0, label="SPHINX")

        # Detections
        sc = draw_detections(ax, x, y, x_err, y_err,
                             det_mask & is_strong, det_mask & is_weak,
                             av, norm, logx=logx, label_ids=label_ids, ids=ids)
        if sc is not None and sc_per_row[row] is None:
            sc_per_row[row] = sc

        # AGN — only in top row, and only if global show_agn is True
        if show_agn and agn_in_row:
            draw_agn(ax, x, y, is_agn, x_ok, label_ids=label_ids, ids=ids)

        # Upper limits
        draw_upper_limits(ax, x, y_ul, ul_mask, logx=logx,
                          label_ids=label_ids, ids=ids)

        # Axes formatting
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        if xlim:
            ax.set_xlim(*xlim)
        
        # Mark sources clipped by x-axis limits
        if xlim is not None:

            # detections clipped by x limits
            draw_clipped_points(ax, x, y,
                (is_strong | is_weak | is_agn) & np.isfinite(x) & np.isfinite(y),
                xlim=xlim,
                color=C_DET if x_col != "beta" else P["c3"],
                size=45,
            )

            # upper limits clipped by x limits
            draw_clipped_points(ax, x, y_ul,
                is_ul & np.isfinite(x) & np.isfinite(y_ul),
                xlim=xlim,
                color=C_UL,
                size=45,
            )
     


        if cidx == 0:
            ax.set_ylabel(r"$f^\mathrm{Ly\alpha}_\mathrm{esc}$", fontsize=12)

        # Correlation text
        ax.text(0.50, 0.03, corr_text(rho, p_rho, tau, p_tau),
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

        # Legend in leftmost panel of each row only
        if cidx == 0:
            leg = add_legend(ax, show_agn=show_agn and agn_in_row, loc="upper right")
            if leg is not None: # won't let me add fontsiex in above
                for text in leg.get_texts():
                    text.set_fontsize(12)

    # Colourbars
    for row in range(nrows):
        if sc_per_row[row] is not None:
            fig.colorbar(sc_per_row[row], cax=cbar_axes[row], label=AV_LABEL)
        else:
            cbar_axes[row].set_visible(False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SPHINX COLUMN PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_sphinx_columns(sphinx_tab: pd.DataFrame):
    """Print SPHINX table columns so you can identify new overlay possibilities."""
    print("\n" + "=" * 60)
    print("SPHINX TABLE COLUMNS")
    print("=" * 60)
    for i, c in enumerate(sphinx_tab.columns):
        dtype = sphinx_tab[c].dtype
        n_fin = int(pd.to_numeric(sphinx_tab[c], errors="coerce").notna().sum())
        print(f"  {i:3d}  {c:<35s}  dtype={str(dtype):<10s}  n_finite={n_fin}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="fesc plots from merged CSV")
    p.add_argument("--merged-csv",   required=True,
                   help="Path to final_merged CSV")
    p.add_argument("--out-dir",      required=True,
                   help="Output directory for plots")
    p.add_argument("--caseb",        type=float, default=8.7,
                   help="Case B Lyα/Hα ratio (default 8.7)")
    p.add_argument("--remove-agn",   action="store_true",
                   help="Exclude AGN from all plots")
    p.add_argument("--sphinx-table", default=None,
                   help="Optional SPHINX FITS table for background overlay")
    p.add_argument("--label-ids",    action="store_true",
                   help="Annotate each point with its source ID")
    p.add_argument("--dust-tag",     default="",
                   help="Tag appended to output filenames (e.g. eta_1)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag      = f"_{args.dust_tag}" if args.dust_tag else ""
    show_agn = not args.remove_agn

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.merged_csv)
    df["ID"] = df["ID"].astype(int)
    print(f"[INFO] Loaded {len(df)} rows from {args.merged_csv}")
    print(f"[INFO] CSV columns: {list(df.columns)}")

    # ── Convenience columns ───────────────────────────────────────────────────
    # Use E_BV from CSV directly; if absent fall back to 'ebv'
    if "E_BV" not in df.columns and "ebv" in df.columns:
        df["E_BV"] = df["ebv"]

    # Best-available Lyα EW
    df["ew_lya"] = get_ew_lya(df)

    # ── Masks ─────────────────────────────────────────────────────────────────
    is_strong, is_weak, is_ul, is_agn = build_masks(df)
    print(f"[INFO] Strong={is_strong.sum()}  Weak={is_weak.sum()}  "
          f"UL={is_ul.sum()}  AGN={'excluded' if args.remove_agn else is_agn.sum()}")

    # Global Av normalisation (consistent across all plots)
    norm = av_norm(df)

    # ── SPHINX ────────────────────────────────────────────────────────────────
    sphinx_tab = None
    sphinx_lya = sphinx_ha = None
    if args.sphinx_table:
        try:
            sphinx_tab = AstropyTable.read(args.sphinx_table).to_pandas()
            #print_sphinx_columns(sphinx_tab)
            sphinx_lya = sphinx_tab["obs_lya"].to_numpy(float)
            sphinx_ha  = sphinx_tab["int_ha"].to_numpy(float)
            print(f"[INFO] SPHINX table loaded: {len(sphinx_tab)} rows")
        except Exception as e:
            print(f"[WARN] Could not load SPHINX table: {e}")

    # ── Plot 1: L(Lyα) vs L(Hα) — two panels ─────────────────────────────────
    print("\n[PLOT] L(Lyα) vs L(Hα) [two-panel: uncorr | corr]")
    plot_lya_vs_ha(
        df, is_strong, is_weak, is_ul, is_agn,
        out_path=os.path.join(args.out_dir, f"Lya_vs_Ha{tag}.pdf"),
        label_ids=args.label_ids,
        sphinx_x=sphinx_ha,
        sphinx_y=sphinx_lya,
        show_agn=show_agn,
    )

    # ── Plot 2: fesc vs individual properties ──────────────────────────────────
    # (x_col, x_err_col, xlabel, logx, xlim, ylim, sphinx_x_col, sphinx_y_col)
    solo_props = [
        ("E_BV",           None,                r"$E(B{-}V)$",                                 False, None,     None, "E_BV_median", "fesc_lya"),
        ("av_50",          None,                r"$A_V$",                                      False, None,     None, None,          None),
        ("beta",           "beta_err",          r"$\beta_\mathrm{UV}$",                        False, None,     None, None,          None),
        ("m_uv_ab_uncorr", "m_uv_ab_err_uncorr",r"$M_\mathrm{UV}$ (AB, uncorr)",              False, None,     None, None,          None),
        ("ssfr_10myr_50",  None,                r"$\mathrm{sSFR}_{10\,\mathrm{Myr}}$",        False, None,     None, None,          None),
        ("ssfr_100myr_50", None,                r"$\mathrm{sSFR}_{100\,\mathrm{Myr}}$",       False, None,     None, None,          None),
        ("m_star_50",      None,                r"$M_\star\ [M_\odot]$",                       False, None,     None, None,          None),
        ("ew_lya",         None,                r"$\mathrm{EW}_\mathrm{Ly\alpha,rest}\ [\AA]$",False, (0, 250), None, None,          None),
    ]

    for (x_col, x_err_col, xlabel, logx, xlim, ylim,
         sph_x_col, sph_y_col) in solo_props:

        if x_col not in df.columns:
            print(f"[SKIP] {x_col} not in CSV")
            continue

        safe = x_col.replace("_50", "").replace("_ab_uncorr", "")
        out  = os.path.join(args.out_dir, f"fesc_vs_{safe}{tag}.pdf")
        print(f"\n[PLOT] fesc vs {x_col}")

        plot_fesc_vs_property(
            df, is_strong, is_weak, is_ul, is_agn,
            x_col=x_col, x_err_col=x_err_col, xlabel=xlabel,
            out_path=out,
            logx=logx, xlim=xlim, ylim=ylim,
            label_ids=args.label_ids,
            sphinx_tab=sphinx_tab if sphinx_tab is not None else None,
            sphinx_x_col=sph_x_col,
            sphinx_y_col=sph_y_col,
            norm=norm,
            show_agn=show_agn,
        )

    # ── Plot 3: 2×3 panel ──────────────────────────────────────────────────────
    print("\n[PLOT] Panel plot (2×3)")
    plot_fesc_panel(
        df, is_strong, is_weak, is_ul, is_agn,
        out_path=os.path.join(args.out_dir, f"fesc_panel{tag}.pdf"),
        label_ids=args.label_ids,
        sphinx_tab=sphinx_tab,
        show_agn=show_agn,
    )

    print("\n[DONE] All plots saved.")


if __name__ == "__main__":
    main()


"""
# ── example command ───────────────────────────────────────────────────────────
 
 python fesc_plots.py \
   --merged-csv /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_merged_eta_1.csv \
   --out-dir    /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/fesc_plots/ \
   --remove-agn \
   --sphinx-table /home/apatrick/P1/JELSDP/sphinx_lya_ha_fesc_table.fits \
   --label-ids \
   --dust-tag eta_1 

python fesc_plots.py \
   --merged-csv /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/final_merged_eta_1.csv \
   --out-dir    /cephfs/apatrick/musecosmos/scripts/sample_select/outputs/fesc_plots/ \
   --sphinx-table /home/apatrick/P1/JELSDP/sphinx_lya_ha_fesc_table.fits \
   --dust-tag eta_1
"""
