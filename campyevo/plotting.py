"""Figures. Colours follow the manuscript: no heating blue, 2-log orange,
4-log green, 6-log red."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .analysis import survival_band

COLORS = {"none": "#1f77b4", "2log": "#ff7f0e", "4log": "#2ca02c", "6log": "#d62728"}
LABELS = {"none": "No heating", "2log": "2-log reduction", "4log": "4-log reduction",
          "6log": "6-log reduction"}
ORDER = ["none", "2log", "4log", "6log"]


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42,
        "svg.fonttype": "none",          # text stays text in SVG (editable in PowerPoint)
    })


def savefig(fig, name: str, outdir: str | Path | None = None):
    """Save as PDF (vector, for submission), PNG (300 dpi) and SVG (vector, for PowerPoint;
    notebook 04 puts the SVG on a slide, where "Convert to Shape" makes every element editable)."""
    from . import FIGURES
    outdir = Path(outdir) if outdir is not None else FIGURES
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")


def cov_ellipse(ax, mu, cov, n_sd=2.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_sd * np.sqrt(vals[1]), 2 * n_sd * np.sqrt(vals[0])
    ax.add_patch(Ellipse(mu, w, h, angle=ang, fill=False, **kw))


# ---------------------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------------------

def fig_trajectories(results: dict, chains=(0, 1, 2), conds=("2log", "4log", "6log"),
                     title_prefix: str = "", log_axes: bool = False):
    """Fig. 3: a few individual chains in the (delta, p) plane.

    Dot = start, triangle = end of a surviving chain, cross = extinction.
    """
    fig, axes = plt.subplots(1, len(conds), figsize=(3.3 * len(conds), 3.1), sharex=True, sharey=True)
    cmap = plt.get_cmap("Dark2")
    for ax, c in zip(np.atleast_1d(axes), conds):
        res = results[c]
        for i, k in enumerate(chains):
            tr = res.traj[k]
            ok = ~np.isnan(tr[:, 0])
            x, y = (tr[ok, 0], tr[ok, 1]) if log_axes else (np.exp(tr[ok, 0]), np.exp(tr[ok, 1]))
            col = cmap(i)
            ax.plot(x, y, lw=0.6, color=col)
            ax.plot(x[0], y[0], "o", ms=4, color=col)
            if ok.all():
                ax.plot(x[-1], y[-1], ">", ms=6, color=col)
            else:
                ax.plot(x[-1], y[-1], "x", ms=7, mew=1.5, color=col)
        ax.set_title(f"{title_prefix}{LABELS[c]}")
        ax.set_xlabel("ln δ" if log_axes else "δ (min)")
    np.atleast_1d(axes)[0].set_ylabel("ln p" if log_axes else "p")
    fig.tight_layout()
    return fig


def fig_final_params(results: dict, post=None, conds=ORDER, max_points: int = 3000, seed: int = 0):
    """Fig. 4: surviving parameter couples after the last cycle, with the path of the means."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for c in conds:
        if c not in results:
            continue
        X = results[c].final_params()
        if len(X) > max_points:
            X = X[rng.choice(len(X), max_points, replace=False)]
        ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.15, color=COLORS[c], lw=0)
    for c in conds:
        if c not in results:
            continue
        sm = results[c].summary()
        ax.plot(sm["lndelta_mean"], sm["lnp_mean"], color=COLORS[c], lw=1.6, label=LABELS[c])
        ax.plot(sm["lndelta_mean"].iloc[0], sm["lnp_mean"].iloc[0], "o", color=COLORS[c], ms=5)
        ax.plot(sm["lndelta_mean"].iloc[-1], sm["lnp_mean"].iloc[-1], ">", color=COLORS[c], ms=7,
                mec="k", mew=0.5)
    if post is not None:
        mu = post.mu_inter.mean(axis=0)
        cov = post.cov_inter.mean(axis=0)
        cov_ellipse(ax, mu, cov, 2.0, ls="--", lw=1, color="0.3")
        ax.plot([], [], ls="--", color="0.3", label="Inter-strain variability (±2 SD)")
    ax.set_xlabel("ln δ")
    ax.set_ylabel("ln p")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig


def fig_means_over_cycles(results: dict, conds=ORDER, cols=("lndelta_mean", "lnp_mean"),
                          ylabels=("Mean ln δ", "Mean ln p")):
    """Fig. 5: mean ln delta and ln p of surviving chains over cycles."""
    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 3.0))
    for ax, col, yl in zip(np.atleast_1d(axes), cols, ylabels):
        for c in conds:
            if c in results:
                sm = results[c].summary()
                ax.plot(sm["cycle"], sm[col], color=COLORS[c], label=LABELS[c])
        ax.set_xlabel("Heating–regrowth cycle")
        ax.set_ylabel(yl)
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_survival_curves(results: dict, conds=ORDER, t_max: float = 12.0, q=(0.005, 0.5, 0.995)):
    """Fig. 6: median and 99 % range of survival curves implied by the surviving parameters."""
    times = np.linspace(0, t_max, 121)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for c in conds:
        if c not in results:
            continue
        X = results[c].final_params()
        if len(X) == 0:
            continue
        lo, med, hi = survival_band(X, times, q)
        ax.plot(times, med, color=COLORS[c], label=LABELS[c])
        ax.plot(times, lo, color=COLORS[c], ls="--", lw=0.8)
        ax.plot(times, hi, color=COLORS[c], ls="--", lw=0.8)
    ax.set_ylim(-10.5, 0.5)
    ax.set_xlabel("Heating time at 55 °C (min)")
    ax.set_ylabel("log10 (N / N0)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def fig_survival_ratio(results: dict, conds=ORDER):
    """Fig. 7: share of chains (lineages) still alive after each cycle."""
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for c in conds:
        if c in results:
            ax.plot(results[c].survival_ratio(), color=COLORS[c], label=LABELS[c])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Heating–regrowth cycle")
    ax.set_ylabel("Share of lineages surviving")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity figures
# ---------------------------------------------------------------------------

def fig_sensitivity(results: dict, values, conds=("2log", "4log", "6log"), value_label: str = "",
                    col: str = "lndelta_mean", ylabel: str = "Mean ln δ", fmt=str, cmap: str = "viridis",
                    show_survival: bool = True):
    """Rows: mean ln delta (and survival ratio) over cycles; columns: heating condition.
    results keyed by (value, cond)."""
    nrow = 2 if show_survival else 1
    fig, axes = plt.subplots(nrow, len(conds), figsize=(3.3 * len(conds), 2.8 * nrow),
                             sharex=True, squeeze=False)
    cm = plt.get_cmap(cmap)
    cols = [cm(i / max(1, len(values) - 1) * 0.9) for i in range(len(values))]
    for j, c in enumerate(conds):
        for v, colr in zip(values, cols):
            res = results.get((v, c))
            if res is None:
                continue
            sm = res.summary()
            axes[0, j].plot(sm["cycle"], sm[col], color=colr, label=fmt(v))
            if show_survival:
                axes[1, j].plot(sm["cycle"], sm["survival_ratio"], color=colr)
        axes[0, j].set_title(LABELS[c])
        axes[-1, j].set_xlabel("Heating–regrowth cycle")
    axes[0, 0].set_ylabel(ylabel)
    if show_survival:
        axes[1, 0].set_ylabel("Share of lineages surviving")
        for j in range(len(conds)):
            axes[1, j].set_ylim(-0.03, 1.03)
    axes[0, 0].legend(title=value_label, frameon=False, fontsize=7, title_fontsize=7)
    fig.tight_layout()
    return fig



# ===========================================================================
# Revised main figures (microbiology-oriented)
#
# Units: delta is shown in minutes (time per log10 reduction, averaged over a
# 6-log reduction, i.e. comparable to a D-value) and p as the curve shape
# (p > 1 shoulder, p < 1 tail), on their natural scales.
# Heating intensity is an ordered variable, so it gets an ordinal ramp of one
# hue (validated with the dataviz validator, --ordinal); "no heating" is the
# neutral reference in gray.
# ===========================================================================

HEAT = {"none": "#8a8984", "2log": "#ee8f55", "4log": "#cc4d1a", "6log": "#7f270a"}
HEAT_LABEL = {"none": "No heating", "2log": "2-log heating", "4log": "4-log heating",
              "6log": "6-log heating"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STRAIN_DOT = "#b9b8b2"


def setup_style_v2():
    setup_style()
    plt.rcParams.update({
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.6, "legend.frameon": False,
    })


DELTA_LABEL = "δ (min): time to 6-log reduction ÷ 6"
DELTA_SHORT = "δ (min)"
P_LABEL = "Curve shape, p\n(>1 shoulder, <1 tail)"


def _shape_guides(ax, xmax):
    ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
    ax.text(xmax, 1.0, "  log-linear (p = 1)", va="center", ha="left", fontsize=7, color=INK2,
            clip_on=False)


def _strain_background(ax, sm):
    if sm is not None:
        ax.scatter(np.exp(sm[:, 0]), np.exp(sm[:, 1]), s=9, color=STRAIN_DOT, lw=0, zorder=1)


def _hdr_contour(ax, X, color, levels=(0.5, 0.9), natural=True, lw=(1.6, 0.9), fill=True):
    """Highest-density regions of a 2-D sample (kernel density estimate)."""
    from scipy.stats import gaussian_kde
    pts = np.exp(X) if natural else X
    if len(pts) < 20:
        return
    kde = gaussian_kde(pts.T)
    dens = kde(pts.T)
    thr = [np.quantile(dens, 1 - lv) for lv in levels]
    x0, x1 = np.quantile(pts[:, 0], [0.001, 0.999]); y0, y1 = np.quantile(pts[:, 1], [0.001, 0.999])
    dx, dy = (x1 - x0) * 0.25, (y1 - y0) * 0.25
    gx, gy = np.meshgrid(np.linspace(x0 - dx, x1 + dx, 160), np.linspace(y0 - dy, y1 + dy, 160))
    gz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    order = np.argsort(thr)
    thr_sorted = [thr[i] for i in order]
    if fill:
        ax.contourf(gx, gy, gz, levels=[thr_sorted[0], gz.max() * 1.01], colors=[color], alpha=0.12, zorder=2)
    for t, w in zip(thr, lw):
        ax.contour(gx, gy, gz, levels=[t], colors=[color], linewidths=w, zorder=3)


def _halo(t, w: float = 2.6):
    """White outline around a text so that lines and points behind it never hide it."""
    from matplotlib import patheffects as pe
    t.set_path_effects([pe.withStroke(linewidth=w, foreground="white")])
    return t


def _finish_row(fig, keys, ncol, xlabel=None):
    """Tight layout, then one shared x label (optional) and the legend under the panels,
    at fixed distances in inches so that nothing overlaps whatever the figure height."""
    h = fig.get_figheight()
    leg = 0.30 / h
    lab = 0.30 / h if xlabel else 0.0
    fig.tight_layout(rect=(0, leg + lab, 1, 1))
    if xlabel:
        size = plt.rcParams["axes.labelsize"]
        fig.supxlabel(xlabel, y=leg + 0.03 / h, va="bottom", color=INK,
                      fontsize=size if isinstance(size, (int, float)) else 10)
    fig.legend(handles=keys, loc="lower center", ncol=ncol, fontsize=7.5, bbox_to_anchor=(0.5, 0.0),
               frameon=False)


def _shared_xlabel(fig, label):
    """One x-axis label under a row of panels (instead of repeating it on each panel)."""
    fig.supxlabel(label, fontsize=plt.rcParams["axes.labelsize"] if isinstance(
        plt.rcParams["axes.labelsize"], (int, float)) else 10, y=0.0, color=INK)


def _representative_lineage(res) -> int:
    """Index of the lineage alive at the last cycle whose final (ln delta, ln p) is closest
    (in SD units) to the median over all surviving lineages."""
    alive = np.flatnonzero(res.alive[:, -1])
    if len(alive) == 0:
        return 0
    X = res.traj[alive, -1, :]
    z = (X - np.median(X, axis=0)) / X.std(axis=0).clip(1e-9)
    return int(alive[np.argmin((z ** 2).sum(axis=1))])


def fig3_example_lineages(results_atcc: dict, post, conds=("2log", "4log", "6log"),
                          xlim=(0.5, 5.0), ylim=(0.35, 3.6), marks=(10, 50)):
    """Fig. 3 (revised): one example lineage started from ATCC 33560 for each heating
    intensity, drawn light-to-dark from cycle 0 to cycle 100 on the plane of the measured
    strains (gray dots). The example is the surviving lineage whose cycle-100 parameters
    are closest to the median of all surviving lineages (so it is typical, not extreme);
    the share of lineages lost is written in the panel."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from .analysis import strain_means
    sm = strain_means(post)
    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 2.95), sharex=True, sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        _strain_background(ax, sm)
        res = results_atcc[c]
        k = _representative_lineage(res)
        tr = np.exp(res.traj[k])
        ok = ~np.isnan(tr[:, 0])
        xy = tr[ok]
        cmap = LinearSegmentedColormap.from_list("h", ["#f6d2bd", HEAT[c]])
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segs, colors=cmap(np.linspace(0, 1, len(segs))), linewidths=1.1, zorder=3)
        ax.add_collection(lc)
        ax.plot(*xy[0], marker="o", ms=7, color=INK, mec="white", mew=0.8, zorder=6)
        ax.plot(*xy[-1], marker="^", ms=8, color=HEAT[c], mec="white", mew=0.8, zorder=6)
        for m in marks:
            if m < len(xy):
                ax.plot(*xy[m], marker="o", ms=3.5, color=HEAT[c], mec="white", mew=0.5, zorder=5)
                _halo(ax.annotate(f"{m}", xy[m], xytext=(4, 3), textcoords="offset points", fontsize=7,
                                  color=INK, zorder=7))
        _halo(ax.annotate("100", xy[-1], xytext=(6, -2), textcoords="offset points", fontsize=7, color=INK,
                          zorder=7))
        lost = 100 * (1 - res.survival_ratio()[-1])
        _halo(ax.text(0.03, 0.97, f"{lost:.0f}% of lineages lost\nwithin 100 cycles", transform=ax.transAxes,
                      ha="left", va="top", fontsize=7, color=INK2))
        ax.set_title(HEAT_LABEL[c], color=INK, loc="left", fontsize=9)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
    np.atleast_1d(axes)[0].set_ylabel(P_LABEL)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], marker="o", ls="", color=INK, ms=6, label="Start: ATCC 33560"),
            Line2D([], [], color=HEAT["4log"], lw=1.1, label="Path of one lineage (light → dark = cycle 0 → 100; numbers = cycle)"),
            Line2D([], [], marker="o", ls="", color=STRAIN_DOT, ms=4, label="Measured strains (n = 50)")]
    _finish_row(fig, keys, 3, DELTA_LABEL)
    return fig


def fig4_where_lineages_end(results: dict, post, conds=("none", "2log", "4log", "6log"),
                            xlim=(0.4, 6.0), ylim=(0.35, 3.6)):
    """Fig. 4 (revised): where the surviving lineages are after 100 cycles (50 % and 90 % regions),
    against the measured strains and the range of present-day strains."""
    from .analysis import strain_means
    sm = strain_means(post)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, post.n_draws, 4000)
    base = np.array([rng.multivariate_normal(post.mu_inter[i], post.cov_inter[i]) for i in idx])
    fig, axes = plt.subplots(1, len(conds), figsize=(2.75 * len(conds), 2.9), sharex=True, sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        _strain_background(ax, sm)
        _hdr_contour(ax, base, "#6f6e69", levels=(0.9,), lw=(0.8,), fill=False)
        X = results[c].final_params()
        _hdr_contour(ax, X, HEAT[c])
        m0 = np.exp(results[c].traj[:, 0, :].mean(axis=0))
        m1 = np.exp(np.nanmean(results[c].traj[:, -1, :], axis=0))
        if c != "none":
            ax.annotate("", xy=m1, xytext=m0, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0), zorder=6)
        n_alive = len(X)
        ax.set_title(f"{HEAT_LABEL[c]}", color=INK, loc="left", fontsize=9)
        _halo(ax.text(0.97, 0.03, f"{100 * n_alive / results[c].n_chains:.0f}% of lineages\nsurvived",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2))
        ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    np.atleast_1d(axes)[0].set_ylabel(P_LABEL)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    keys = [Line2D([], [], marker="o", ls="", color=STRAIN_DOT, ms=4, label="Measured strains (n = 50)"),
            Line2D([], [], color="#6f6e69", lw=0.8, label="Range of present-day strains (90 %)"),
            Patch(facecolor="#cc4d1a", alpha=0.25, edgecolor="#cc4d1a",
                  label="Surviving lineages after 100 cycles (50 % and 90 %)"),
            Line2D([], [], color=INK, lw=1.0, marker=">", ms=4, label="Shift of the mean")]
    _finish_row(fig, keys, 4, DELTA_LABEL)
    return fig


def fig6_treatment_effect(results_inter: dict, results_atcc: dict, T: dict, conds=("2log", "4log", "6log")):
    """Fig. 6 (revised): (a) log reduction that the cycling treatment still achieves on the
    surviving lineages; (b) share of lineages not yet lost."""
    from .analysis import reduction_over_cycles
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    labels_y = {}
    for c in conds:
        r = reduction_over_cycles(results_inter[c], T[c])
        a.fill_between(r["cycle"], r["q25"], r["q75"], color=HEAT[c], alpha=0.18, lw=0)
        a.plot(r["cycle"], r["q50"], color=HEAT[c], lw=1.8)
        ra = reduction_over_cycles(results_atcc[c], T[c])
        a.plot(ra["cycle"], ra["q50"], color=HEAT[c], lw=1.0, ls=(0, (4, 2)))
        labels_y[c] = r["q50"].iloc[-1]
        b.plot(results_inter[c].survival_ratio() * 100, color=HEAT[c], lw=1.8, label=HEAT_LABEL[c])
        b.plot(results_atcc[c].survival_ratio() * 100, color=HEAT[c], lw=1.0, ls=(0, (4, 2)))
    # direct labels at the right end, nudged apart so they never overlap
    ys = sorted(labels_y.items(), key=lambda kv: kv[1])
    placed = []
    for c, y in ys:
        if placed and y - placed[-1] < 0.35:
            y = placed[-1] + 0.35
        placed.append(y)
        a.annotate(HEAT_LABEL[c].replace(" heating", ""), (1.01, y), xycoords=("axes fraction", "data"),
                   color=INK, fontsize=7.5, va="center")
    a.set_xlabel("Heating–regrowth cycle")
    a.set_ylabel("log10 reduction achieved\nby the same treatment")
    a.set_ylim(bottom=0); a.set_xlim(0, None)
    a.set_title("(a) Effect of the treatment on surviving lineages", loc="left", fontsize=9, color=INK)
    ys = sorted(((c, results_inter[c].survival_ratio()[-1] * 100) for c in conds), key=lambda kv: kv[1])
    placed = []
    for c, y in ys:
        if placed and y - placed[-1] < 6:
            y = placed[-1] + 6
        placed.append(y)
        b.annotate(HEAT_LABEL[c].replace(" heating", ""), (1.01, y), xycoords=("axes fraction", "data"),
                   color=INK, fontsize=7.5, va="center")
    b.set_xlabel("Heating–regrowth cycle")
    b.set_ylabel("Lineages not lost (%)")
    b.set_ylim(-2, 102)
    b.set_title("(b) Lineages surviving", loc="left", fontsize=9, color=INK)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=INK2, lw=1.8, label="Start: random present-day strain (median, IQR)"),
            Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label="Start: ATCC 33560 (median)")]
    _finish_row(fig, keys, 2)
    return fig


def fig5_survival_curves_before_after(results_inter: dict, post, T: dict, conds=("2log", "4log", "6log"),
                                      t_max: float = 12.0, q=(0.05, 0.5, 0.95), reference=None,
                                      reference_label: str = "ATCC 33560 (start)"):
    """Fig. 5 (revised): survival curves of present-day strains (gray) and of the lineages
    surviving 100 cycles (color); vertical line = heating time used in each cycle."""
    times = np.linspace(0, t_max, 121)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, post.n_draws, 4000)
    if reference is None:        # present-day strains
        base = np.array([rng.multivariate_normal(post.mu_inter[i], post.cov_inter[i]) for i in idx])
        ref_label = "Present-day strains (median, 90 % range)"
    else:                        # a single starting strain, e.g. ATCC 33560
        base = np.tile(np.asarray(reference, float), (2, 1))
        ref_label = reference_label
    b_lo, b_med, b_hi = survival_band(base, times, q)
    fig, axes = plt.subplots(1, len(conds), figsize=(2.9 * len(conds), 2.9), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        ax.fill_between(times, b_lo, b_hi, color="#c9c8c2", alpha=0.6, lw=0)
        ax.plot(times, b_med, color="#6f6e69", lw=1.2)
        X = results_inter[c].final_params()
        if len(X):
            lo, med, hi = survival_band(X, times, q)
            ax.fill_between(times, lo, hi, color=HEAT[c], alpha=0.22, lw=0)
            ax.plot(times, med, color=HEAT[c], lw=1.8)
        ax.axvline(T[c], color=INK, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        # what the same heating does: log reduction at T[c], present-day vs after 100 cycles
        from .sim import log_reduction
        r0 = float(np.median(log_reduction(T[c], np.exp(base[:, 0]), np.exp(base[:, 1]))))
        ax.plot(T[c], -r0, "o", ms=5, color="#6f6e69", mec="white", mew=0.8, zorder=6)
        sub = f"{T[c]:.1f} min at 55 °C: {r0:.1f} log"
        if len(X):
            r1 = float(np.median(log_reduction(T[c], np.exp(X[:, 0]), np.exp(X[:, 1]))))
            ax.plot(T[c], -r1, "o", ms=5, color=HEAT[c], mec="white", mew=0.8, zorder=6)
            sub += f" → {r1:.1f} log"
        ax.set_ylim(-10, 0.6); ax.set_xlim(0, t_max)
        ax.set_title(f"{HEAT_LABEL[c]}\n{sub}", loc="left", fontsize=8.5, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("log10 (N / N0)")
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color="#6f6e69", lw=1.2, label=ref_label),
            Line2D([], [], color=HEAT["4log"], lw=1.8, label="Lineages after 100 cycles (median, 90 % range)"),
            Line2D([], [], color=INK, lw=0.8, ls=(0, (1, 1.5)), label="Heating time used in each cycle")]
    _finish_row(fig, keys, 3, "Heating time at 55 °C (min)")
    return fig


# ---------------------------------------------------------------------------
# Fig. 1 (new): variability of present-day strains, from the data
# ---------------------------------------------------------------------------

LOW_RES, HIGH_RES = "#2a78d6", "#1baf7a"     # categorical slots 1 and 3 (validated all-pairs)


def fig1_present_day_variability(cd, post, t_max: float = 16.0, xlim=(0.5, 3.2), ylim=(0.35, 2.0),
                                 reference: str = "ATCC 33560"):
    """Fig. 1: (a) measured survival curves of the least and the most heat-resistant strain
    (points = plate counts, thin = each replicate experiment, thick = strain), with δ shown
    as the 6-log time divided by 6; (b) all 50 strains on the δ–p plane, the range of
    present-day strains, and the replicate experiments of three strains, which differ less
    than strains do (the within-strain spread the simulation uses for change between
    generations)."""
    from .analysis import curve_means, strain_means
    from .sim import log_reduction, time_for_reduction
    th_c, c0 = curve_means(post)
    sm = strain_means(post)
    if th_c is None or sm is None:
        raise ValueError("posterior has no curve-level estimates; refit with Posterior.from_cmdstan")
    curves = cd.curves
    strain_of = curves["strain"].to_numpy()                 # 1-based
    lo_s, hi_s = int(np.argmin(sm[:, 0])), int(np.argmax(sm[:, 0]))
    show = [(lo_s, LOW_RES, "least resistant"), (hi_s, HIGH_RES, "most resistant")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw=dict(width_ratios=[1.05, 1]))
    y_floor = -8.5
    tab = cd.table
    t_obs_max = float(tab["t_heat"].max())
    for s, col, tag in show:
        name = cd.strains[s]
        d, p = np.exp(sm[s])
        t_end = min(t_max, float(time_for_reduction(-y_floor, d, p)))
        for j in np.flatnonzero(strain_of == s + 1):          # 0-based curve index
            g = tab[(tab["curve"] == j + 1) & (tab["Nt_count"] > 0)]
            y = np.log10(g["Nt_count"] / g["V"]) + g["n_dilu"] - c0[j]
            a.scatter(g["t_heat"], y, s=9, color=col, alpha=0.6, lw=0, zorder=3)
            dj, pj = np.exp(th_c[j])
            tt = np.linspace(0, min(t_end, t_obs_max), 100)
            a.plot(tt, -log_reduction(tt, dj, pj), color=col, lw=0.5, alpha=0.8, zorder=2)
        tt = np.linspace(0, t_end, 200)                        # strain curve, drawn on to the 6-log time
        a.plot(tt, -log_reduction(tt, d, p), color=col, lw=2.0, zorder=4)
        # delta = (time to 6-log reduction) / 6
        a.plot([6 * d, 6 * d], [-6, y_floor], color=col, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        right = 6 * d > 0.6 * t_max
        # label on the side of the dotted line away from the curves
        _halo(a.text(6 * d - 0.25, -6.25, f"6δ = {6 * d:.1f} min\nδ = {d:.2f} min",
                     color=INK, fontsize=7, va="top", ha="right", zorder=6))
        if tag == "least resistant":
            _halo(a.text(0.3, -7.75, f"{name}\n({tag})", color=INK, fontsize=7, ha="left", va="center",
                         zorder=6))
        else:                                   # two lines, up and to the right of the end of its curve
            x_lab, lift = t_obs_max + 0.35, 1.2         # left edge of the label (min), height above the line (log)
            _halo(a.text(x_lab, -log_reduction(x_lab, d, p) + lift, f"{name}\n({tag})", color=INK, fontsize=7,
                         ha="left", va="bottom", zorder=6))
    a.axhline(-6, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    _halo(a.text(0.3, -5.85, "6-log reduction", fontsize=7, color=INK2, va="bottom", ha="left", zorder=6))
    a.set_xlim(0, t_max); a.set_ylim(y_floor, 0.6)
    a.set_xlabel("Heating time at 55 °C (min)"); a.set_ylabel("log10 (N / N0)")
    a.set_title("(a) Measured survival curves", loc="left", fontsize=9, color=INK)

    # (b) delta–p plane
    rng = np.random.default_rng(1)
    idx = rng.integers(0, post.n_draws, 4000)
    base = np.array([rng.multivariate_normal(post.mu_inter[i], post.cov_inter[i]) for i in idx])
    _hdr_contour(b, base, "#6f6e69", levels=(0.9,), lw=(0.8,), fill=False)
    _strain_background(b, sm)
    ref = cd.strains.index(reference) if reference in cd.strains else None
    trio = show + ([(ref, INK, "start of Fig. 3")] if ref is not None else [])
    label_gap = 5          # points between the highest point of a strain (mean or replicate) and its name
    for s, col, tag in trio:
        cs = np.exp(th_c[strain_of == s + 1]); ms = np.exp(sm[s])
        for c in cs:
            b.plot([ms[0], c[0]], [ms[1], c[1]], color=col, lw=0.7, zorder=3)
        b.scatter(cs[:, 0], cs[:, 1], s=14, facecolor="white", edgecolor=col, lw=1.0, zorder=4)
        b.scatter(*ms, s=36, color=col, edgecolor="white", lw=0.8, zorder=5)
        top = (ms[0], max(ms[1], cs[:, 1].max()))           # name centered above the strain's points
        _halo(b.annotate(f"{cd.strains[s]}", top, xytext=(0, label_gap), textcoords="offset points",
                         fontsize=7.5, color=INK, ha="center", va="bottom", zorder=7))
    b.axhline(1.0, color=GRID, lw=0.8, zorder=0)
    b.set_xlim(*xlim); b.set_ylim(*ylim)
    b.set_xlabel(DELTA_LABEL); b.set_ylabel(P_LABEL)
    b.set_title("(b) 50 strains and their replicates", loc="left", fontsize=9, color=INK)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], marker="o", ls="", color=STRAIN_DOT, ms=4, label="Strain (n = 50)"),
            Line2D([], [], color="#6f6e69", lw=0.8, label="Range of present-day strains (90 %)"),
            Line2D([], [], marker="o", ls="-", lw=0.7, color=INK2, mfc="white", ms=4,
                   label="Replicate experiments of one strain")]
    _finish_row(fig, keys, 3)
    return fig


# ---------------------------------------------------------------------------
# Supplementary figures in the same visual language as the main figures
# (δ and p in natural units, heating intensity = HEAT ramp, direct labels)
# ---------------------------------------------------------------------------

START_ATCC, START_INTER = INK, "#8a8984"


def _label_right(ax, x, ys: dict, min_gap: float, fmt=lambda k: k):
    """Direct labels just right of the plot area (x is ignored; kept for compatibility),
    nudged apart so that they never overlap."""
    placed = []
    for k, y in sorted(ys.items(), key=lambda kv: kv[1]):
        if not np.isfinite(y):
            continue
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(fmt(k), (1.01, y), xycoords=("axes fraction", "data"), color=INK, fontsize=7.5,
                    va="center", annotation_clip=False)


def figS1_validation(val_atcc, val_inter, mu_t, cov_t, md2):
    """Fig. S1: without heating, lineages started far from (ATCC 33560) or within the
    present-day distribution end in that distribution: (a, b) mean δ and p over cycles
    against the target; (c) squared Mahalanobis distances against χ²(2)."""
    from scipy import stats
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    tgt = np.exp(np.asarray(mu_t))
    for ax, k, yl in [(axes[0], 0, DELTA_SHORT + "\n(geometric mean of lineages)"),
                      (axes[1], 1, "Curve shape, p\n(geometric mean of lineages)")]:
        col = "lndelta_mean" if k == 0 else "lnp_mean"
        ax.plot(np.exp(val_atcc.summary()[col]), color=START_ATCC, lw=1.4)
        ax.plot(np.exp(val_inter.summary()[col]), color=START_INTER, lw=1.4)
        ax.axhline(tgt[k], color=INK2, lw=0.8, ls=(0, (3, 2)))
        n = len(val_atcc.summary())
        ax.annotate("target", (1.01, tgt[k]), xycoords=("axes fraction", "data"), fontsize=7, color=INK2,
                    va="center", annotation_clip=False)
        ax.set_xlabel("Cycle (no heating)"); ax.set_ylabel(yl)
    axes[0].set_title("(a)", loc="left", fontsize=9); axes[1].set_title("(b)", loc="left", fontsize=9)
    q = np.linspace(0.001, 0.999, 300)
    lim = [0, stats.chi2.ppf(0.999, 2)]
    axes[2].plot(lim, lim, color=INK2, lw=0.8, ls=(0, (3, 2)))
    axes[2].plot(stats.chi2.ppf(q, 2), np.quantile(md2, q), "o", ms=2.5, color=START_ATCC)
    ks = stats.kstest(md2, "chi2", args=(2,))
    _halo(axes[2].text(0.04, 0.96, f"KS test: D = {ks.statistic:.3f}, P = {ks.pvalue:.2f}",
                 transform=axes[2].transAxes, fontsize=7, va="top", color=INK))
    axes[2].set_xlabel("Expected (χ² with 2 df)"); axes[2].set_ylabel("Observed squared\nMahalanobis distance")
    axes[2].set_title("(c) End points vs present-day distribution", loc="left", fontsize=9)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=START_ATCC, lw=1.4, label="Start: ATCC 33560"),
            Line2D([], [], color=START_INTER, lw=1.4, label="Start: present-day strains"),
            Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="Target (present-day distribution)")]
    _finish_row(fig, keys, 3)
    return fig


def figS2_means_over_cycles(results_inter: dict, results_atcc: dict, conds=("none", "2log", "4log", "6log")):
    """Fig. S2: geometric-mean δ and p of the surviving lineages over cycles (former Fig. 5).
    Solid: start from present-day strains; dashed: start from ATCC 33560."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for ax, col, yl in [(axes[0], "lndelta_mean", DELTA_SHORT), (axes[1], "lnp_mean", "Curve shape, p")]:
        ends = {}
        for c in conds:
            si, sa = results_inter[c].summary(), results_atcc[c].summary()
            ax.plot(si["cycle"], np.exp(si[col]), color=HEAT[c], lw=1.6)
            ax.plot(sa["cycle"], np.exp(sa[col]), color=HEAT[c], lw=1.0, ls=(0, (4, 2)))
            ends[HEAT_LABEL[c].replace(" heating", "")] = float(np.exp(si[col].iloc[-1]))
        _label_right(ax, si["cycle"].iloc[-1] + 2, ends, min_gap=0.07 * np.ptp(ax.get_ylim()))
        ax.set_xlabel("Heating–regrowth cycle"); ax.set_ylabel(yl + "\n(geometric mean of surviving lineages)")
    axes[0].set_title("(a) δ: time to 6-log reduction ÷ 6", loc="left", fontsize=9)
    axes[1].set_title("(b) Curve shape", loc="left", fontsize=9)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=INK2, lw=1.6, label="Start: present-day strains"),
            Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label="Start: ATCC 33560")]
    _finish_row(fig, keys, 2)
    return fig


def fig_sensitivity_summary(res: dict, values, value_label: str, fmt=str, conds=("2log", "4log", "6log"),
                            main_value=None, min_alive: int = 30, title: str = ""):
    """Sensitivity figures S3–S5: for each setting (x) and heating intensity (color),
    (a) the δ the surviving lineages reach (plateau; geometric mean) and (b) the share of
    lineages not lost after the last cycle. Points are omitted when fewer than `min_alive`
    lineages survive. The setting used in the main analysis is shaded."""
    from .analysis import plateau
    xs = np.arange(len(values))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.0))
    start = None
    ends_a, ends_b = {}, {}
    for c in conds:
        ya, yb = [], []
        for v in values:
            r = res[(v, c)]
            sm = r.summary()
            start = float(np.exp(sm["lndelta_mean"].iloc[0])) if start is None else start
            ok = int(r.alive[:, -1].sum()) >= min_alive
            ya.append(float(np.exp(plateau(sm))) if ok else np.nan)
            yb.append(100 * float(sm["survival_ratio"].iloc[-1]))
        a.plot(xs, ya, "-o", color=HEAT[c], ms=4.5, mec="white", mew=0.8, lw=1.4)
        b.plot(xs, yb, "-o", color=HEAT[c], ms=4.5, mec="white", mew=0.8, lw=1.4)
        lab = HEAT_LABEL[c].replace(" heating", "")
        fin = [v for v in ya if np.isfinite(v)]
        ends_a[lab], ends_b[lab] = (fin[-1] if fin else np.nan), yb[-1]
    if start is not None:
        a.axhline(start, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
        if "none" not in conds:          # with no heating the gray line already marks the start
            _halo(a.text(xs[0] - 0.4, start, "start", fontsize=7, color=INK2, va="bottom"))
    lo, hi = a.get_ylim(); a.set_ylim(min(lo, 0.9 * (start or lo)), hi)
    _label_right(a, xs[-1] + 0.15, ends_a, min_gap=0.07 * np.ptp(a.get_ylim()))
    _label_right(b, xs[-1] + 0.15, ends_b, min_gap=7)
    for ax in (a, b):
        if main_value is not None and main_value in list(values):
            i = list(values).index(main_value)
            ax.axvspan(i - 0.35, i + 0.35, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.set_xticks(xs, [fmt(v) for v in values])
        ax.set_xlabel(value_label); ax.set_xlim(-0.5, len(values) - 0.5)
    a.set_ylabel("δ reached after 100 cycles (min)")
    b.set_ylabel("Lineages not lost (%)"); b.set_ylim(-3, 103)
    a.set_title("(a) How far lineages move", loc="left", fontsize=9)
    b.set_title("(b) How many lineages survive", loc="left", fontsize=9)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    return fig


def figS6_qmra(r_base, r_conv, label_conv="After 100 cycles of 6-log heating"):
    """Fig. S6: probability of infection per portion after a 4-log treatment, for present-day
    strains and for lineages after repeated heating (log scale)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    lo = max(min(r_base.min(), r_conv.min()), 1e-8)
    bins = np.logspace(np.log10(lo), 0, 60)
    for r, col, lab in [(r_base, "#6f6e69", "Present-day strains"), (r_conv, HEAT["6log"], label_conv)]:
        ax.hist(r, bins=bins, density=False, weights=np.full(len(r), 100 / len(r)), histtype="stepfilled",
                color=col, alpha=0.25, lw=0)
        ax.hist(r, bins=bins, density=False, weights=np.full(len(r), 100 / len(r)), histtype="step",
                color=col, lw=1.2)
    ax.set_xscale("log")
    top = ax.get_ylim()[1] * 1.35
    ax.set_ylim(0, top)
    for r, col, lab, ha in [(r_base, "#6f6e69", "Present-day strains", "right"),
                            (r_conv, HEAT["6log"], label_conv.replace(" of ", "\nof "), "left")]:
        m = float(np.mean(r))
        ax.axvline(m, color=col, lw=0.9, ls=(0, (3, 2)))
        pad = " " if ha == "left" else ""
        _halo(ax.text(m, top * 0.98, f"{pad}{lab}\n{pad}mean {m:.3g} " if ha == "right" else f"{pad}{lab}\n{pad}mean {m:.3g}",
                      fontsize=7, color=INK, va="top", ha=ha))
    ax.set_xlabel("Probability of infection per portion"); ax.set_ylabel("Share of strains or lineages (%)")
    fig.tight_layout()
    return fig
