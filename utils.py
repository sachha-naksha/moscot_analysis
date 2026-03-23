import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

import moscot as mt
import moscot.plotting as mpl
from moscot.problems.time import TemporalProblem

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

def drivers_tf_for_subset(
    tp0,
    subset,
    t_early=(1.5, 3.5),
    t_late=(3.5, 5.5),
    data_key="sub_cell_type",
    features="human",
    obs_key_prefix=None,
    normalize_pull=False,
    corr_annotation=None,
    qval_thresh=0.05,
    corr_thresh=0.1,
):
    prefix = obs_key_prefix if obs_key_prefix is not None else f"{subset[0]}_{subset[1]}"
    key_early = f"{prefix}_pull_early"
    key_late  = f"{prefix}_pull_late"
    key_sum   = f"{prefix}_pull"

    # --- 1. Pull back early transition ---
    tp0.pull(
        t_early[0], t_early[1],
        data=data_key,
        subset=subset[0],
        key_added=key_early,
        normalize=normalize_pull,
    )

    # --- 2. Pull back late transition ---
    tp0.pull(
        t_late[0], t_late[1],
        data=data_key,
        subset=subset[1],
        key_added=key_late,
        normalize=normalize_pull,
    )

    # --- 3. Sum pull weights and validate ---
    tp0.adata.obs[key_sum] = tp0.adata.obs[key_early] + tp0.adata.obs[key_late]

    pull_sum = tp0.adata.obs[key_sum].sum()
    print(f"[pull] key='{key_sum}' | combined weight sum = {pull_sum:.4f}")
    if pull_sum == 0:
        raise ValueError(
            f"All combined pull weights are 0. Check that subsets {subset} exist "
            f"at timepoints {t_early} / {t_late} in obs['{data_key}']."
        )

    # --- 4. Compute feature correlation ---
    drivers = tp0.compute_feature_correlation(
        obs_key=key_sum,
        features=features,
        annotation=corr_annotation,
    )

    # --- 5. Rename columns to be prefix-specific ---
    drivers.columns = [c.replace(key_sum, prefix) for c in drivers.columns]
    corr_col = f"{prefix}_corr"
    qval_col = f"{prefix}_qval"

    # --- 6. Add significance flag and sort ---
    drivers["significant"] = (
        (drivers[qval_col] < qval_thresh) &
        (drivers[corr_col].abs() > corr_thresh)
    )
    drivers = drivers.dropna(subset=[corr_col]).sort_values(corr_col, ascending=False)

    n_sig = drivers["significant"].sum()
    print(f"\n[result] {n_sig} significant drivers "
          f"(qval<{qval_thresh}, |corr|>{corr_thresh})")

    # --- 7. Display top 10 positive and negative drivers ---
    top_pos = drivers.head(10).style.set_caption(
        f"TOP 10 POSITIVE DRIVERS  ({subset[0]} → {subset[1]})"
    ).background_gradient(subset=[corr_col], cmap="Reds")

    top_neg = drivers.tail(10).iloc[::-1].style.set_caption(
        f"TOP 10 NEGATIVE DRIVERS  ({subset[0]} → {subset[1]})"
    ).background_gradient(subset=[corr_col], cmap="Blues_r")

    from IPython.display import display
    display(top_pos)
    display(top_neg)

    return drivers, drivers.head(10), drivers.tail(10).iloc[::-1]

def plot_drivers(
    drivers: "pd.DataFrame",
    subset: tuple,
    obs_key_prefix: str | None = None,
    n_top: int = 10,
    power: float = 0.4,          # <1 compresses large values → stretches small ones
    figsize: tuple = (9, 8),
    cmap_pos: str = "Reds",
    cmap_neg: str = "Blues",
    bar_height: float = 0.65,
    title_fontsize: int = 13,
    label_fontsize: int = 10,
):
    """
    Horizontal bar plot of top/bottom TF drivers with a power-norm colorbar.

    Parameters
    ----------
    drivers : pd.DataFrame
        Full drivers table returned by `drivers_tf_for_subset`.
    subset : tuple
        (source_state, target_state) used to derive column names.
    obs_key_prefix : str, optional
        Override the column prefix (default: f"{subset[0]}_{subset[1]}").
    n_top : int
        Number of positive and negative drivers to show.
    power : float
        Exponent for the power-norm colour stretch (< 1 → boost small values).
        0.4 is a good default; lower → more contrast at small corr values.
    figsize : tuple
    cmap_pos / cmap_neg : str
        Matplotlib colourmap names for positive / negative bars.
    bar_height : float
    title_fontsize / label_fontsize : int
    """
    prefix = obs_key_prefix if obs_key_prefix is not None else f"{subset[0]}_{subset[1]}"
    corr_col = f"{prefix}_corr"

    top_pos = drivers[drivers[corr_col] > 0].head(n_top).copy()
    top_neg = drivers[drivers[corr_col] < 0].tail(n_top).iloc[::-1].copy()

    # --- power-transform for colour mapping (preserves sign, stretches small vals) ---
    def _pow(x, g):
        return np.sign(x) * np.abs(x) ** g

    pos_vals = top_pos[corr_col].values
    neg_vals = top_neg[corr_col].values

    abs_pos = np.abs(pos_vals)
    abs_neg = np.abs(neg_vals)
    pos_norm = mcolors.PowerNorm(gamma=power, vmin=abs_pos.min() if len(abs_pos) else 0, vmax=abs_pos.max() if len(abs_pos) else 1)
    neg_norm = mcolors.PowerNorm(gamma=power, vmin=abs_neg.min() if len(abs_neg) else 0, vmax=abs_neg.max() if len(abs_neg) else 1)

    cmap_p = cm.get_cmap(cmap_pos)
    cmap_n = cm.get_cmap(cmap_neg)

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={"wspace": 0.55})
    fig.patch.set_facecolor("white")

    def _draw_panel(ax, df, vals, abs_vals, norm, cmap, title, xlabel_sign, negate_cbar=False):
        colors = cmap(norm(abs_vals))
        y = np.arange(len(df))
        bars = ax.barh(y, vals, height=bar_height, color=colors,
                       edgecolor="white", linewidth=0.4)

        ax.set_yticks(y)
        ax.set_yticklabels(df.index, fontsize=label_fontsize)
        ax.set_xlabel("Pearson r", fontsize=label_fontsize)
        ax.set_title(title, fontsize=title_fontsize, pad=8, fontweight="semibold")
        ax.axvline(0, color="#aaaaaa", lw=0.8, ls="--")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(axis="both", colors="#555555")
        ax.set_facecolor("#fafafa")

        # invert so highest corr is at top
        ax.invert_yaxis()

        # --- colorbar ---
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                             fraction=0.04, pad=0.02, aspect=25)
        cbar.set_label(f"|r|^{power:.1f}  (power-scaled)", fontsize=8,
                       color="#555555")
        cbar.ax.tick_params(labelsize=7, colors="#555555")
        if negate_cbar:
            cbar.set_ticks(cbar.get_ticks())
            cbar.set_ticklabels([f"{-t:.2f}" for t in cbar.get_ticks()])
        cbar.outline.set_edgecolor("#cccccc")

    _draw_panel(
        axes[0], top_pos, pos_vals, abs_pos, pos_norm, cmap_p,
        f"Top {n_top} positive drivers\n{subset[0]} → {subset[1]}", +1,
    )
    _draw_panel(
        axes[1], top_neg, neg_vals, abs_neg, neg_norm, cmap_n,
        f"Top {n_top} negative drivers\n{subset[0]} → {subset[1]}", -1, negate_cbar=True,
    )

    plt.suptitle(
        f"TF drivers  ·  {subset[0]} → {subset[1]}"
        f"  (colour power-norm  γ={power})",
        fontsize=title_fontsize + 1, y=1.02, color="#222222",
    )
    plt.tight_layout()
    plt.show()
    return fig