#!/usr/bin/env python
"""
group_plots_with_fwer_overlay_fixed_per_arousal.py

Load subject-level peri-event trial arrays, compute group-averaged timecourses,
run time-resolved LMEs for Signal contrasts (D2 vs D1, NE vs D1), apply permutation-based
cluster FWER on the resulting p-value vectors (shuffle p-values), and overlay
FWER-significant clusters as horizontal bars on the group figures.

CHANGES:
- Compute p-values and permutation-based nulls separately FOR EACH AROUSAL TYPE (Sustained/Transient/Loss)
- Save p-values per-arousal-type CSVs
- Use the corresponding per-arousal mask when drawing cluster bars on that row
(keeps the rest of your original structure)
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import os

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


# Configuration

base_dir = "/home/vob3"
subjects = [
    "sub-racsleep04b", "sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11",
    "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", "sub-racsleep20",
    "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30",
    "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
    "sub-racsleep38"
]

window = [-15, 20]
TR = 2.22
# compute time_axis AFTER we see the first trial length 

# Signals
selected_signals = ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]
signal_short = {"D1_with_reg_PSC": "D1", "D2_with_reg_PSC": "D2", "Norepinephrine_with_reg": "NE"}

# plotting params
n_boot = 2000
ci = 95
output_dir = Path("group_with_fwer_outputs")
output_dir.mkdir(exist_ok=True, parents=True)

# Helper functions 

def fit_model_pval_for_term(df_time, formula, term):
    """
    Fit a mixedlm on df_time (grouping by Subject) and return p-value for term.
    Returns np.nan if model fails or term missing.
    """
    if df_time.shape[0] == 0:
        return np.nan
    try:
        model = smf.mixedlm(formula, df_time, groups=df_time["Subject"])
        res = model.fit()
    except Exception:
        return np.nan
    return float(res.pvalues.get(term, np.nan))


def get_pvals_for_term(df_long, time_axis, formula, term):
    """
    Run model at each time point, return pvals array of length T (np.array).
    """
    pvals = []
    for t in time_axis:
        df_t = df_long[df_long["Time"] == t]
        p = fit_model_pval_for_term(df_t, formula, term)
        # If a p couldn't be computed, use 1.0 (non-significant)
        pvals.append(1.0 if np.isnan(p) else p)
    return np.array(pvals)


def max_cluster_size_from_binary(sig_vec):
    """Return maximum consecutive length of True in boolean array ."""
    max_run = 0
    cur = 0
    for v in sig_vec:
        if v:
            cur += 1
        else:
            if cur > max_run:
                max_run = cur
            cur = 0
    if cur > max_run:
        max_run = cur
    return max_run


def null_dist_from_pvalues_shuffle(pvals, n_perm=1000, alpha=0.05, seed=None):
    """
    Build null distribution of max cluster sizes by shuffling the p-value vector.
    Lightweight — does not refit models.
    """
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_perm):
        shuffled = rng.permutation(pvals)
        sig = shuffled < alpha
        null.append(max_cluster_size_from_binary(sig))
        if (i + 1) % 100 == 0:
            print(f"  permutation {i+1}/{n_perm}")
    return np.array(null)


def mask_from_cluster_threshold(obs_pvals, null_dist, alpha=0.05, percentile=95):
    """
    Given observed p-values and null distribution of max-cluster sizes,
    return boolean mask (length T) marking timepoints that belong to observed clusters
    whose size > threshold (percentile of null).
    """
    cutoff = np.percentile(null_dist, percentile)
    sig_binary = obs_pvals < alpha

    # find observed clusters (start,end indices)
    clusters = []
    start = None
    for i, val in enumerate(sig_binary):
        if val and start is None:
            start = i
        if (not val or i == len(sig_binary) - 1) and start is not None:
            # compute end properly
            if val:  # last element was True and we're at last index
                end = i
            else:
                end = i - 1
            if end >= start:
                clusters.append((start, end))
            start = None

    # mark indices that belong to clusters that exceed cutoff
    mask = np.zeros_like(sig_binary, dtype=bool)
    for (s, e) in clusters:
        length = e - s + 1
        if length > cutoff:   # cluster must be larger than null cutoff
            mask[s:e+1] = True
    return mask, cutoff


# Build long-format DataFrame from peri_trials (and time_axis)

all_rows = []
n_timepoints = None
time_axis = None

for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing: {fpath}")
        continue
    data = np.load(fpath, allow_pickle=True)["peri_trials"].item()

    for sig_key in selected_signals:
        sig_short = signal_short[sig_key]
        trials_by_arousal = data.get("with_reg", {}).get(sig_key, {})
        # for group-level Signal contrasts we keep all arousal levels
        # collect all trials for each subject for that signal
        for arousal_type, trials in trials_by_arousal.items():
            if trials is None or len(trials) == 0:
                continue
            for trial in trials:
                # detect n_timepoints from first trial we encounter
                if n_timepoints is None:
                    n_timepoints = int(len(trial))
                    time_axis = np.linspace(window[0], window[1], n_timepoints)
                    print(f"Detected n_timepoints={n_timepoints}; time_axis set.")
                # safety check
                if len(trial) != n_timepoints:
                    raise RuntimeError(f"Subject {subj} signal {sig_key} trial length {len(trial)} != expected {n_timepoints}")
                for t_idx, t_val in enumerate(time_axis):
                    all_rows.append({
                        "Subject": subj,
                        "Signal": sig_short,
                        "Time": float(t_val),
                        "PSC": float(trial[t_idx]),
                        "ArousalType": arousal_type
                    })

if len(all_rows) == 0:
    raise RuntimeError("No peri-event trials found. Check subject files and paths.")

df_long = pd.DataFrame(all_rows)
print(f"Data loaded: {df_long.shape} rows")

# categorical ordering so D1 is baseline
df_long["Signal"] = pd.Categorical(df_long["Signal"], categories=["D1", "D2", "NE"])
df_long["ArousalType"] = pd.Categorical(df_long["ArousalType"])


# time-resolved p-values & permutation FWER FOR EACH AROUSAL TYPE

formula_signal = "PSC ~ Signal"
term_D2 = "Signal[T.D2]"
term_NE = "Signal[T.NE]"

print("Computing observed p-values and FWER masks per arousal type")

# PER-AROUSAL containers 
arousal_order = ['Sustained', 'Transient', 'Loss']   # same as your plotting rows
pvals_D2_by_arousal = {}
pvals_NE_by_arousal = {}
mask_D2_by_arousal = {}
mask_NE_by_arousal = {}
cutoff_D2_by_arousal = {}
cutoff_NE_by_arousal = {}

n_perm = 1000
alpha = 0.05

for ar in arousal_order:
    print(f"\nProcessing ArousalType = {ar}")
    df_ar = df_long[df_long["ArousalType"] == ar]
    if df_ar.empty:
        print(f"  No data for {ar}; skipping.")
        pvals_D2_by_arousal[ar] = np.ones(n_timepoints)
        pvals_NE_by_arousal[ar] = np.ones(n_timepoints)
        mask_D2_by_arousal[ar] = np.zeros(n_timepoints, dtype=bool)
        mask_NE_by_arousal[ar] = np.zeros(n_timepoints, dtype=bool)
        cutoff_D2_by_arousal[ar] = 0
        cutoff_NE_by_arousal[ar] = 0
        continue

    # observed p-values
    pvals_D2 = get_pvals_for_term(df_ar, time_axis, formula_signal, term_D2)
    pvals_NE = get_pvals_for_term(df_ar, time_axis, formula_signal, term_NE)

    pvals_D2_by_arousal[ar] = pvals_D2
    pvals_NE_by_arousal[ar] = pvals_NE

    # save raw p-values per-arousal
    pd.DataFrame({"Time": time_axis, "pval_D2_vs_D1": pvals_D2}).to_csv(output_dir / f"pvals_D2_vs_D1_{ar}.csv", index=False)
    pd.DataFrame({"Time": time_axis, "pval_NE_vs_D1": pvals_NE}).to_csv(output_dir / f"pvals_NE_vs_D1_{ar}.csv", index=False)

    # null distributions (shuffle p-value vector)
    null_D2 = null_dist_from_pvalues_shuffle(pvals_D2, n_perm=n_perm, alpha=alpha)
    null_NE = null_dist_from_pvalues_shuffle(pvals_NE, n_perm=n_perm, alpha=alpha)

    # build masks (clusters that exceed cutoff)
    mask_D2, cutoff_D2 = mask_from_cluster_threshold(pvals_D2, null_D2, alpha=alpha, percentile=95)
    mask_NE, cutoff_NE = mask_from_cluster_threshold(pvals_NE, null_NE, alpha=alpha, percentile=95)

    pvals_D2_by_arousal[ar] = pvals_D2
    pvals_NE_by_arousal[ar] = pvals_NE
    mask_D2_by_arousal[ar] = mask_D2
    mask_NE_by_arousal[ar] = mask_NE
    cutoff_D2_by_arousal[ar] = cutoff_D2
    cutoff_NE_by_arousal[ar] = cutoff_NE

    print(f"  {ar}: D2 cutoff={cutoff_D2:.1f}, NE cutoff={cutoff_NE:.1f}")


# Build group_data for bootstrapped plots 

signal_sets = {
    "with_reg": ["GM_PSC", "Thalamus_PSC", "D2_with_reg_PSC", "D1_with_reg_PSC", "Norepinephrine_with_reg"],
    "without_reg": ["GM_PSC", "Thalamus_PSC", "D2_without_reg_PSC", "D1_without_reg_PSC", "Norepinephrine_without_reg"]
}

group_data = {}
for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        continue
    data = np.load(fpath, allow_pickle=True)["peri_trials"].item()
    for reg_type in signal_sets:
        for signal in signal_sets[reg_type]:
            for arousal_type in ['Sustained', 'Transient', 'Loss']:
                trials = data.get(reg_type, {}).get(signal, {}).get(arousal_type, None)
                if trials is None or len(trials) == 0:
                    continue
                group_data.setdefault(arousal_type, {}) \
                          .setdefault(reg_type, {}) \
                          .setdefault(signal, {}) \
                          .setdefault(subj, []).append(np.asarray(trials))

def hierarchical_bootstrap_timecourses(trial_dict, n_boot=2000, ci=95):
    subj_ids = list(trial_dict.keys())
    n_subj = len(subj_ids)
    if n_subj == 0:
        return None, None, None
    boot_samples = []
    for _ in range(n_boot):
        sampled_subjects = np.random.choice(subj_ids, size=n_subj, replace=True)
        all_resampled = []
        for sid in sampled_subjects:
            subj_trials = np.vstack(trial_dict[sid])  # shape (n_trials, T)
            n_trials = subj_trials.shape[0]
            resamp_idx = np.random.choice(n_trials, n_trials, replace=True)
            resampled = subj_trials[resamp_idx]
            all_resampled.append(resampled)
        all_concat = np.vstack(all_resampled)
        boot_samples.append(np.mean(all_concat, axis=0))
    boot_samples = np.array(boot_samples)
    mean = np.mean(boot_samples, axis=0)
    lower = np.percentile(boot_samples, (100 - ci) / 2, axis=0)
    upper = np.percentile(boot_samples, 100 - (100 - ci) / 2, axis=0)
    return mean, lower, upper

# helper to add cluster bar 
def add_cluster_bar(ax, time_axis, mask, color='yellow', y_offset=0.05, thickness=0.02):
    if mask is None or not mask.any():
        return
    ylim = ax.get_ylim()
    y_top = ylim[1]
    y_range = ylim[1] - ylim[0]
    y_bar = y_top + y_offset * y_range
    # find contiguous runs
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = i
            prev = i
    runs.append((start, prev))
    for (s, e) in runs:
        ax.hlines(y_bar, xmin=time_axis[s], xmax=time_axis[e], colors=color, linewidth=6, alpha=0.9)
    # expand ylim so bars are visible
    ax.set_ylim(top=y_bar + thickness * y_range)


signal_colors_left = {'D1_with_reg_PSC': 'purple', 'D2_with_reg_PSC': 'orange', 'Norepinephrine_with_reg': 'cyan'}
signal_colors_right = {'GM_PSC': 'blue', 'Thalamus_PSC': 'green'}

fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=False)
arousal_order = ['Sustained', 'Transient', 'Loss']

for i, arousal_type in enumerate(arousal_order):
    # LEFT: neuroreceptors (D1/D2/NE)
    for signal in ['D1_with_reg_PSC', 'D2_with_reg_PSC', 'Norepinephrine_with_reg']:
        trial_dict = group_data.get(arousal_type, {}).get('with_reg', {}).get(signal, {})
        if trial_dict:
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis_plot = np.linspace(window[0], window[1], mean.shape[0])
            axs[i, 0].plot(time_axis_plot, mean, label=signal.split('_')[0], color=signal_colors_left[signal])
            axs[i, 0].fill_between(time_axis_plot, lower, upper, color=signal_colors_left[signal], alpha=0.2)

    axs[i, 0].set_title(f"{arousal_type} - D1/D2/NE")
    axs[i, 0].axvline(0, linestyle='--', color='black')
    axs[i, 0].legend()

    # Overlay FWER cluster bars for this arousal type
    mask_D2 = mask_D2_by_arousal.get(arousal_type, np.zeros(n_timepoints, dtype=bool))
    mask_NE = mask_NE_by_arousal.get(arousal_type, np.zeros(n_timepoints, dtype=bool))
    add_cluster_bar(axs[i, 0], time_axis, mask_D2, color='orange', y_offset=0.04)
    add_cluster_bar(axs[i, 0], time_axis, mask_NE, color='cyan', y_offset=0.08)

    # RIGHT: GM + Thalamus
    for signal in ['GM_PSC', 'Thalamus_PSC']:
        trial_dict = group_data.get(arousal_type, {}).get('with_reg', {}).get(signal, {})
        if trial_dict:
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis_plot = np.linspace(window[0], window[1], mean.shape[0])
            axs[i, 1].plot(time_axis_plot, mean, label=signal.split('_')[0], color=signal_colors_right[signal])
            axs[i, 1].fill_between(time_axis_plot, lower, upper, color=signal_colors_right[signal], alpha=0.2)

    axs[i, 1].set_title(f"{arousal_type} - GM & Thalamus")
    axs[i, 1].axvline(0, linestyle='--', color='black')
    axs[i, 1].legend()

# axis labels
for ax in axs[:, 0]:
    ax.set_ylabel("% Signal Change")
for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")

fig.suptitle("Group-Averaged Signal Traces by Arousal Type (FWER clusters overlaid)", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(output_dir / "group_summary_with_fwer_grid.png")
plt.close(fig)

print(f"All done. Outputs in {output_dir.resolve()}")



