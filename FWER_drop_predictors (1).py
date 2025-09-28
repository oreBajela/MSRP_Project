"""
FWER_drop_predictors.py

Cluster-based permutation FWER for drop-predictor mixed models.

- Runs time-resolved likelihood-ratio (LR) tests comparing a full vs reduced mixed model
  at each peri-event timepoint (giving a p-value vector of length T).
- Builds a null distribution of max consecutive-significant-cluster sizes by
  shuffling the observed p-value vector N times, thresholding at p < alpha,
  and recording the largest run of consecutive 1's.
- Determines the 95th-percentile critical cluster size from the null distribution.
- significancant if the observed max-cluster size > critical_size.
- also computes a permutation style corrected p-value (proportion of null max clusters >= observed).

"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


base_dir = "/home/vob3"
subjects = [
    "sub-racsleep04b", "sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11",
    "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", "sub-racsleep20",
    "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30",
    "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
    "sub-racsleep38"
]

TR = 2.22
window = [-15, 20]
time_axis = np.linspace(window[0], window[1], int((window[1] - window[0]) / TR) + 1)

signal_list = ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]
arousal_types = ['Sustained', 'Transient', 'Loss']

output_dir = Path("drop_predictors_FWER_results")
output_dir.mkdir(exist_ok=True, parents=True)

# Load data
rows = []
for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing file: {fpath}")
        continue

    data = np.load(fpath, allow_pickle=True)['peri_trials'].item()
    for signal in signal_list:
        for arousal_type in arousal_types:
            trials = data.get("with_reg", {}).get(signal, {}).get(arousal_type, None)
            if trials is None or len(trials) == 0:
                continue

            trials_concat = np.vstack(trials)  # shape (n_trials, timepoints)
            for trial in trials_concat:
                for t_idx, value in enumerate(trial):
                    rows.append({
                        "Subject": subj,
                        "Signal": signal.replace("_with_reg_PSC", ""),
                        "ArousalType": arousal_type,
                        "Time": float(time_axis[t_idx]),
                        "SignalChange": float(value)
                    })

df = pd.DataFrame(rows).dropna()
print(f"\nLoaded data: {df.shape} rows")


def compare_models_lr(df_sub, formula_full, formula_reduced):
    """
    Fit full and reduced mixed models at this timepoint and return LR stat and p-value.
    Returns (lr_stat, df_diff, p_value) or (nan, nan, nan) if model fails.
    """
    try:
        model_full = smf.mixedlm(formula_full, df_sub, groups=df_sub["Subject"])
        res_full = model_full.fit()

        model_reduced = smf.mixedlm(formula_reduced, df_sub, groups=df_sub["Subject"])
        res_reduced = model_reduced.fit()

        lr_stat = 2 * (res_full.llf - res_reduced.llf)
        df_diff = res_full.df_modelwc - res_reduced.df_modelwc
        p_value = 1 - stats.chi2.cdf(lr_stat, df=df_diff)
        return lr_stat, df_diff, p_value
    except Exception as e:
        # return NaNs where model fits

        return np.nan, np.nan, np.nan


def run_timewise_lr(df_sub, formula_full, formula_reduced, time_axis, alpha=0.05):
    """
    Run LR tests at all timepoints, return p-values vector and observed max cluster size
    (clusters computed on threshold p < alpha).
    """
    lr_stats = []
    pvals = []
    for t in time_axis:
        cur = df_sub[df_sub["Time"] == t]
        lr, df_diff, p = compare_models_lr(cur, formula_full, formula_reduced)
        lr_stats.append(lr)
        pvals.append(p)

    pvals = np.array(pvals, dtype=float)
    # Treat NaN p-values (e.g., failed fits) as non-significant (set to 1.0)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)

    sig = pvals < alpha

    # compute max consecutive-run length
    clusters = []
    cur_run = 0
    for s in sig:
        if s:
            cur_run += 1
        else:
            if cur_run > 0:
                clusters.append(cur_run)
            cur_run = 0
    if cur_run > 0:
        clusters.append(cur_run)

    obs_max_cluster = max(clusters) if clusters else 0
    return pvals, obs_max_cluster



def permutation_on_pvals(pvals, n_perm=1000, alpha=0.05, seed=None):
    """
    Build null distribution of max cluster sizes by shuffling the p-value vector.

    Input
      pvals : 1D numpy array of length T (observed p-values across timepoints)
      n_perm: number of permutations
      alpha : cluster-forming threshold
    Output:
      null_max_clusters : numpy array (n_perm,)
    """
    rng = np.random.default_rng(seed)
    null_max = []
    T = len(pvals)

    for i in range(n_perm):
        shuffled = rng.permutation(pvals)  # shuffle the p-values 
        sig = shuffled < alpha

        # find longest consecutive run
        clusters = []
        cur_run = 0
        for s in sig:
            if s:
                cur_run += 1
            else:
                if cur_run > 0:
                    clusters.append(cur_run)
                cur_run = 0
        if cur_run > 0:
            clusters.append(cur_run)
        max_cluster = max(clusters) if clusters else 0
        null_max.append(max_cluster)

        if (i + 1) % 50 == 0:
            print(f"  Permutation {i+1}/{n_perm} done.")

    return np.array(null_max, dtype=int)



# FWER wrapper using p-value shuffling

def fwer_analysis(df_sub, label, formula_full, formula_reduced, time_axis, n_perm=500, alpha=0.05, seed=None):
    """
    For one data subset (df_sub) and one model comparison:
      - compute observed p-values vector across time
      - build null distribution by shuffling p-values (permutation_on_pvals)
      - compute 95th percentile critical cluster size and permutation p-value
      - save CSV + histogram and print summary
    """
    print(f"\n=== FWER analysis: {label} ===")
    obs_pvals, obs_max_cluster = run_timewise_lr(df_sub, formula_full, formula_reduced, time_axis, alpha=alpha)
    print(f"Observed max cluster (p<{alpha}) = {obs_max_cluster}")

    # Build null distribution by shuffling observed p-values
    null_dist = permutation_on_pvals(obs_pvals, n_perm=n_perm, alpha=alpha, seed=seed)

    # Critical cluster size = 95th percentile of null distribution
    crit = int(np.ceil(np.percentile(null_dist, 95)))  # ceil to make conservative
    # permutation-style corrected p-value
    p_corr = (np.sum(null_dist >= obs_max_cluster) + 1) / (len(null_dist) + 1)

    # Decision by 95th percentile: observed cluster > crit => significant cluster
    significant_by_crit = bool(obs_max_cluster > crit)

    print(f"Null dist (n={len(null_dist)}): 95th percentile = {crit}")
    print(f"Permutation p-value (proportion null >= observed) = {p_corr:.4f}")
    print(f"Significant by 95th-percentile rule? {'YES' if significant_by_crit else 'NO'}")

    # Save observed per-timepoint p-values
    out_csv = output_dir / f"{label.replace(' ', '_')}_per_time_pvals.csv"
    pd.DataFrame({"Time": time_axis, "pval": obs_pvals}).to_csv(out_csv, index=False)

    # Save histogram of null distribution with observed line and critical value
    plt.figure(figsize=(8, 5))
    plt.hist(null_dist, bins=max(10, min(50, len(null_dist)//5)), alpha=0.7, color="gray")
    plt.axvline(obs_max_cluster, color="red", linestyle="--", linewidth=2, label=f"Observed = {obs_max_cluster}")
    plt.axvline(crit, color="blue", linestyle=":", linewidth=2, label=f"95th pct = {crit}")
    plt.xlabel("Maximum cluster size (consecutive significant timepoints)")
    plt.ylabel("Frequency")
    plt.title(f"{label}\nCorrected p = {p_corr:.4f} | Significant: {significant_by_crit}")
    plt.legend()
    plt.tight_layout()
    out_png = output_dir / f"{label.replace(' ', '_')}_FWER_hist.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    # summary dict
    return {
        "label": label,
        "obs_max_cluster": int(obs_max_cluster),
        "n_perm": int(len(null_dist)),
        "crit_95": int(crit),
        "p_corr": float(p_corr),
        "significant_by_crit": significant_by_crit,
        "out_csv": str(out_csv),
        "out_hist": str(out_png)
    }

# run the four analyses (time window 0-10s )

analysis_df = df[(df["Time"] >= 0) & (df["Time"] <= 10)]

summaries = []

# 1) Does ArousalType improve the model beyond Signal alone?
summaries.append(fwer_analysis(
    analysis_df,
    label="ArousalType_vs_SignalOnly",
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ Signal",
    time_axis=time_axis,
    n_perm=500,
    alpha=0.05,
    seed=123
))

# 2) Does Signal type improve the model beyond ArousalType alone?
summaries.append(fwer_analysis(
    analysis_df,
    label="Signal_vs_ArousalOnly",
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ ArousalType",
    time_axis=time_axis,
    n_perm=500,
    alpha=0.05,
    seed=124
))

# 3) Active (Sustained+Transient) vs Loss
df_pooled = analysis_df.copy()
df_pooled["ArousalGroup"] = df_pooled["ArousalType"].replace({"Sustained": "Active", "Transient": "Active", "Loss": "Loss"})
summaries.append(fwer_analysis(
    df_pooled,
    label="Active_vs_Loss",
    formula_full="SignalChange ~ Signal + ArousalGroup",
    formula_reduced="SignalChange ~ Signal",
    time_axis=time_axis,
    n_perm=500,
    alpha=0.05,
    seed=125
))

# 4) Sustained vs Transient (drop Loss)
df_st = analysis_df[analysis_df["ArousalType"].isin(["Sustained", "Transient"])]
summaries.append(fwer_analysis(
    df_st,
    label="Sustained_vs_Transient",
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ Signal",
    time_axis=time_axis,
    n_perm=500,
    alpha=0.05,
    seed=126
))

# Save summary CSV
summary_df = pd.DataFrame(summaries)
summary_df.to_csv(output_dir / "FWER_drop_predictors_summary.csv", index=False)
print(f"\nSaved summary to {output_dir / 'FWER_drop_predictors_summary.csv'}")
