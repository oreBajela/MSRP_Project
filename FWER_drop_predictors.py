"""
FWER_drop_predictors.py

This script extends the drop-predictors mixed effects analysis by adding
permutation-based Family-Wise Error Rate (FWER) correction.

Original purpose of drop_predictors was to run group-level statistical analyses on peri-arousal fMRI data. It combines
subject-level peri-trial data into a long-format DataFrame and uses mixed-effects
models to test hypotheses about signal and arousal type effects by dropping certain
predictors and comparing model fits.

Signals analyzed:
    - D1 receptor
    - D2 receptor
    - Norepinephrine

Arousal types:
    - Sustained
    - Transient
    - Loss

Main Analyses (Likelihood Ratio Tests):
---------------------------------------
1. Does ArousalType improve the model beyond Signal alone?
2. Does Signal type improve the model beyond ArousalType alone?
3. Do Active arousals (Sustained + Transient) differ from Loss?
4. Do Sustained vs Transient responses differ (excluding Loss)?

New FWER Extension:
---------------------------------------
- For each model comparison,compute likelihood ratio (LR) statistics
  at each peri-event timepoint.
- Clusters of consecutive significant LR tests are identified.
- Null distributions of max cluster size are built by permuting time order
  within each Signal × ArousalType series.
- Corrected p-values are computed from the null distribution.

Outputs:
    - output of standard model comparisons
    - CSV files of LR test results per timepoint
    - Histograms of null vs observed cluster sizes (FWER correction)
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import os


# CONFIGURATION


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

output_dir = "drop_predictors_FWER_results"
os.makedirs(output_dir, exist_ok=True)


# load data


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
                        "Time": time_axis[t_idx],
                        "SignalChange": value
                    })

df = pd.DataFrame(rows).dropna()
print(f"\nLoaded data: {df.shape} rows")

# functions

def compare_models_lr(df_sub, formula_full, formula_reduced):
    """
    Fit full and reduced models at one timepoint, return likelihood ratio test.
    """
    try:
        model_full = smf.mixedlm(formula_full, df_sub, groups=df_sub["Subject"])
        result_full = model_full.fit()

        model_reduced = smf.mixedlm(formula_reduced, df_sub, groups=df_sub["Subject"])
        result_reduced = model_reduced.fit()

        lr_stat = 2 * (result_full.llf - result_reduced.llf)
        df_diff = result_full.df_modelwc - result_reduced.df_modelwc
        p_value = 1 - stats.chi2.cdf(lr_stat, df=df_diff)
        return lr_stat, df_diff, p_value
    except:
        return np.nan, np.nan, np.nan

def run_timewise_lr(df, formula_full, formula_reduced, time_axis, alpha=0.05):
    """
    Run LR tests at all timepoints, compute clusters of significance.
    """
    lr_stats, pvals = [], []
    for t in time_axis:
        sub_df = df[df["Time"] == t]
        lr, df_diff, p = compare_models_lr(sub_df, formula_full, formula_reduced)
        lr_stats.append(lr)
        pvals.append(p)

    pvals = np.array(pvals)
    sig = pvals < alpha

    # Find max cluster size
    clusters, cluster_size = [], 0
    for val in sig:
        if val:
            cluster_size += 1
        else:
            if cluster_size > 0:
                clusters.append(cluster_size)
            cluster_size = 0
    if cluster_size > 0:
        clusters.append(cluster_size)

    max_cluster = max(clusters) if clusters else 0
    return pvals, max_cluster

def permutation_test(df, formula_full, formula_reduced, time_axis, n_perm=500, alpha=0.05):
    """
    Build null distribution of max cluster sizes by permuting time order.
    """
    null_max_clusters = []
    for i in range(n_perm):
        df_perm = df.copy()
        df_perm["Time"] = np.random.permutation(df_perm["Time"].values)

        _, max_cluster = run_timewise_lr(df_perm, formula_full, formula_reduced, time_axis, alpha)
        null_max_clusters.append(max_cluster)

        if (i + 1) % 50 == 0:
            print(f"Permutation {i+1}/{n_perm} done.")
    return np.array(null_max_clusters)

def fwer_analysis(df, label, formula_full, formula_reduced, time_axis, n_perm=500, alpha=0.05):
    """
    Full pipeline: observed LR clusters + permutation null + corrected p.
    """
    print(f"\n=== Running FWER analysis: {label} ===")
    pvals, obs_max_cluster = run_timewise_lr(df, formula_full, formula_reduced, time_axis, alpha)
    null_dist = permutation_test(df, formula_full, formula_reduced, time_axis, n_perm, alpha)

    p_corr = (np.sum(null_dist >= obs_max_cluster) + 1) / (len(null_dist) + 1)
    print(f"Observed cluster size = {obs_max_cluster}, Corrected p = {p_corr:.4f}")

    # Save results
    results = pd.DataFrame({"Time": time_axis, "pval": pvals})
    results.to_csv(f"{output_dir}/{label.replace(' ', '_')}.csv", index=False)

    plt.figure(figsize=(8,5))
    plt.hist(null_dist, bins=30, alpha=0.7, color="gray")
    plt.axvline(obs_max_cluster, color="red", linestyle="--", label=f"Observed = {obs_max_cluster}")
    plt.title(f"{label}\nFWER-corrected p = {p_corr:.4f}")
    plt.xlabel("Max cluster size")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label.replace(' ', '_')}_FWER.png", dpi=300)
    plt.close()


#run analysis


analysis_df = df[(df["Time"] >= 0) & (df["Time"] <= 10)]

# 1. Does ArousalType improve the model?
fwer_analysis(
    analysis_df,
    label="ArousalType_vs_SignalOnly",
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ Signal",
    time_axis=time_axis
)

# 2. Does Signal type improve the model?
fwer_analysis(
    analysis_df,
    label="Signal_vs_ArousalOnly",
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ ArousalType",
    time_axis=time_axis
)

# 3. Active (Sustained+Transient) vs Loss
df_pooled = analysis_df.copy()
df_pooled["ArousalGroup"] = df_pooled["ArousalType"].replace(
    {"Sustained": "Active", "Transient": "Active", "Loss": "Loss"}
)
fwer_analysis(
    df_pooled,
    label="Active_vs_Loss",
    formula_full="SignalChange ~ Signal + ArousalGroup",
    formula_reduced="SignalChange ~ Signal",
    time_axis=time_axis
)

# 4. Sustained vs Transient (Loss excluded)
df_st = analysis_df[analysis_df["ArousalType"].isin(["Sustained", "Transient"])]
fwer_analysis(
    df_st,
    label="Sustained_vs_Transient",
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ Signal",
    time_axis=time_axis
)

