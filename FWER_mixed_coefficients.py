"""

Permutation-Based FWER Correction for LME Results


This script fits linear mixed-effects (LME) models to peri-event fMRI
signal change time series at each timepoint. To account for multiple
comparisons across many timepoints, it performs permutation based
Family-Wise Error Rate (FWER) control:

1. Observed analysis:
   - Fit LME at each timepoint: PSC ~ ArousalType + (1|Subject)
   - Collect p-values and find largest cluster of consecutive
     significant timepoints.

2. Null distribution:
   - Shuffle arousal labels within the dataset.
   - Recompute cluster sizes across N permutations.
   - Build distribution of maximum cluster sizes expected by chance.

3. Corrected inference:
   - Compare observed cluster size to null distribution.
   - Save corrected results to CSV and generate plots.

Inputs:
-------
- Peri-trial .npz files (one per subject) containing trial × time data
  for signals (D1, D2, NE) and arousal types.
- Output directory (created automatically).

Outputs:
--------
- CSV of observed p-values per timepoint.
- Histogram plot showing observed vs null cluster size distribution.
- Printed corrected p-value.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt

# fit LME at a single timepoint
def fit_model_at_timepoint(df, time_val):
    """
    Fit a linear mixed-effects (LME) model at a single timepoint:
        PSC ~ ArousalType + (1 | Subject)

    Returns the p-value for Sustained vs other (baseline = Transient or Loss).
    """
    df_time = df[df["Time"] == time_val]
    if df_time["ArousalType"].nunique() < 2:
        return None
    try:
        model = smf.mixedlm("PSC ~ ArousalType", df_time, groups=df_time["Subject"])
        result = model.fit()
        return result.pvalues.get("ArousalType[T.Sustained]", np.nan)
    except:
        return None



# compute observed clusters
def get_observed_clusters(df, time_axis, alpha=0.05):
    """
    Run LME at all timepoints, return p-values and largest cluster size.
    """
    pvals = []
    for t in time_axis:
        p = fit_model_at_timepoint(df, t)
        pvals.append(p if p is not None else 1.0)

    pvals = np.array(pvals)
    sig = pvals < alpha

    clusters = []
    cluster_size = 0
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


# Permutation test
def permutation_test(df, time_axis, n_perm=1000, alpha=0.05):
    """
    Shuffle ArousalType labels to generate null distribution of
    maximum cluster sizes.
    """
    null_max_clusters = []
    for i in range(n_perm):
        df_perm = df.copy()
        df_perm["ArousalType"] = np.random.permutation(df_perm["ArousalType"].values)
        _, max_cluster = get_observed_clusters(df_perm, time_axis, alpha=alpha)
        null_max_clusters.append(max_cluster)
        if (i+1) % 50 == 0:
            print(f"Permutation {i+1}/{n_perm} done.")
    return np.array(null_max_clusters)



# Main execution
if __name__ == "__main__":

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

    selected_signals = ["D2_with_reg_PSC"]  # example: run for D2 first
    arousal_types = ['Sustained', 'Transient', 'Loss']


    # reading long format data 

    all_rows = []
    for subj in subjects:
        fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
        if not fpath.exists():
            print(f"Missing: {fpath}")
            continue
        data = np.load(fpath, allow_pickle=True)['peri_trials'].item()
        for signal in selected_signals:
            for arousal_type in arousal_types:
                trials = data.get("with_reg", {}).get(signal, {}).get(arousal_type, None)
                if trials is None or len(trials) == 0:
                    continue
                for trial in trials:
                    for t_idx, t_val in enumerate(time_axis):
                        all_rows.append({
                            "Subject": subj,
                            "ArousalType": arousal_type,
                            "Signal": signal.replace("_with_reg", ""),
                            "Time": t_val,
                            "PSC": trial[t_idx]
                        })

    df_long = pd.DataFrame(all_rows)
    print(f"Data loaded: {df_long.shape} rows")

    #observed cluster analysis
    alpha = 0.05
    obs_pvals, obs_max_cluster = get_observed_clusters(df_long, time_axis, alpha=alpha)
    print(f"Observed max cluster size = {obs_max_cluster}")


    # permutation on null distribution
    null_dist = permutation_test(df_long, time_axis, n_perm=500, alpha=alpha)

    # compute corrected p-value
    p_corr = (np.sum(null_dist >= obs_max_cluster) + 1) / (len(null_dist) + 1)
    print(f"FWER-corrected p = {p_corr:.4f}")


    # Saving my results

    results = pd.DataFrame({
        "Time": time_axis,
        "pval": obs_pvals
    })
    results.to_csv("FWER_corrected_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(null_dist, bins=30, alpha=0.7, color="gray", label="Null distribution")
    plt.axvline(obs_max_cluster, color="red", linestyle="--", linewidth=2,
                label=f"Observed = {obs_max_cluster}")
    plt.xlabel("Maximum cluster size")
    plt.ylabel("Frequency")
    plt.title("Permutation FWER Correction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("FWER_cluster_hist.png", dpi=300)
    plt.close()


