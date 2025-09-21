
"""
Permutation-Based FWER Correction for LME Results

This script validates significance of peri-event fMRI signal changes
using permutation-based Family-Wise Error Rate (FWER) control:


- Instead of shuffling ArousalType labels, we apply the permutation
  procedure separately for each peri-event average time series
  (Signal × ArousalType).
- On each permutation, the order of the timepoints is scrambled.
generating a proper null distribution.

Steps:

1. Observed analysis:
   - Fit LME at each timepoint: PSC ~ ArousalType + (1|Subject)
   - Collect p-values and find largest cluster of consecutive
     significant timepoints.

2. Null distribution:
   - For each Signal × ArousalType series, scramble time order N times.
   - Recompute cluster sizes across permutations.
   - Build distribution of maximum cluster sizes expected by chance.

3. Corrected inference:
   - Compare observed cluster size to null distribution.
   - Save corrected results to CSV and generate plots.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 



# Fit LME at a single timepoint
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
    Build null distribution of max cluster sizes by permuting
    the time axis within each Signal × ArousalType series.

    This ensures the FWER procedure is applied separately
    for each peri-event average time series.
    """
    null_max_clusters = []
    for i in range(n_perm):
        df_perm = df.copy()
        # Shuffle the Time labels across the whole dataset
        # while keeping PSC values fixed per trial
        shuffled_times = np.random.permutation(df_perm["Time"].values)
        df_perm["Time"] = shuffled_times

        _, max_cluster = get_observed_clusters(df_perm, time_axis, alpha=alpha)
        null_max_clusters.append(max_cluster)

        if (i + 1) % 50 == 0:
            print(f"Permutation {i+1}/{n_perm} done.")
    return np.array(null_max_clusters)



# Sanity check fucntions
def generate_fake_data(n_subjects=10, n_timepoints=50, signal="D1", arousal="Sustained"):
    """
    Generate fake peri-event dataset for testing.

    - Random Gaussian noise PSC
    - Random Subject IDs
    - Returns df in long format
    """
    all_rows = []
    time_axis = np.arange(n_timepoints)
    for subj in range(n_subjects):
        for t in time_axis:
            all_rows.append({
                "Subject": f"sub-{subj:02d}",
                "ArousalType": arousal,
                "Signal": signal,
                "Time": t,
                "PSC": np.random.randn()
            })
    return pd.DataFrame(all_rows), time_axis


def sanity_check():
    """
    Run a quick sanity check with fake data (random noise).
    Expect corrected p-values - uniform / non-significant.
    """
    df_fake, time_axis = generate_fake_data()
    obs_pvals, obs_max_cluster = get_observed_clusters(df_fake, time_axis)
    null_dist = permutation_test(df_fake, time_axis, n_perm=200)

    p_corr = (np.sum(null_dist >= obs_max_cluster) + 1) / (len(null_dist) + 1)
    print(f"[Sanity check] Observed max cluster = {obs_max_cluster}, Corrected p = {p_corr:.4f}")

    plt.hist(null_dist, bins=30, alpha=0.7, color="gray")
    plt.axvline(obs_max_cluster, color="red", linestyle="--")
    plt.title("Sanity check: Null distribution vs observed")
    plt.savefig("sanity_check.png", dpi=300)
    plt.close()



def run_fwer_analysis_by_signal(df_long, time_axis, output_dir="FWER_results", n_perm=500, alpha=0.05):
    """
    Run FWER correction for each Signal (keeping all ArousalType levels).
    Saves:
    - CSV of p-values for each timepoint
    - Histogram plot of null distribution vs observed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    signals = df_long["Signal"].unique()

    for signal in signals:
        print(f"\n=== Running analysis for {signal} (all ArousalType levels) ===")

        
        df_sub = df_long[df_long["Signal"] == signal]
        if df_sub.empty:
            print(f"Skipping {signal}: no data")
            continue

        # Observed cluster analysis -uses ArousalType as predictor
        obs_pvals, obs_max_cluster = get_observed_clusters(df_sub, time_axis, alpha=alpha)
        print(f"Observed max cluster size = {obs_max_cluster}")

        # Null distribution (permute Time labels within the signal subset)
        null_dist = permutation_test(df_sub, time_axis, n_perm=n_perm, alpha=alpha)

        # Corrected p-value
        p_corr = (np.sum(null_dist >= obs_max_cluster) + 1) / (len(null_dist) + 1)
        print(f"FWER-corrected p = {p_corr:.4f}")

        # Save CSV of raw p-values
        results = pd.DataFrame({
            "Time": time_axis,
            "pval": obs_pvals
        })
        csv_path = output_dir / f"{signal}_FWER_results.csv"
        results.to_csv(csv_path, index=False)

        # Save histogram plot
        plt.figure(figsize=(8, 5))
        plt.hist(null_dist, bins=30, alpha=0.7, color="gray", label="Null distribution")
        plt.axvline(obs_max_cluster, color="red", linestyle="--", linewidth=2,
                    label=f"Observed = {obs_max_cluster}")
        plt.xlabel("Maximum cluster size")
        plt.ylabel("Frequency")
        plt.title(f"FWER Correction: {signal}\nCorrected p = {p_corr:.4f}")
        plt.legend()
        plt.tight_layout()
        fig_path = output_dir / f"{signal}_FWER_hist.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"Saved: {csv_path}, {fig_path}")






# main execution
if __name__ == "__main__":
    sanity_check()
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

    selected_signals = ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]  
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

    run_fwer_analysis_by_signal(df_long, time_axis, output_dir="FWER_results", n_perm=500, alpha=0.05)


