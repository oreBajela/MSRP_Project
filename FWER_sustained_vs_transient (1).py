"""
FWER_sustained_vs_transient.py

This script tests how dopamine- and norepinephrine-sensitive fMRI
signals (D1, D2, NE) differ between sustained vs transient arousal events
using time-resolved mixed-effects models + permutation-based cluster FWER.

Two sets of analysis are run:
1. Arousal Comparison (Main Analysis):
   - Compares signal change between sustained and transient arousal
     at each timepoint for D1, D2, and NE separately.
   - Uses term: ArousalType[T.Sustained] (Sustained relative to Transient).
   - Applies cluster-based permutation FWER correction by scrambling Time
     labels within each Subject.

2. Signal Comparison (Secondary Analysis):
   - Uses D1 as the reference (baseline).
   - Tests whether D2 and NE responses differ from D1 across time.
   - Uses terms: Signal[T.D2], Signal[T.NE].


Outputs:
    - Per-signal CSVs of observed p-values and coefficients for the effect term
    - Histogram plots of null max-cluster sizes vs observed
    - Summary CSV with observed cluster size and corrected p-value
    - Sanity-check plots / files for injected-effect tests
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import os
import copy
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")  # mixedlm throws many convergence warnings


# Configuration

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

signals_in_files = ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]
signals = ["D1", "D2", "NE"]
arousal_types = ["Transient", "Sustained"]   # only analyze these two for main test

output_dir = Path("mixed_model_results_with_FWER")
output_dir.mkdir(exist_ok=True, parents=True)

# model fit at single timepoint
def fit_mixedlm_pval(df_time, formula, group_col, term):
    """
    Fit mixedlm on df_time; return dict with coef, ci_low, ci_high, pval for given term.
    If the term isn't present or the model fails, returns None.
    """
    if df_time.shape[0] == 0:
        return None
    # term must be a column in model design (e.g'ArousalType[T.Sustained]' or 'Signal[T.D2]')
    try:
        model = smf.mixedlm(formula, df_time, groups=df_time[group_col])
        res = model.fit()
    except Exception:
        return None

    if term not in res.params.index:
        return None

    coef = float(res.params[term])
    ci_low, ci_high = res.conf_int().loc[term].values
    pval = float(res.pvalues[term])
    return {"coef": coef, "ci_low": ci_low, "ci_high": ci_high, "pval": pval}


# Compute observed pvals & cluster size 
def get_observed_pvals_and_maxcluster(df_sub, time_axis, formula, group_col, term, alpha=0.05):
    """
    For each timepoint in time_axis, fit the mixed model and collect p-values
    (and coefficient/CI). Compute observed maximum consecutive-significant-cluster size.
    """
    pvals = []
    coefs = []
    ci_lows = []
    ci_highs = []
    for t in time_axis:
        df_time = df_sub[df_sub["Time"] == t]
        res = fit_mixedlm_pval(df_time, formula, group_col, term)
        if res is None:
            pvals.append(1.0)
            coefs.append(np.nan)
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)
        else:
            pvals.append(res["pval"])
            coefs.append(res["coef"])
            ci_lows.append(res["ci_low"])
            ci_highs.append(res["ci_high"])

    pvals = np.array(pvals)
    sig = pvals < alpha

    # find consecutive clusters
    clusters = []
    cur = 0
    for s in sig:
        if s:
            cur += 1
        else:
            if cur > 0:
                clusters.append(cur)
            cur = 0
    if cur > 0:
        clusters.append(cur)

    max_cluster = max(clusters) if clusters else 0

    results_df = pd.DataFrame({
        "Time": time_axis,
        "Coef": coefs,
        "CI_low": ci_lows,
        "CI_high": ci_highs,
        "Pval": pvals
    })
    return results_df, max_cluster


def permutation_null_maxclusters_from_pvalues(pvals, n_perm=500, alpha=0.05, seed=None):
    """
    Build null distribution of maximum cluster sizes by shuffling the observed
    p-value vector (length T) without re-fitting any models.
    """
    rng = np.random.default_rng(seed)
    T = len(pvals)
    sig_threshold = alpha

    def max_cluster(sig_binary):
        clusters, run = [], 0
        for s in sig_binary:
            if s: run += 1
            else:
                if run > 0: clusters.append(run)
                run = 0
        if run > 0: clusters.append(run)
        return max(clusters) if clusters else 0

    null_max = []
    for _ in range(n_perm):
        shuffled = rng.permutation(pvals)
        sig = shuffled < sig_threshold
        null_max.append(max_cluster(sig))

    return np.array(null_max)



#  FWER wrapper for a single analysis 
def run_fwer_for_term(df_sub, time_axis, formula, group_col, term, outprefix, n_perm=500, alpha=0.05):
    """
    Run observed analysis + permutation-based FWER for a single df_sub and a model term.
    Saves results CSV + histogram and returns summary info.
    """
    print(f"Running observed analysis for term '{term}' ...")
    obs_df, obs_max = get_observed_pvals_and_maxcluster(df_sub, time_axis, formula, group_col, term, alpha=alpha)

    print(f"Observed max cluster size = {obs_max}. Building null distribution with {n_perm} permutations ...")
    null_dist = permutation_null_maxclusters_from_pvalues(obs_df["Pval"].values,
                                                      n_perm=n_perm, alpha=alpha)

    # corrected p
    p_corr = (np.sum(null_dist >= obs_max) + 1) / (len(null_dist) + 1)

    # Save observed per-timepoint CSV
    csv_path = output_dir / f"{outprefix}_per_time_results.csv"
    obs_df.to_csv(csv_path, index=False)

    # Save histogram
    plt.figure(figsize=(8, 5))
    plt.hist(null_dist, bins=30, alpha=0.7, label="Null distribution")
    plt.axvline(obs_max, color="red", linestyle="--", linewidth=2, label=f"Observed = {obs_max}")
    plt.xlabel("Maximum cluster size")
    plt.ylabel("Frequency")
    plt.title(f"FWER Null vs Observed ({outprefix})\nCorrected p = {p_corr:.4f}")
    plt.legend()
    plt.tight_layout()
    hist_path = output_dir / f"{outprefix}_FWER_hist.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()

    summary = {
        "outprefix": outprefix,
        "term": term,
        "observed_max_cluster": int(obs_max),
        "n_perm": len(null_dist),
        "p_corr": float(p_corr),
        "csv": str(csv_path),
        "hist": str(hist_path)
    }
    return summary


# - Sanity check data generators 
def generate_injected_arousal_effect(n_subjects=12, n_trials_per_subj=6, n_timepoints=50,
                                     effect_start=20, effect_len=6, effect_size=1.5):
    """
    Generate fake dataset with two arousal types. Inject a sustained > transient
    effect across a contiguous block of timepoints.
    """
    rows = []
    time_axis_local = np.arange(n_timepoints)
    for subj in range(n_subjects):
        for trial in range(n_trials_per_subj):
            ar = np.random.choice(["Transient", "Sustained"], p=[0.5, 0.5])
            base = np.random.randn(n_timepoints) * 0.5
            if ar == "Sustained":
                base[effect_start:effect_start + effect_len] += effect_size
            for t_idx, t in enumerate(time_axis_local):
                rows.append({
                    "Subject": f"sub-{subj:02d}",
                    "Signal": "Injected",
                    "ArousalType": ar,
                    "Time": t,
                    "SignalChange": base[t_idx]
                })
    df_inj = pd.DataFrame(rows)
    # ensure time axis matches the rest of pipeline (float times)
    df_inj["Time"] = df_inj["Time"].astype(float)
    return df_inj, time_axis_local.astype(float)


def generate_injected_signal_effect(n_subjects=12, n_trials_per_subj=6, n_timepoints=50,
                                    effect_start=20, effect_len=6, effect_size=1.5):
    """
    Generate fake dataset with three signals where D2 has an injected effect vs D1
    across a contiguous block of timepoints. Used for signal-comparison sanity check.
    """
    rows = []
    time_axis_local = np.arange(n_timepoints)
    signals_local = ["D1", "D2", "NE"]
    for subj in range(n_subjects):
        for trial in range(n_trials_per_subj):
            # each trial choose a signal (approx balanced)
            sig = np.random.choice(signals_local)
            base = np.random.randn(n_timepoints) * 0.5
            if sig == "D2":
                base[effect_start:effect_start + effect_len] += effect_size
            for t_idx, t in enumerate(time_axis_local):
                rows.append({
                    "Subject": f"sub-{subj:02d}",
                    "Signal": sig,
                    "ArousalType": np.random.choice(["Transient", "Sustained"]),  # irrelevant here
                    "Time": t,
                    "SignalChange": base[t_idx]
                })
    df_inj = pd.DataFrame(rows)
    df_inj["Time"] = df_inj["Time"].astype(float)
    return df_inj, time_axis_local.astype(float)


# real data reading
def load_real_data(base_dir, subjects, signals_in_files, selected_arousals, time_axis):
    """
    Load peri-trial .npz files and return long-format DataFrame with columns:
    Subject, Signal (D1/D2/NE), ArousalType, Time, SignalChange
    """
    rows = []
    for subj in subjects:
        fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
        if not fpath.exists():
            print(f"Missing: {fpath}")
            continue
        data = np.load(fpath, allow_pickle=True)["peri_trials"].item()
        for sig_key, sig_short in zip(signals_in_files, signals):
            for ar in selected_arousals:
                trials = data.get("with_reg", {}).get(sig_key, {}).get(ar, None)
                if trials is None or len(trials) == 0:
                    continue
                for trial in trials:
                    for t_idx, t_val in enumerate(time_axis):
                        rows.append({
                            "Subject": subj,
                            "Signal": sig_short,
                            "ArousalType": ar,
                            "Time": float(t_val),
                            "SignalChange": float(trial[t_idx])
                        })

    df = pd.DataFrame(rows)
    # ensure categorical ordering
    df["ArousalType"] = pd.Categorical(df["ArousalType"], categories=arousal_types)
    df["Signal"] = pd.Categorical(df["Signal"], categories=["D1", "D2", "NE"])
    return df



# main analysis pipeline

def main_pipeline(run_real=True, n_perm=500, alpha=0.05):
    summary_rows = []

    # sanity checks first 
    print("Running sanity checks with injected effects (quick tests)...")
    df_ar_inj, ta_ar = generate_injected_arousal_effect(n_subjects=10, n_trials_per_subj=6, n_timepoints=50,
                                                        effect_start=20, effect_len=6, effect_size=2.0)
    # run FWER for injected arousal effect (Injected signal, term = ArousalType[T.Sustained])
    s = run_fwer_for_term(df_ar_inj, ta_ar, formula="SignalChange ~ ArousalType",
                          group_col="Subject", term="ArousalType[T.Sustained]",
                          outprefix="sanity_injected_arousal", n_perm=200, alpha=alpha)
    print("Sanity (arousal injected) summary:", s)
    summary_rows.append(s)

    df_sig_inj, ta_sig = generate_injected_signal_effect(n_subjects=10, n_trials_per_subj=6, n_timepoints=50,
                                                         effect_start=20, effect_len=6, effect_size=2.0)
    # run FWER for injected signal effect: term Signal[T.D2]
    s2 = run_fwer_for_term(df_sig_inj, ta_sig, formula="SignalChange ~ Signal",
                           group_col="Subject", term="Signal[T.D2]",
                           outprefix="sanity_injected_signal_D2", n_perm=200, alpha=alpha)
    print("Sanity (signal injected) summary:", s2)
    summary_rows.append(s2)

    # real data analysis
    if run_real:
        print("\nLoading real data ...")
        df_real = load_real_data(base_dir, subjects, signals_in_files, arousal_types, time_axis)
        print(f"Real data loaded: {df_real.shape[0]} rows")

        # A) Arousal comparison: per signal, model SignalChange ~ ArousalType
        for sig in signals:
            print(f"\n=== Arousal comparison for signal: {sig} ===")
            df_sub = df_real[df_real["Signal"] == sig].copy()
            if df_sub.shape[0] == 0:
                print(f"Skipping {sig}: no data")
                continue
            # term name expected in result params
            term = "ArousalType[T.Sustained]"
            outprefix = f"{sig}_Arousal_Sustained_vs_Transient"
            summary = run_fwer_for_term(df_sub, time_axis, formula="SignalChange ~ ArousalType",
                                       group_col="Subject", term=term, outprefix=outprefix,
                                       n_perm=n_perm, alpha=alpha)
            summary_rows.append(summary)

            # also save a quick plot of uncorrected coef/CI across time
            res_df, _ = get_observed_pvals_and_maxcluster(df_sub, time_axis, "SignalChange ~ ArousalType", "Subject", term, alpha=alpha)
            plt.figure(figsize=(10, 4))
            plt.plot(res_df["Time"], res_df["Coef"], label="Sustained vs Transient")
            plt.fill_between(res_df["Time"], res_df["CI_low"], res_df["CI_high"], alpha=0.25)
            plt.axhline(0, linestyle="--", color="k")
            plt.title(f"{sig}: ArousalType Effect (uncorrected)")
            plt.xlabel("Time (s)")
            plt.ylabel("Coefficient")
            plt.tight_layout()
            plt.savefig(output_dir / f"{sig}_arousal_uncorrected_coef.png", dpi=300)
            plt.close()

        # B) Signal comparison: model SignalChange ~ Signal across all signals
        print("\n=== Signal comparison (D2 & NE vs D1) across all signals ===")
        df_all = df_real.copy()
        # run for each term of interest separately (D2 vs D1 and NE vs D1)
        for term in ["Signal[T.D2]", "Signal[T.NE]"]:
            outprefix = f"AllSignals_{term.replace(':','').replace('[','').replace(']','')}"
            summary = run_fwer_for_term(df_all, time_axis, formula="SignalChange ~ Signal",
                                       group_col="Subject", term=term, outprefix=outprefix,
                                       n_perm=n_perm, alpha=alpha)
            summary_rows.append(summary)

            # save uncorrected coef plot for term
            res_df, _ = get_observed_pvals_and_maxcluster(df_all, time_axis, "SignalChange ~ Signal", "Subject", term, alpha=alpha)
            plt.figure(figsize=(10, 4))
            plt.plot(res_df["Time"], res_df["Coef"], label=term)
            plt.fill_between(res_df["Time"], res_df["CI_low"], res_df["CI_high"], alpha=0.25)
            plt.axhline(0, linestyle="--", color="k")
            plt.title(f"{term} (uncorrected coef)")
            plt.xlabel("Time (s)")
            plt.ylabel("Coefficient")
            plt.tight_layout()
            plt.savefig(output_dir / f"{outprefix}_uncorrected_coef.png", dpi=300)
            plt.close()

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "FWER_summary.csv", index=False)
    print(f"\nAll done. Summary written to {output_dir / 'FWER_summary.csv'}")



# run main pipeline

if __name__ == "__main__":
    # set run_real=False to only run sanity checks 
    main_pipeline(run_real=True, n_perm=500, alpha=0.05)

