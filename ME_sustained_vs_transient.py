'''
Thiis script tests how dopamine and norepinephrine-sensitive fMRI 
signals (D1, D2, NE) differ between sustained vs transient arousal events 
using time-resolved mixed effects models. 

Two sets of analyses are run:
1. Arousal Comparison (Main Analysis):
   - Compares signal change between sustained and transient arousal 
     at each timepoint for D1, D2, and NE separately.
   - Saves results and plots the effect of sustained vs transient arousal.

2. **Signal Comparison (Secondary Analysis):**
   - Uses D1 as the reference (baseline).
   - Tests whether D2 and NE responses differ from D1 across time.

Usage:
    python ME_sustained_vs_transient.py

Inputs:
    - Subject-specific peri-event trial files (`sub-XXXX_peri_trials.npz`)
      located in the base_dir.

Outputs:
    - CSV files of mixed model results
    - Figures showing effect sizes and confidence intervals:
        * Arousal Type Effect (sustained vs transient, per signal + combined)
        * Signal Comparison (D2 and NE relative to D1 baseline)
'''



import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")



# Configuration
# Base directory where subject data are stored
base_dir = "/home/vob3"

#List of subjects to include
subjects = [  
    "sub-racsleep04b", "sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11",
    "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", "sub-racsleep20",
    "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30",
    "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
    "sub-racsleep38"
]

#fmri parameters
TR = 2.22
window = [-15, 20]
time_axis = np.linspace(window[0], window[1], int((window[1] - window[0]) / TR) + 1)

# signals and arousal events types to analyse
signals = ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]
arousal_types = ['Sustained', 'Transient']  

# Load long DataFrame 
rows = []

for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing: {fpath}")
        continue

    print(f"Loading {subj}")
    data = np.load(fpath, allow_pickle=True)['peri_trials'].item()

    for signal in signals:
        for arousal in arousal_types:
            trials = data.get("with_reg", {}).get(signal, {}).get(arousal, None)
            if trials is None or len(trials) == 0:
                continue

            for trial in trials:
                for t_idx, time in enumerate(time_axis):
                    rows.append({
                        "Subject": subj,
                        "Signal": signal,
                        "ArousalType": arousal,
                        "Time": time,
                        "SignalChange": trial[t_idx]
                    })

df = pd.DataFrame(rows)
df["ArousalType"] = pd.Categorical(df["ArousalType"], categories=["Transient", "Sustained"])
df["Signal"] = df["Signal"].replace({
    "D1_with_reg_PSC": "D1",
    "D2_with_reg_PSC": "D2",
    "Norepinephrine_with_reg": "NE"
})
df["Signal"] = pd.Categorical(df["Signal"], categories=["D1", "D2", "NE"])

print(f"\nFull data loaded: {df.shape} rows")
print(df.head())

#  Mixed Effects Model Per Time Point
def run_mixed_model_timewise(df, formula, group_var="Subject", time_var="Time"):
     '''
    Runs mixed effects models at each unique timepoint in the dataset.

    Args:
        df (pd.DataFrame): Data in long format containing at least columns 
            for signal change, time, and grouping variable (subject).
        formula (str): formula for mixedlm (e.g. "SignalChange ~ ArousalType").
        group_var (str): Column used as the grouping variable (default = "Subject").
        time_var (str): Column name that indexes timepoints (default = "Time").

    Returns:
        pd.DataFrame: DataFrame containing coefficient estimates, 
            confidence intervals, and p-values for each predictor term 
            across time.
    '''

    times = sorted(df[time_var].unique())
    results = []

    for t in times:
        sub_df = df[df[time_var] == t]
        try:
            model = smf.mixedlm(formula, sub_df, groups=sub_df[group_var])
            result = model.fit()

            # collect results for all predictor terms skip intercept)
            for term in result.params.index[1:]:
                ci = result.conf_int().loc[term]
                results.append({
                    "Time": t,
                    "Term": term,
                    "Coef": result.params[term],
                    "CI_low": ci[0],
                    "CI_high": ci[1],
                    "Pval": result.pvalues[term]
                })
        except Exception as e:
            print(f"Model failed at t={t:.2f}s: {e}")
            continue

    return pd.DataFrame(results)




# output directory
output_dir = "mixed_model_results"
os.makedirs(output_dir, exist_ok=True)



# Mixed Model Analyses: Sustained vs Transient Arousal
# Run mixed models per signal, comparing sustained vs transient arousal.
# Formula: SignalChange ~ ArousalType
# Predictor of interest: ArousalType[T.Sustained] = sustained relative to transient

# Arousal Type Model (per signal)
for signal in ["D1", "D2", "NE"]:
    df_sig = df[df["Signal"] == signal].copy()
    res_arousal = run_mixed_model_timewise(df_sig, "SignalChange ~ ArousalType")
    res_arousal.to_csv(f"{output_dir}/{signal}_arousal_mixed_model.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 4))
    subset = res_arousal[res_arousal["Term"] == "ArousalType[T.Sustained]"]
    plt.plot(subset["Time"], subset["Coef"], label="Sustained vs Transient", color='purple')
    plt.fill_between(subset["Time"], subset["CI_low"], subset["CI_high"], alpha=0.3, color='purple')
    plt.axhline(0, linestyle='--', color='black')
    plt.title(f"Arousal Type Effect on {signal}")
    plt.xlabel("Time (s)")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{signal}_arousalType_plot.png")
    plt.close()
# Collect all results first
arousal_model_results = {}
for signal in ["D1", "D2", "NE"]:
    df_sig = df[df["Signal"] == signal].copy()
    res_arousal = run_mixed_model_timewise(df_sig, "SignalChange ~ ArousalType")
    res_arousal.to_csv(f"{output_dir}/{signal}_arousal_mixed_model.csv", index=False)
    arousal_model_results[signal] = res_arousal

# Plot all three in one combined figure
fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True)  # share X-axis

for i, (ax, signal) in enumerate(zip(axes, ["D1", "D2", "NE"])):
    subset = arousal_model_results[signal]
    subset = subset[subset["Term"] == "ArousalType[T.Sustained]"]
    ax.plot(subset["Time"], subset["Coef"], label="Sustained vs Transient", color='purple')
    ax.fill_between(subset["Time"], subset["CI_low"], subset["CI_high"], alpha=0.3, color='purple')
    ax.axhline(0, linestyle='--', color='black', linewidth=1)
    
    ax.set_title(signal)
    ax.set_ylabel("Coefficient")  # all plots have Y label

    # Hide X-axis tick labels for all but bottom plot
    if i < len(axes) - 1:
        ax.tick_params(labelbottom=False)  # hides tick labels but keeps ticks
    else:
        ax.set_xlabel("Time (s)")  # bottom plot gets xlabel

fig.suptitle("Arousal Type Effect on Neurotransmitter Signals", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{output_dir}/combined_arousalType_effects.png", dpi=300)
plt.close()







# Mixed Model Analyses: Signal Comparison (D1 baseline)
# Here, we treat D1 as the baseline reference and test whether D2 and NE
# differ in peri-arousal response compared to D1.
# Formula: SignalChange ~ Signal
# Predictors of interest: Signal[T.D2], Signal[T.NE]

res_signal = run_mixed_model_timewise(df, "SignalChange ~ Signal")
res_signal.to_csv(f"{output_dir}/D1_baseline_mixed_model.csv", index=False)

# Plot D2 vs D1 and NE vs D1 over time
plt.figure(figsize=(10, 4))
for term, color in zip(["Signal[T.D2]", "Signal[T.NE]"], ["teal", "orange"]):
    subset = res_signal[res_signal["Term"] == term]
    plt.plot(subset["Time"], subset["Coef"], label=term, color=color)
    plt.fill_between(subset["Time"], subset["CI_low"], subset["CI_high"], alpha=0.3, color=color)

plt.axhline(0, linestyle='--', color='black')
plt.title("D2 and NE Relative to D1")
plt.xlabel("Time (s)")
plt.ylabel("Coefficient")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/D1_baseline_comparison_plot.png")
plt.close()
