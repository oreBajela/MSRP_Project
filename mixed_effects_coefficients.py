"""
mixed_effects_coefficients.py

Group-level mixed-effects analysis for peri-arousal BOLD responses in the
dopamine project. This script loads per-subject peri-arousal trial matrices
(saved earlier by the plotting/behavior script), stacks them into a long-form
DataFrame, and fits a linear mixed-effects model at each time-point:

    PSC ~ ArousalType + (1 | Subject)

Where PSC is percent signal change for receptor-weighted or anatomical signals,
and ArousalType ∈ {Sustained, Transient, Loss}. By default, Transient is used
as the reference (baseline) level.

Outputs:
  - CSV of time-resolved fixed-effect coefficients and 95% CIs
    (one row per time-point).
  - A PNG plot showing the coefficient time course with confidence bands.

Typical usage:
    python fit_mixed_effects_peri_arousal.py

Requirements:
  - Per-subject NPZ files named "{subject}_peri_trials.npz" located at
    BASE_DIR / subject_id /, each containing a dict "peri_trials" with shape:
        peri_trials["with_reg"][signal][arousal_type] -> array (n_trials, n_timepoints)
  - Files are produced by your subject/behavior scripts in this project.
"""


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib.pyplot as plt

# model at a single time point
def fit_model_at_timepoint(df, time_val):
    """
    Fit a mixed-effects model at a single time-point:
        PSC ~ C(ArousalType, Treatment(reference=baseline)) + (1 | Subject)

    Args:
        df: Long-form DataFrame with columns ["Subject", "ArousalType", "Time", "PSC"].
        time_val: Time (in seconds) at which to fit the model (exact match expected).
        baseline: Category used as the reference level for ArousalType.

    Returns:
        A dict with coefficient(s), CI(s), and p-value(s) for Sustained (and Loss if present),
        or None if the model cannot be fit (e.g., not enough category variation).
        Keys include:
            "Time", "Coef_Sustained", "CI_low_Sustained", "CI_high_Sustained", "pval_Sustained",
            "Coef_Loss", "CI_low_Loss", "CI_high_Loss", "pval_Loss"
    """

    print(f"\n Mixed Model at t = {time_val:.2f}s ")
    df_time = df[df["Time"] == time_val]
    if df_time["ArousalType"].nunique() < 2:
        # Need at least two categories at this time point to estimate contrasts
        print("Insufficient ArousalType variation")
        return None

    try:
        model = smf.mixedlm("PSC ~ ArousalType", df_time, groups=df_time["Subject"])
        result = model.fit()
        print(result.summary())
        return {
            "Time": time_val,
            "Coef": result.fe_params.get("ArousalType[T.Sustained]", np.nan),
            "CI_low": result.conf_int().loc["ArousalType[T.Sustained]", 0] if "ArousalType[T.Sustained]" in result.fe_params else np.nan,
            "CI_high": result.conf_int().loc["ArousalType[T.Sustained]", 1] if "ArousalType[T.Sustained]" in result.fe_params else np.nan,
            "pval": result.pvalues.get("ArousalType[T.Sustained]", np.nan)
        }
    except Exception as e:
        print(f"Model error at t={time_val}: {e}")
        return None


        

base_dir = "/home/vob3"
subjects = ["sub-racsleep04b", "sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11",
            "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", "sub-racsleep20",
            "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30",
            "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
            "sub-racsleep38"]

TR = 2.22
window = [-15, 20]
time_axis = np.linspace(window[0], window[1], int((window[1] - window[0]) / TR) + 1)


selected_signals = ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]
arousal_types = ['Sustained', 'Transient', 'Loss']


all_rows = []

for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing: {fpath}")
        continue

    print(f"Loading {subj}")
    data = np.load(fpath, allow_pickle=True)['peri_trials'].item()

    for signal in selected_signals:
        for arousal_type in arousal_types:
             # Access only the "with_reg" branch for selected signals
            trials = data.get("with_reg", {}).get(signal, {}).get(arousal_type, None)
            if trials is None or len(trials) == 0:
                continue
            # Add one row per time-point for this trial
            for trial in trials:
                for t_idx, t_val in enumerate(time_axis):
                    all_rows.append({
                        "Subject": subj,
                        "ArousalType": arousal_type,
                        "Signal": signal.replace("_with_reg", ""),  # clean name
                        "Time": t_val,
                        "PSC": trial[t_idx]
                    })

df_long = pd.DataFrame(all_rows)
print("\n DataFrame:")
print(df_long.head())


results = []
for t in time_axis:
    res = fit_model_at_timepoint(df_long, t)
    if res is not None:
        results.append(res)

df_results = pd.DataFrame(results)
df_results.to_csv("mixed_effects_coefficients_by_time.csv", index=False)

#time series of coefficients
plt.figure(figsize=(10, 5))
plt.plot(df_results["Time"], df_results["Coef"], label="Coef: Sustained vs other")
plt.fill_between(df_results["Time"], df_results["CI_low"], df_results["CI_high"], alpha=0.3)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Coefficient Estimate")
plt.title("Mixed Effects Model Coefficient Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("coef_timeseries_plot.png")
plt.show()
