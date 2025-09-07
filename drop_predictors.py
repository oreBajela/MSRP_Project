"""
drop_predictors.py

This script runs group-level statistical analyses on peri-arousal fMRI data. 
It combines subject-level peri-trial data into a long-format DataFrame and 
uses mixed-effects models to test hypotheses about signal and arousal type effects by dropping
certain oredictors and comparing the performance of the modesl.

Signals analyzed:
    - D1 receptor
    - D2 receptor
    - Norepinephrine

Arousal types:
    - Sustained
    - Transient
    - Loss

Main analyses:
1. Does **ArousalType** improve the model beyond Signal alone?
2. Does **Signal type** improve the model beyond ArousalType alone?
3. Do **Active arousals (Sustained + Transient)** differ from Loss?
4. Do **Sustained vs Transient** responses differ (excluding Loss)?

Inputs:
    - Subject peri-trials stored in `{subj}_peri_trials.npz`, containing 
      trial-level peri-event arrays by signal and arousal type.

Outputs:
    - Console output of model fits and likelihood ratio tests (χ², df, p-values).
"""


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from statsmodels.stats.anova import anova_lm
import matplotlib
from scipy import stats
matplotlib.use("Agg")



base_dir = "/home/vob3"
subjects = [
    "sub-racsleep04b", "sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11",
    "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", "sub-racsleep20",
    "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30",
    "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
    "sub-racsleep38"
]

def compare_models(df_sub, formula_full, formula_reduced, label):

      """
    Compare two mixed-effects models using likelihood ratio test.

    Parameters:
        df_sub (pd.DataFrame): Data subset for analysis
        formula_full (str): Full model formula (e.g., "SignalChange ~ Signal + ArousalType")
        formula_reduced (str): Reduced model formula
        label (str): Label for printing analysis description
    """

    
    print(f"\n--- Model comparison: {label} ---")
    try:
        model_full = smf.mixedlm(formula_full, df_sub, groups=df_sub["Subject"])
        result_full = model_full.fit()
       
        model_reduced = smf.mixedlm(formula_reduced, df_sub, groups=df_sub["Subject"])
        result_reduced = model_reduced.fit()

        print("\nFull Model:\n", result_full.summary())
        print("\nReduced Model:\n", result_reduced.summary())

        lr_stat = 2 * (result_full.llf - result_reduced.llf)
        df_diff = result_full.df_modelwc - result_reduced.df_modelwc
        p_value = 1 - stats.chi2.cdf(lr_stat, df=df_diff)

        print(f"\nLikelihood Ratio Test: χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.4f}")
    except Exception as e:
        print(f"Model comparison failed: {e}")



TR = 2.22
window = [-15, 20]
time_axis = np.linspace(window[0], window[1], int((window[1] - window[0]) / TR) + 1)


signal_list = ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]
arousal_types = ['Sustained', 'Transient', 'Loss']


rows = []

for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing file: {fpath}")
        continue

    print(f"Loading {fpath}")
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

df = pd.DataFrame(rows)
print("\nLoaded long-format data:")
print(df.head())
df = df.dropna()

# model fomparison functions



# run comparisons 

# analyze 0–10s window
analysis_df = df[(df["Time"] >= 0) & (df["Time"] <= 10)]

# Signal = D1, Arousal = Sustained 

# 1. Compare model WITH vs WITHOUT ArousalType
compare_models(
    analysis_df,
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ Signal",
    label="Does ArousalType improve the model?"
)

# 2. Compare model WITH vs WITHOUT Signal type
compare_models(
    analysis_df,
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ ArousalType",
    label="Does Signal type improve the model?"
)

# 3. Compare pooled Transient+Sustained vs Loss
df_pooled = analysis_df.copy()
df_pooled["ArousalGroup"] = df_pooled["ArousalType"].replace({"Sustained": "Active", "Transient": "Active", "Loss": "Loss"})
compare_models(
    df_pooled,
    formula_full="SignalChange ~ Signal + ArousalGroup",
    formula_reduced="SignalChange ~ Signal",
    label="Do Active (Sustained+Transient) differ from Loss?"
)

# 4. Just compare Sustained vs Transient (drop Loss)
df_st = analysis_df[analysis_df["ArousalType"].isin(["Sustained", "Transient"])]
compare_models(
    df_st,
    formula_full="SignalChange ~ Signal + ArousalType",
    formula_reduced="SignalChange ~ Signal",
    label="Sustained vs Transient only"
)