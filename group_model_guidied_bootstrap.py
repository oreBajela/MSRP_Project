"""
group_model_guided_bootstrap.py formerly groyp_level_2.py

This script performs model-guided bootstrapping of peri-arousal fMRI signals
to generate group-level timecourses with confidence intervals. 

Key Analyses:
1. **Arousal × RegType × Signal Plots (Main Loop)**
   - For each arousal type (Sustained, Transient, Loss), regression type (with_reg, without_reg),
     and signal (GM, Thalamus, D1, D2, NE), runs model-guided bootstrapping across subjects.
   - Produces group mean timecourses with bootstrap-derived confidence intervals.

2. **Signal-wise Plots Across Arousal Types**
   - For each signal within each reg_type, overlays sustained, transient, and loss responses 
     to compare arousal-type differences.

Inputs:
    - Subject-specific peri-event trial files: 
        {subj}_peri_trials.npz (must contain dict["peri_trials"])
    - Expected location: base_dir / subject / file

Outputs:
    - Group-level bootstrap plots (PNG) for:
        * Each arousal × reg_type × signal combination
        * Each signal across arousal types
    - Console logs summarizing missing files, skipped conditions, and bootstrapping progress

Notes:
    - Bootstrapping is model-guided: resamples trial-level data, then averages 
      within subject before group averaging.
    - Confidence intervals are computed via percentiles.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.utils import resample
import pandas as pd
import os


# group boostrap functions
def model_guided_bootstrap_timecourses(trial_dict, n_boot=2000, ci=95):
     """
    Perform model-guided bootstrapping of peri-arousal trial data.

    trial_dict : dict
        Keys = subjects, values = list of trial arrays (n_trials × n_timepoints).
    n_boot : int
        Number of bootstrap iterations.
    ci : int
        Confidence interval (percent).

    Returns
    -------
    mean : ndarray
        Bootstrap mean across subjects per timepoint.
    lower : ndarray
        Lower confidence bound.
    upper : ndarray
        Upper confidence bound.
    """

    subjects = list(trial_dict.keys())
    print(f"Model-guided bootstrapping over {len(subjects)} subjects...")

    timepoints = trial_dict[subjects[0]][0].shape[1]  # assume same shape
    boot_samples = []

    for t in range(timepoints):
        rows = []
        for subj in subjects:
            for arr in trial_dict[subj]:
                for trial in arr:
                    rows.append({'Subject': subj, 'Signal': trial[t]})
        df = pd.DataFrame(rows)

        boot_t = []
        for _ in range(n_boot):
            boot_df = resample(df, replace=True)
            try:
                mean_by_subject = boot_df.groupby("Subject")["Signal"].mean()
                boot_t.append(mean_by_subject.mean())
            except Exception:
                boot_t.append(np.nan)

        boot_t = np.array(boot_t)
        boot_t = boot_t[~np.isnan(boot_t)]
        boot_samples.append(boot_t)

    boot_samples = np.array(boot_samples).T  # (n_boot, time)
    mean = np.mean(boot_samples, axis=0)
    lower = np.percentile(boot_samples, (100 - ci) / 2, axis=0)
    upper = np.percentile(boot_samples, 100 - (100 - ci) / 2, axis=0)

    print(f"Bootstrap result shape: {boot_samples.shape}")
    return mean, lower, upper

    
# configuration
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
n_boot = 2000
ci = 95

# Define signal sets for analysis 
signal_sets = {
    "with_reg": ["GM_PSC", "Thalamus_PSC", "D2_with_reg_PSC", "D1_with_reg_PSC", "Norepinephrine_with_reg"],
    "without_reg": ["GM_PSC", "Thalamus_PSC", "D2_without_reg_PSC", "D1_without_reg_PSC", "Norepinephrine_without_reg"]
}


group_data = {}
#loading subject data 
for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing file: {fpath}")
        continue

    print(f"Loading data from {fpath}")
    data = np.load(fpath, allow_pickle=True)['peri_trials'].item()

   # Organize into nested dict: arousal_type → reg_type → signal → subject

    for reg_type in signal_sets:
        for signal in signal_sets[reg_type]:
            for arousal_type in ['Sustained', 'Transient', 'Loss']:
                trials = data.get(reg_type, {}).get(signal, {}).get(arousal_type, None)
                if trials is None or len(trials) == 0:
                    continue

                group_data.setdefault(arousal_type, {}).setdefault(reg_type, {}).setdefault(signal, {}).setdefault(subj, [])
                group_data[arousal_type][reg_type][signal][subj].append(trials)
                print(f"Loaded: {arousal_type} | {reg_type} | {signal} | {subj} -> {trials.shape}")







# Plot by arousal × reg_type × signal
for arousal_type in group_data:
    for reg_type in group_data[arousal_type]:
        fig, axs = plt.subplots(1, len(group_data[arousal_type][reg_type]), figsize=(15, 4), sharey=True)
        axs = np.atleast_1d(axs)

        for i, signal in enumerate(group_data[arousal_type][reg_type]):
            trial_dict = group_data[arousal_type][reg_type][signal]
            if not trial_dict:
                continue

            print(f"\n==> Bootstrapping {arousal_type} | {reg_type} | {signal}")
            mean, lower, upper = model_guided_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis = np.linspace(window[0], window[1], mean.shape[0])

            axs[i].plot(time_axis, mean, color='blue')
            axs[i].fill_between(time_axis, lower, upper, color='blue', alpha=0.3)
            axs[i].axvline(0, linestyle='--', color='black')
            axs[i].set_title(signal)
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("% Signal Change")

        fig.suptitle(f"Group: {arousal_type} | {reg_type}")
        fig.tight_layout()
        fig.savefig(f"group_{arousal_type}_{reg_type}_modelboot.png")
        plt.close(fig)

# Plot by signal across arousal types
for reg_type in signal_sets:
    for signal in signal_sets[reg_type]:
        fig, ax = plt.subplots(figsize=(8, 4))
        print(f"\n==> Plotting signal {signal} | {reg_type}")
        plotted_any = False

        for arousal_type in group_data:
            trial_dict = group_data[arousal_type].get(reg_type, {}).get(signal, {})
            if not trial_dict:
                print(f"  Skipping {arousal_type} | no data for {signal}")
                continue

            print(f"  Bootstrapping {arousal_type} | {signal}")
            mean, lower, upper = model_guided_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis = np.linspace(window[0], window[1], mean.shape[0])

            color = {'Sustained': 'blue', 'Transient': 'orange', 'Loss': 'red'}[arousal_type]
            ax.plot(time_axis, mean, label=f"{arousal_type}", color=color)
            ax.fill_between(time_axis, lower, upper, alpha=0.2, color=color)
            plotted_any = True

        if plotted_any:
            ax.axvline(0, color='black', linestyle='--')
            ax.set_title(f"{signal} | {reg_type}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("% Signal Change")
            ax.legend(title="Arousal Type")
            fig.tight_layout()
            fig.savefig(f"group_signal_{signal}_{reg_type}_modelboot.png")
            plt.close(fig)
        else:
            print(f"No data to plot for {signal} | {reg_type}")