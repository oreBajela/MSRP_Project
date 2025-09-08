"""
summary_group_bootstrap_plots.py formerly gp3.py

This script performs group-level analysis of peri-arousal fMRI signals 
using hierarchical bootstrapping across subjects and trials. Then creates 
a final 3×2 grid summary plot (D1/D2/NE vs GM/Thalamus)

Pipeline overview:
1. Load subject-level peri-event trial data from .npz files
2. Aggregate peri-trials across subjects for each:
      - Arousal type: Sustained, Transient, Loss
      - Regression type: with_reg, without_reg
      - Signal: GM, Thalamus, D1, D2, NE
3. Apply hierarchical bootstrapping (resample subjects, then trials) 
   to compute mean ± 95% CI group time courses.
4. Generate multiple sets of figures:
      (a) Group plots of all signals per arousal × regression type
      (b) Signal-level comparisons across arousal types
      (c) Final 3×2 grid: D1/D2/NE on left, GM/Thalamus on right,
          one row per arousal type

Inputs:
    - {subj}_peri_trials.npz files in subject folders under base_dir
      Each .npz contains dictionary: peri_trials[reg_type][signal][arousal_type] = list of trial arrays

Outputs:
    - group_<ArousalType>_<RegType>.png        (panel of all signals per arousal × regression type)
    - group_signal_<Signal>_<RegType>.png      (arousal comparisons for each signal)
    - group_summary_by_arousal_type_grid.png   (final grid summary figure)
"""





import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import os
from collections import defaultdict

plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
base_dir = "/home/vob3"
subjects = ["sub-racsleep04b", "sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11", "sub-racsleep14",
            "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", "sub-racsleep20", "sub-racsleep22", "sub-racsleep27",
            "sub-racsleep28", "sub-racsleep29", "sub-racsleep30", "sub-racsleep31", "sub-racsleep32",
            "sub-racsleep34", "sub-racsleep35", "sub-racsleep36", "sub-racsleep38"]

def hierarchical_bootstrap_timecourses(trial_dict, n_boot=2000, ci=95):

     """
    Perform hierarchical bootstrap across subjects and trials.

    trial_dict: {subject_id: [list of trial arrays]}
    Returns:
        mean, lower, upper (bootstrapped group mean and CI)
    """

    subjects = list(trial_dict.keys())
    n_subjects = len(subjects)
    boot_samples = []

    for _ in range(n_boot):
        sampled_subjects = np.random.choice(subjects, size=n_subjects, replace=True)
        all_resampled = []

        for subj in sampled_subjects:
            subj_trials = np.vstack(trial_dict[subj])
            n_trials = subj_trials.shape[0]
            resampled = subj_trials[np.random.choice(n_trials, n_trials, replace=True)]
            all_resampled.append(resampled)

        all_resampled_concat = np.vstack(all_resampled)
        boot_samples.append(np.mean(all_resampled_concat, axis=0))

    boot_samples = np.array(boot_samples)
    mean = np.mean(boot_samples, axis=0)
    lower = np.percentile(boot_samples, (100 - ci) / 2, axis=0)
    upper = np.percentile(boot_samples, 100 - (100 - ci) / 2, axis=0)
    return mean, lower, upper





# Parametes
window = [-15, 20]
TR = 2.22
n_boot = 2000
ci = 95

signal_sets = {
    "with_reg": ["GM_PSC", "Thalamus_PSC", "D2_with_reg_PSC", "D1_with_reg_PSC", "Norepinephrine_with_reg"],
    "without_reg": ["GM_PSC", "Thalamus_PSC", "D2_without_reg_PSC", "D1_without_reg_PSC", "Norepinephrine_without_reg"]
}


# Load data 
group_data = {}

for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing file: {fpath}")
        continue

    print(f"Loading data from {fpath}")
    data = np.load(fpath, allow_pickle=True)['peri_trials'].item()

    for reg_type in signal_sets:
        for signal in signal_sets[reg_type]:
            for arousal_type in ['Sustained', 'Transient', 'Loss']:
                trials = data.get(reg_type, {}).get(signal, {}).get(arousal_type, None)
                if trials is None or len(trials) == 0:
                    continue

                group_data.setdefault(arousal_type, {}).setdefault(reg_type, {}).setdefault(signal, {}).setdefault(subj, [])
                group_data[arousal_type][reg_type][signal][subj].append(trials)


# PLOT 1: Group plots of all signals per arousal × reg_type
for arousal_type in group_data:
    for reg_type in group_data[arousal_type]:
        fig, axs = plt.subplots(1, len(group_data[arousal_type][reg_type]), figsize=(15, 4), sharey=True)
        axs = np.atleast_1d(axs)

        for i, signal in enumerate(group_data[arousal_type][reg_type]):
            trial_dict = group_data[arousal_type][reg_type][signal]
            if not trial_dict:
                continue

            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis = np.linspace(window[0], window[1], mean.shape[0])

            axs[i].plot(time_axis, mean, color='blue')
            axs[i].fill_between(time_axis, lower, upper, color='blue', alpha=0.3)
            axs[i].axvline(0, linestyle='--', color='black')
            axs[i].set_title(signal)
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("% Signal Change")

        fig.suptitle(f"Group: {arousal_type} | {reg_type}")
        fig.tight_layout()
        fig.savefig(f"group_{arousal_type}_{reg_type}.png")
        plt.close(fig)

# PLOT 2: Signal-level comparisons across arousal types
for reg_type in signal_sets:
    for signal in signal_sets[reg_type]:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted_any = False

        for arousal_type, color in zip(['Sustained', 'Transient', 'Loss'], ['blue', 'orange', 'red']):
            trial_dict = group_data[arousal_type].get(reg_type, {}).get(signal, {})
            if not trial_dict:
                continue

            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis = np.linspace(window[0], window[1], mean.shape[0])

            ax.plot(time_axis, mean, label=arousal_type, color=color)
            ax.fill_between(time_axis, lower, upper, color=color, alpha=0.2)
            plotted_any = True

        if plotted_any:
            ax.axvline(0, color='black', linestyle='--')
            ax.set_title(f"{signal} | {reg_type}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("% Signal Change")
            ax.legend(title="Arousal Type")
            fig.tight_layout()
            fig.savefig(f"group_signal_{signal}_{reg_type}.png")
            plt.close(fig)




# PLOT 3: Final 3×2 grid summary (D1/D2/NE vs GM/Thalamus)
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=False)

# Now neuroreceptors are on the left (axs[:, 0]), GM & Thalamus on the right (axs[:, 1])
signal_colors_left = {'D1_with_reg_PSC': 'purple', 'D2_with_reg_PSC': 'orange', 'Norepinephrine_with_reg': 'cyan'}
signal_colors_right = {'GM_PSC': 'blue', 'Thalamus_PSC': 'green'}
arousal_order = ['Sustained', 'Transient', 'Loss']

for i, arousal_type in enumerate(arousal_order):
    # === LEFT: D1, D2, NE
    for signal in ['D1_with_reg_PSC', 'D2_with_reg_PSC', 'Norepinephrine_with_reg']:
        trial_dict = group_data[arousal_type]['with_reg'].get(signal, {})
        if trial_dict:
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis = np.linspace(window[0], window[1], mean.shape[0])
            axs[i, 0].plot(time_axis, mean, label=signal.split('_')[0], color=signal_colors_left[signal])
            axs[i, 0].fill_between(time_axis, lower, upper, color=signal_colors_left[signal], alpha=0.2)

    axs[i, 0].set_title(f"{arousal_type} - D1/D2/NE")
    axs[i, 0].axvline(0, linestyle='--', color='black')
    axs[i, 0].legend()

    # === RIGHT: GM + Thalamus
    for signal in ['GM_PSC', 'Thalamus_PSC']:
        trial_dict = group_data[arousal_type]['with_reg'].get(signal, {})
        if trial_dict:
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            time_axis = np.linspace(window[0], window[1], mean.shape[0])
            axs[i, 1].plot(time_axis, mean, label=signal.split('_')[0], color=signal_colors_right[signal])
            axs[i, 1].fill_between(time_axis, lower, upper, color=signal_colors_right[signal], alpha=0.2)

    axs[i, 1].set_title(f"{arousal_type} - GM & Thalamus")
    axs[i, 1].axvline(0, linestyle='--', color='black')
    axs[i, 1].legend()

# Label axes
for ax in axs[:, 0]:
    ax.set_ylabel("% Signal Change")
for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")

fig.suptitle("Group-Averaged Signal Traces by Arousal Type", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("group_summary_by_arousal_type_grid.png")
plt.close(fig)


