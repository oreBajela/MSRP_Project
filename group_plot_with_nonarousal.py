"""
group_plot_with_nonarousal.py
Extends the previous group-level plotting script to include a
“Non-Arousal” condition as an additional arousal type.

Changes from the original:
- Loads peri-event trials for Non-Arousal from each subject
- Adds Non-Arousal traces to all single-panel plots
- Expands the summary grid to 4 rows:
      Sustained | Transient | Loss | Non-Arousal
- Uses hierarchical bootstrapping for group means & 95% CIs
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pprint

plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})


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

signal_sets = {
    "with_reg": [
        "GM_PSC", "Thalamus_PSC",
        "D2_with_reg_PSC", "D1_with_reg_PSC", "Norepinephrine_with_reg"
    ],
    "without_reg": [
        "GM_PSC", "Thalamus_PSC",
        "D2_without_reg_PSC", "D1_without_reg_PSC", "Norepinephrine_without_reg"
    ]
}

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
group_data = {}

for subj in subjects:
    fpath = Path(base_dir) / subj / f"{subj}_peri_trials.npz"
    if not fpath.exists():
        print(f"Missing file: {fpath}")
        continue

    print(f"Loading {fpath}")
    npz = np.load(fpath, allow_pickle=True)
    if "peri_trials" not in npz.files:
        print(f" {fpath} has no key 'peri_trials' – skipping.")
        continue
     # convert the numpy object array back into dict
    data = npz["peri_trials"].item()

    for reg_type in signal_sets:
        for signal in signal_sets[reg_type]:
            for arousal_type in ["Sustained", "Transient", "Loss", "Non-Arousal"]:
                trials = data.get(reg_type, {}).get(signal, {}).get(arousal_type, None)
                if trials is None or len(trials) == 0:
                    continue
                group_data \
                    .setdefault(arousal_type, {}) \
                    .setdefault(reg_type, {}) \
                    .setdefault(signal, {}) \
                    .setdefault(subj, []).append(trials)


# Hierarchical bootstrap

def hierarchical_bootstrap_timecourses(trial_dict, n_boot=2000, ci=95):
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


# PLOT
"""
time_axis = np.linspace(window[0], window[1],
                        hierarchical_bootstrap_timecourses(
                            next(iter(next(iter(next(iter(group_data.values())))).values())),
                            10  # just to get shape
                        )[0].shape[0])

"""
def find_first_trial_dict(gd):
    for a in gd.values():
        for r in a.values():
            for s in r.values():
                if isinstance(s, dict) and len(s) > 0:
                    return s
    return None

trial_dict_example = find_first_trial_dict(group_data)
if trial_dict_example is None:
    raise RuntimeError("No non-empty trial data found!")
time_axis = np.linspace(window[0], window[1],
    hierarchical_bootstrap_timecourses(trial_dict_example, 10)[0].shape[0])

# --- Per-arousal per-regression-type ---
for arousal_type in group_data:
    for reg_type in group_data[arousal_type]:
        fig, axs = plt.subplots(1, len(group_data[arousal_type][reg_type]),
                                figsize=(15, 4), sharey=True)
        axs = np.atleast_1d(axs)

        for i, signal in enumerate(group_data[arousal_type][reg_type]):
            trial_dict = group_data[arousal_type][reg_type][signal]
            if not trial_dict:
                continue
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            t_axis = np.linspace(window[0], window[1], mean.shape[0])
            axs[i].plot(t_axis, mean, color='blue')
            axs[i].fill_between(t_axis, lower, upper, color='blue', alpha=0.3)
            axs[i].axvline(0, linestyle='--', color='black')
            axs[i].set_title(signal)
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("% Signal Change")

        fig.suptitle(f"Group: {arousal_type} | {reg_type}")
        fig.tight_layout()
        fig.savefig(f"group_{arousal_type}_{reg_type}.png")
        plt.close(fig)

# --- Single-panel per signal comparing all arousal types ---
color_map = {
    "Sustained": "blue",
    "Transient": "orange",
    "Loss": "red",
    "Non-Arousal": "darkgreen"
}

for reg_type in signal_sets:
    for signal in signal_sets[reg_type]:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted_any = False

        for arousal_type in ["Sustained", "Transient", "Loss", "Non-Arousal"]:
            trial_dict = group_data.get(arousal_type, {}).get(reg_type, {}).get(signal, {})
            if not trial_dict:
                continue
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            t_axis = np.linspace(window[0], window[1], mean.shape[0])
            ax.plot(t_axis, mean, label=arousal_type, color=color_map[arousal_type])
            ax.fill_between(t_axis, lower, upper, color=color_map[arousal_type], alpha=0.2)
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

# --- 4×2 summary grid ---
fig, axs = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=False)
signal_colors_left = {
    "D1_with_reg_PSC": "purple",
    "D2_with_reg_PSC": "orange",
    "Norepinephrine_with_reg": "cyan"
}
signal_colors_right = {
    "GM_PSC": "blue",
    "Thalamus_PSC": "green"
}

arousal_order = ["Sustained", "Transient", "Loss", "Non-Arousal"]

for i, arousal_type in enumerate(arousal_order):
    # LEFT column: neuroreceptors
    for signal in ["D1_with_reg_PSC", "D2_with_reg_PSC", "Norepinephrine_with_reg"]:
        trial_dict = group_data.get(arousal_type, {}).get("with_reg", {}).get(signal, {})
        if trial_dict:
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            t_axis = np.linspace(window[0], window[1], mean.shape[0])
            axs[i, 0].plot(t_axis, mean, label=signal.split('_')[0],
                           color=signal_colors_left[signal])
            axs[i, 0].fill_between(t_axis, lower, upper,
                                   color=signal_colors_left[signal], alpha=0.2)
    axs[i, 0].set_title(f"{arousal_type} - D1/D2/NE")
    axs[i, 0].axvline(0, linestyle='--', color='black')
    axs[i, 0].legend()

    # RIGHT column: GM + Thalamus
    for signal in ["GM_PSC", "Thalamus_PSC"]:
        trial_dict = group_data.get(arousal_type, {}).get("with_reg", {}).get(signal, {})
        if trial_dict:
            mean, lower, upper = hierarchical_bootstrap_timecourses(trial_dict, n_boot=n_boot, ci=ci)
            t_axis = np.linspace(window[0], window[1], mean.shape[0])
            axs[i, 1].plot(t_axis, mean, label=signal.split('_')[0],
                           color=signal_colors_right[signal])
            axs[i, 1].fill_between(t_axis, lower, upper,
                                   color=signal_colors_right[signal], alpha=0.2)
    axs[i, 1].set_title(f"{arousal_type} - GM & Thalamus")
    axs[i, 1].axvline(0, linestyle='--', color='black')
    axs[i, 1].legend()

# Axis labels
for ax in axs[:, 0]:
    ax.set_ylabel("% Signal Change")
for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")

fig.suptitle("Group-Averaged Signal Traces by Arousal Type (Including Non-Arousal)",
             fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("group_summary_by_arousal_type_grid_with_nonarousal.png")
plt.close(fig)

print("All plots saved including Non-Arousal row ✅")

