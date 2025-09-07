"""
peri_arousal_behavior_plots.py used to be behavioral_plots.py

This script extracts and visualizes peri-arousal fMRI responses 
(time-locked to button press events) for individual subjects.

The pipeline identifies three arousal event types from button press behavior:
    - Sustained
    - Transient
    - Loss

For each subject and each fMRI signal (GM, Thalamus, D1, D2, NE),
it generates peri-event responses with baseline correction, then
saves plots in multiple formats:

1. **Peri-arousal Trials (Raw)**
   - Subplots of all individual peri-event trials (one subplot per signal).

2. **Mean ± 95% CI (by signal)**
   - Subplots showing the averaged peri-event response per signal with 
     confidence intervals, separated by arousal type.

3. **Multi-Signal Comparison (by arousal type)**
   - Subplots of mean ± 95% CI across all signals, one subplot per arousal type.

Inputs:
    - Button press times (.mat files from behavioral logs)
    - Subject BOLD time series (.csv with signals GM, Thalamus, D1, D2, NE)

Outputs (saved under subject folder in base_dir):
    - Button press timeline figure
    - Peri-arousal plots:
        * Panel of raw peri-trials per signal
        * Panel of mean ± 95% CI per signal
        * Panel of multi-signal overlays per arousal type
"""








import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt

TR = 2.8
window = [-15, 20]
baseline_window = (-15, -10)   # used for baseline correction


# Mapping of signals, with/without regression versions
signal_sets = {
    "with_reg": ["GM_PSC", "Thalamus_PSC", "D2_with_reg_PSC", "D1_with_reg_PSC", "Norepinephrine_with_reg"],
    "without_reg": ["GM_PSC", "Thalamus_PSC", "D2_without_reg_PSC", "D1_without_reg_PSC", "Norepinephrine_without_reg"]
}

# Color maps for plotting
arousal_colors = {"Sustained": "blue", "Transient": "orange", "Loss": "red"}
signal_colors = {
    "GM_PSC": "black",
    "Thalamus_PSC": "purple",
    "D2_with_reg_PSC": "green",
    "D1_with_reg_PSC": "orange",
    "Norepinephrine_with_reg": "teal",
    "D2_without_reg_PSC": "green",
    "D1_without_reg_PSC": "orange",
    "Norepinephrine_without_reg": "teal"
}



# FUNCTION: extract_peri_with_baseline
# Extract peri-event signal segments with baseline correction.
# Returns an array of (trials × timepoints).

def extract_peri_with_baseline(signal, arousal_times, window, TR, baseline_window=(-15, -10)):
    peri = []
    n_before = round(abs(window[0]) / TR)
    n_after = round(window[1] / TR)
    time_axis = np.linspace(window[0], window[1], n_before + n_after)
    for t in arousal_times:
        idx = int(t / TR)
        start = idx - n_before
        end = idx + n_after
        if start >= 0 and end <= len(signal):
            chunk = signal[start:end]
            baseline_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
            baseline = np.mean(chunk[baseline_mask])
            peri.append(chunk - baseline)
    return np.vstack(peri) if peri else np.empty((0, n_before + n_after))





# FUNCTION: plot_subject
# For one subject:
# 1. Load button press times and identify sustained / transient / loss arousal events
# 2. Load subject fMRI signal time series
# 3. Generate three sets of plots:
#    (a) All peri-event trials per signal
#    (b) Mean ± 95% CI per signal
#    (c) Multi-signal overlays per arousal type

def plot_subject(subject_id, base_dir, button_root):
    subj_dir = Path(base_dir) / subject_id
    print("Processing {}...".format(subject_id))
    os.makedirs(subj_dir, exist_ok=True)

    short_id = subject_id.replace("sub-", "")
    button_path = Path(button_root) / subject_id / "behav" / "{}_run01_buttonpresses_clean.mat".format(short_id)
    mat = sio.loadmat(button_path)
    press_times = mat['buttonpresses_clean'].flatten()

    intervals = np.diff(press_times)
    arousal_indices = np.where(intervals >= 20)[0] + 1
    arousal_times = press_times[arousal_indices]

    sustained, transient, loss = [], [], []
    for i, t in zip(arousal_indices, arousal_times):
        following = press_times[(press_times > t) & (press_times <= t + 20)]
        if len(following) >= 5:
            sustained.append(t)
        elif len(following) <= 2:
            transient.append(t)
        loss.append(press_times[i - 1])

    arousals = {"Sustained": sustained, "Transient": transient, "Loss": loss}

    plt.figure(figsize=(14, 3))
    plt.eventplot([press_times], colors='black', lineoffsets=1.0, linelengths=0.9, label="All Presses")
    if sustained:
        plt.eventplot([sustained], colors='blue', lineoffsets=1.2, linelengths=0.9, label='Sustained')
    if transient:
        plt.eventplot([transient], colors='orange', lineoffsets=1.4, linelengths=0.9, label='Transient')
    if loss:
        plt.eventplot([loss], colors='red', lineoffsets=1.6, linelengths=0.9, label='Loss')
    plt.title("Button Press Timeline")
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(subj_dir / "{}_button_timeline.png".format(subject_id))
    plt.close()

    csv = np.loadtxt(subj_dir / "{}_bold_timeseries.csv".format(subject_id), delimiter=",", skiprows=1)

    for reg_type, signal_names in signal_sets.items():
        signals = {name: csv[:, i+1] for i, name in enumerate(signal_names)}

        # PLOT: All peri-arousal trials (subplot by signal)
        fig1, axs1 = plt.subplots(len(signals), 1, figsize=(10, 3 * len(signals)), sharex=True)
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            peri_data = {}
            for arousal_type, times in arousals.items():
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] > 0:
                    peri_data[arousal_type] = peri

            for arousal_type, peri in peri_data.items():
                time_axis = np.linspace(window[0], window[1], peri.shape[1])
                for row in peri:
                    axs1[i].plot(time_axis, row, color=arousal_colors[arousal_type], alpha=0.2)
            axs1[i].axvline(0, color='black', linestyle='--')
            axs1[i].set_title("{} | ALL peri-arousal trials".format(signal_name))
            axs1[i].set_ylabel('% Signal Change')
        axs1[-1].set_xlabel('Time (s)')
        fig1.tight_layout()
        fig1.savefig(subj_dir / "{}_{}_panel_peri_trials.png".format(subject_id, reg_type))
        plt.close(fig1)

        # PLOT: Mean ± 95% CI, by signal
        fig2, axs2 = plt.subplots(len(signals), 1, figsize=(10, 3 * len(signals)), sharex=True)
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            peri_data = {}
            for arousal_type, times in arousals.items():
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] > 0:
                    peri_data[arousal_type] = peri

            for arousal_type, peri in peri_data.items():
                time_axis = np.linspace(window[0], window[1], peri.shape[1])
                mean = np.mean(peri, axis=0)
                sem = np.std(peri, axis=0, ddof=1) / np.sqrt(peri.shape[0])
                ci95 = stats.t.ppf(0.975, peri.shape[0] - 1) * sem
                axs2[i].plot(time_axis, mean, color=arousal_colors[arousal_type], label=arousal_type)
                axs2[i].fill_between(time_axis, mean - ci95, mean + ci95, color=arousal_colors[arousal_type], alpha=0.2)
            axs2[i].axvline(0, color='black', linestyle='--')
            axs2[i].set_title("{} | Mean ± 95% CI".format(signal_name))
            axs2[i].set_ylabel('% Signal Change')
            axs2[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs2[-1].set_xlabel('Time (s)')
        fig2.tight_layout(rect=[0, 0, 0.85, 1])
        fig2.savefig(subj_dir / "{}_{}_panel_mean_ci.png".format(subject_id, reg_type))
        plt.close(fig2)

        # PLOT: Multi-signal per arousal type
        fig3, axs3 = plt.subplots(len(arousals), 1, figsize=(10, 3 * len(arousals)), sharex=True)
        for i, (arousal_type, times) in enumerate(arousals.items()):
            for signal_name, signal_data in signals.items():
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] == 0:
                    continue
                time_axis = np.linspace(window[0], window[1], peri.shape[1])
                mean = np.mean(peri, axis=0)
                sem = np.std(peri, axis=0, ddof=1) / np.sqrt(peri.shape[0])
                ci95 = stats.t.ppf(0.975, peri.shape[0] - 1) * sem
                axs3[i].plot(time_axis, mean, color=signal_colors[signal_name], label="{}".format(signal_name))
                axs3[i].fill_between(time_axis, mean - ci95, mean + ci95, color=signal_colors[signal_name], alpha=0.15)
            axs3[i].axvline(0, color='black', linestyle='--')
            axs3[i].set_title("{} | Multiple Signals (Mean ± 95% CI)".format(arousal_type))
            axs3[i].set_ylabel('% Signal Change')
            axs3[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs3[-1].set_xlabel('Time (s)')
        fig3.tight_layout(rect=[0, 0, 0.85, 1])
        fig3.savefig(subj_dir / "{}_{}_panel_multi_signal.png".format(subject_id, reg_type))
        plt.close(fig3)

if __name__ == "__main__":
    subjects = ["sub-racsleep02"]
    base_dir = "/home/vob3"
    button_root = "/orcd/data/ldlewis/001/om/hf303/for_ore"
    for subject_id in subjects:
        plot_subject(subject_id, base_dir, button_root)
    print("All plots done.")
	

