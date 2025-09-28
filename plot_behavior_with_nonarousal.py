'''
plot_behavior_with_nonarousal.py

This script extends my plot_behavior.py code by adding control analyses using
button presses that occur during non-arousal periods.

For each subject:
- Loads cleaned button-press events from .mat files
- Identifies sustained, transient, and loss arousal events
- Identifies non-arousal button-press events that meet control criteria:
      • No long pauses (≥20 s) in the 20 s before or after the press
      • At least ~6–8 presses occur in both the 20 s before and after
      • Not within ±20 s of an arousal or another non-arousal event
- Plots behavioral timelines showing all presses plus labeled arousal
  and non-arousal events
- Extracts peri-event BOLD time-series (baseline-corrected) for:
      • Sustained, transient, loss, and non-arousal events
      • GM, Thalamus, D1, D2, and NE signals (with/without regression)
- Generates figures:
      (1) All peri-event trials by signal
      (2) Mean ±95 % CI for each signal
      (3) Mean ±95 % CI grouped by event type
- Saves peri-event trial arrays (including non-arousal) to compressed .npz

Inputs:
- Subject ID(s)
- Base output directory
- Button-press root directory containing `*_buttonpresses_clean.mat`
- Precomputed subject-level BOLD time-series (CSV)

Outputs:
- PNG plots of behavior and peri-event BOLD responses
- Peri-event trial data saved as .npz files for further group analyses

'''




import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


TR = 2.22
window = [-15, 20]
baseline_window = (-15, -10)

signal_sets = {
    "with_reg": ["GM_PSC", "Thalamus_PSC", "D2_with_reg_PSC", "D1_with_reg_PSC", "Norepinephrine_with_reg"],
    "without_reg": ["GM_PSC", "Thalamus_PSC", "D2_without_reg_PSC", "D1_without_reg_PSC", "Norepinephrine_without_reg"]
}

arousal_colors = {"Sustained": "blue", "Transient": "orange", "Loss": "red", "Non-Arousal": "green"}
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


pretty_names = {
    "GM_PSC": "GM",
    "Thalamus_PSC": "Thalamus",
    "D2_with_reg_PSC": "D2",
    "D1_with_reg_PSC": "D1",
    "Norepinephrine_with_reg": "NE",
    "D2_without_reg_PSC": "D2",
    "D1_without_reg_PSC": "D1",
    "Norepinephrine_without_reg": "NE"
}



def find_nonarousal_events(press_times, arousals, pre_t=20, post_t=20, min_rate=6/20, t_exclude=120):
    """
    Identify button-press events during non-arousal periods.

    Criteria:
    - Must have at least `min_rate * pre_t` presses in the 20 s before and after.
    - Must not be within ±pre_t seconds of an arousal or another non-arousal event.
    - Excludes first `t_exclude` seconds of the run.

    Returns non_arousals : list of timestamps (s)
    """

    non_arousals = []
    press_times = np.asarray(press_times)

    # combine all arousal times into one array
    arousal_times = np.concatenate(list(arousals.values())) if arousals else np.array([])

    # start searching after equilibration
    t_next_ok = t_exclude
    npresses_needed = int(min_rate * pre_t)

    for bp in press_times:
        if bp < t_next_ok:
            continue

        # skip if within ±pre_t of an arousal
        if arousal_times.size and np.any(np.abs(arousal_times - bp) < pre_t):
            continue

        # count button presses in pre_t and post_t windows
        before = np.sum((press_times >= bp - pre_t) & (press_times < bp))
        after  = np.sum((press_times > bp) & (press_times <= bp + post_t))

        if before >= npresses_needed and after >= npresses_needed:
            non_arousals.append(bp)
            # push next allowed event 
            t_next_ok = bp + post_t

    return non_arousals


def extract_peri_with_baseline(signal, arousal_times, window, TR, baseline_window=(-15, -10)):
    """
    Extract peri-event segments of a signal with baseline correction.

    Args:
        signal (np.ndarray): 1D BOLD signal array.
        arousal_times (list or np.ndarray): Times of arousal events (in seconds).
        window (tuple): (start, end) peri-event window in seconds.
        TR (float): Repetition time in seconds.
        baseline_window (tuple): (start, end) window used for baseline correction.

    Returns:
        np.ndarray: 2D array (n_events x n_timepoints) of baseline-corrected peri-event     segments.
    """

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


    
def plot_subject(subject_id, base_dir, button_root):
    """
    Generate behavior timeline and peri-event plots for a subject.

    Args:
        subject_id (str): Subject identifier (e.g., "sub-racsleep02").
        base_dir (str or Path): Base directory to save outputs.
        button_root (str or Path): Directory containing button press .mat files.

    Saves:
        - Behavior eventplot (timeline of button presses)
        - Panel plots of peri-event BOLD signals
        - Compressed .npz file with peri-event trial data
    """
    
    subj_dir = Path(base_dir)/subject_id
    os.makedirs(subj_dir, exist_ok=True)
    short_id = subject_id.replace("sub-", "")

    mat_path = Path(button_root) / subject_id / "behav" / "{}_run01_buttonpresses_clean.mat".format(short_id)
    mat = sio.loadmat(mat_path)
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

    # find non-arousal events 
    non_arousals = find_nonarousal_events(
       press_times,
       arousals,
       pre_t=20,
       post_t=20,
       min_rate=6/20
      )
    print(f"Found {len(non_arousals)} non-arousal events for {subject_id}")


    plt.figure(figsize=(14, 3))
    plt.eventplot([press_times], colors='black', lineoffsets=1.0, linelengths=0.9, label="All Presses")
    if sustained:
        plt.eventplot([sustained], colors='blue', lineoffsets=1.2, linelengths=0.9, label='Sustained')
    if transient:
        plt.eventplot([transient], colors='orange', lineoffsets=1.4, linelengths=0.9, label='Transient')
    if loss:
        plt.eventplot([loss], colors='red', lineoffsets=1.6, linelengths=0.9, label='Loss')
    if non_arousals:
        plt.eventplot([non_arousals], colors='green',lineoffsets=1.8, linelengths=0.9, label='Non-Arousal')

    plt.title("Button Press Timeline")
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(subj_dir / "{}_button_timeline.png".format(subject_id))
    plt.close()

    csv = np.loadtxt(subj_dir / "{}_bold_timeseries.csv".format(subject_id), delimiter=",", skiprows=1)
    csv_df = pd.read_csv(subj_dir / f"{subject_id}_bold_timeseries.csv")
    for reg_type, reg_signals in signal_sets.items():
        all_events = dict(arousals)
        all_events["Non-Arousal"] = non_arousals
        signals = {name: csv_df[name].values for name in reg_signals}

        # PLOT 1: All peri-arousal trials by signal (panel)
        fig1, axs1 = plt.subplots(len(signals), 1, figsize=(10, len(signals) * 2.5))
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            peri_data = {}
            for event_type, times in all_events.items(): 
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] > 0:
                    peri_data[event_type] = peri

            time_axis = np.linspace(window[0], window[1], peri.shape[1]) if peri_data else []
            for event_type, peri in peri_data.items():
                for row in peri:
                    axs1[i].plot(time_axis, row, color=arousal_colors[event_type], alpha=0.2)
                axs1[i].plot([], [], color=arousal_colors[event_type], label=event_type)
            axs1[i].axvline(0, color='black', linestyle='--')
            axs1[i].set_title("{}".format(signal_name))
            axs1[i].legend()
            axs1[i].set_ylabel("% Signal Change")
        axs1[-1].set_xlabel("Time (s)")
        fig1.suptitle("{} | {} | All peri-arousal trials".format(subject_id, reg_type))
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(subj_dir / "{}_{}_panel_peri_trials.png".format(subject_id, reg_type))
        plt.close(fig1)

        # PLOT 2: Mean ± 95% CI by signal (panel)
        fig2, axs2 = plt.subplots(len(signals), 1, figsize=(10, len(signals) * 2.5))
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            peri_data = {}
            for event_type, times in all_events.items():
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] > 0:
                    peri_data[event_type] = peri

            time_axis = np.linspace(window[0], window[1], peri.shape[1]) if peri_data else []
            for event_type, peri in peri_data.items():
                mean = np.mean(peri, axis=0)
                sem = np.std(peri, axis=0, ddof=1) / np.sqrt(peri.shape[0])
                ci95 = stats.t.ppf(0.975, peri.shape[0] - 1) * sem
                n_trials= peri.shape[0]
                label_with_n = f"{event_type}(n= {n_trials})"
                axs2[i].plot(time_axis, mean, color=arousal_colors[event_type], label=label_with_n)
                axs2[i].fill_between(time_axis, mean - ci95, mean + ci95, color=arousal_colors[event_type], alpha=0.2)
            axs2[i].axvline(0, color='black', linestyle='--')
            axs2[i].set_title("{}".format(signal_name))
            axs2[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs2[i].set_ylabel("% Signal Change")
        axs2[-1].set_xlabel("Time (s)")
        fig2.suptitle("{} | {} | Mean ± 95% CI by Signal".format(subject_id, reg_type))
        fig2.tight_layout(rect=[0, 0.03, 0.85, 0.95])
        fig2.savefig(subj_dir / "{}_{}_panel_mean_ci.png".format(subject_id, reg_type))
        plt.close(fig2)

        # PLOT 3: Mean ± 95% CI by arousal type (panel)
        fig3, axs3 = plt.subplots(len(all_events), 1, figsize=(10, len(arousals) * 2.5))
        for i, (event_type, times) in enumerate(all_events.items()):
            for signal_name, signal_data in signals.items():
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] > 0:
                    time_axis = np.linspace(window[0], window[1], peri.shape[1])
                    mean = np.mean(peri, axis=0)
                    sem = np.std(peri, axis=0, ddof=1) / np.sqrt(peri.shape[0])
                    ci95 = stats.t.ppf(0.975, peri.shape[0] - 1) * sem
                    axs3[i].plot(time_axis, mean, color=signal_colors[signal_name], label=pretty_names.get(signal_name, signal_name))
                    axs3[i].fill_between(time_axis, mean - ci95, mean + ci95, color=signal_colors[signal_name], alpha=0.2)
            
            axs3[i].axvline(0, color='black', linestyle='--')
            axs3[i].set_title("{}".format(event_type))
            handles, labels = axs3[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs3[i].legend(by_label.values(), by_label.keys(), loc="upper right")
            # axs3[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs3[i].set_ylabel("% Signal Change")
        axs3[-1].set_xlabel("Time (s)")
        fig3.suptitle("{} | {} | Mean ± 95% CI by Arousal".format(subject_id, reg_type))
        fig3.tight_layout(rect=[0, 0.03, 0.85, 0.95])
        fig3.savefig(subj_dir / "{}_{}_panel_by_arousal.png".format(subject_id, reg_type))
         

        peri_trials_dict = {
             reg_type: {
                signal_name: {
                   event_type: []
                   for event_type in all_events
                }
                for signal_name in signal_sets[reg_type]
             }
             for reg_type in signal_sets
        }
        # Fill peri_trials_dict from current subject
        for reg_type, reg_signals in signal_sets.items():
            signals = {name: csv_df[name].values for name in reg_signals}
            for signal_name, signal_data in signals.items():
                for event_type, times in all_events.items():
                   peri = extract_peri_with_baseline(signal_data, times, window, TR)
                   if peri.shape[0] > 0:
                      peri_trials_dict[reg_type][signal_name][event_type] = peri

        # Save to .npz file in subject directory
        save_path = Path(base_dir) / subject_id / f"{subject_id}_peri_trials.npz"
        np.savez_compressed(save_path, peri_trials=peri_trials_dict)
        print(f"Saved peri-arousal trials to {save_path}")


        plt.close(fig3)
        
def get_button_press_data(subject_id, button_root):  
    """
    Load button press times and classify arousal events for a subject.

    Args:
        subject_id (str): Subject identifier (e.g., "sub-racsleep02").
        button_root (str or Path): Directory containing button press .mat files.

    Returns:
        tuple: (press_times, sustained, transient, loss)
            - press_times (np.ndarray): All button press times
            - sustained (list): Times of sustained arousal events
            - transient (list): Times of transient arousal events
            - loss (list): Times of loss events
      """


    short_id = subject_id.replace("sub-", "")
    mat_path = Path(button_root) / subject_id / "behav" / f"{short_id}_run01_buttonpresses_clean.mat"
    mat = sio.loadmat(mat_path)
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

    return press_times, sustained, transient, loss

if __name__ == "__main__":
    subjects = ["sub-racsleep10", "sub-racsleep11"]
    base_dir = "/home/vob3"
    button_root = "/orcd/data/ldlewis/001/om/hf303/for_ore"
    for subject_id in subjects:
        plot_subject(subject_id, base_dir, button_root)
    print("All plots done.")

'''
"sub-racsleep04b","sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11", "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", 
  "sub-racsleep20", "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30", "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
 "sub-racsleep38"
'''
	
