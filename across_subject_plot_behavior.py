"""
across_subject_plot_behavior formerly pb.py
-----
This script processes button press data for multiple subjects, extracts peri-arousal fMRI signals, and generates subject-level plots and summary statistics. It differs from plot_behavior.py in that it focuses on aggregating and summarizing event counts across subject, not just plotting a single subject.
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
import pandas as pd 
import numpy as np


TR = 2.8
window = [-15, 20]
baseline_window = (-15, -10)

#define signal sets with and without regression
signal_sets = {
    "with_reg": ["GM_PSC", "Thalamus_PSC", "D2_with_reg_PSC", "D1_with_reg_PSC", "Norepinephrine_with_reg"],
    "without_reg": ["GM_PSC", "Thalamus_PSC", "D2_without_reg_PSC", "D1_without_reg_PSC", "Norepinephrine_without_reg"]
}

# colors for plotting
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





def extract_peri_with_baseline(signal, arousal_times, window, TR, baseline_window=(-15, -10)):
"""
    Extract peri-event time series around arousal times and apply baseline correction.

    Args:
        signal (np.ndarray): fMRI signal for a region/ROI
        arousal_times (list): event onset times
        window (tuple): time window (start, end) in seconds
        TR (float): repetition time (sec)
        baseline_window (tuple): window used to compute baseline

    Returns:
        np.ndarray: (trials × timepoints) peri-event matrix, baseline-corrected
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
    Process a single subject:
    - Load button press times
    - Identify sustained, transient, and loss events
    - Generate peri-arousal plots by signal and arousal type
    - Save trial-level data to .npz
    """

    subj_dir = Path(base_dir) / subject_id
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
    csv_df = pd.read_csv(subj_dir / f"{subject_id}_bold_timeseries.csv")
    for reg_type, reg_signals in signal_sets.items():
        signals = {name: csv_df[name].values for name in reg_signals}

        # PLOT 1: All peri-arousal trials by signal (panel)
        fig1, axs1 = plt.subplots(len(signals), 1, figsize=(10, len(signals) * 2.5))
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            peri_data = {}
            for arousal_type, times in arousals.items():
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] > 0:
                    peri_data[arousal_type] = peri

            time_axis = np.linspace(window[0], window[1], peri.shape[1]) if peri_data else []
            for arousal_type, peri in peri_data.items():
                for row in peri:
                    axs1[i].plot(time_axis, row, color=arousal_colors[arousal_type], alpha=0.2)
                axs1[i].plot([], [], color=arousal_colors[arousal_type], label=arousal_type)
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
            for arousal_type, times in arousals.items():
                peri = extract_peri_with_baseline(signal_data, times, window, TR)
                if peri.shape[0] > 0:
                    peri_data[arousal_type] = peri

            time_axis = np.linspace(window[0], window[1], peri.shape[1]) if peri_data else []
            for arousal_type, peri in peri_data.items():
                mean = np.mean(peri, axis=0)
                sem = np.std(peri, axis=0, ddof=1) / np.sqrt(peri.shape[0])
                ci95 = stats.t.ppf(0.975, peri.shape[0] - 1) * sem
                n_trials= peri.shape[0]
                label_with_n = f"{arousal_type}(n= {n_trials})"
                axs2[i].plot(time_axis, mean, color=arousal_colors[arousal_type], label=label_with_n)
                axs2[i].fill_between(time_axis, mean - ci95, mean + ci95, color=arousal_colors[arousal_type], alpha=0.2)
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
        fig3, axs3 = plt.subplots(len(arousals), 1, figsize=(10, len(arousals) * 2.5))
        for i, (arousal_type, times) in enumerate(arousals.items()):
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
            axs3[i].set_title("{}".format(arousal_type))
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
                   arousal_type: []
                   for arousal_type in arousals
                }
                for signal_name in signal_sets[reg_type]
             }
             for reg_type in signal_sets
        }
        # Fill peri_trials_dict from current subject
        for reg_type, reg_signals in signal_sets.items():
            signals = {name: csv_df[name].values for name in reg_signals}
            for signal_name, signal_data in signals.items():
                for arousal_type, times in arousals.items():
                   peri = extract_peri_with_baseline(signal_data, times, window, TR)
                   if peri.shape[0] > 0:
                      peri_trials_dict[reg_type][signal_name][arousal_type] = peri

        # Save to .npz file in subject directory
        save_path = Path(base_dir) / subject_id / f"{subject_id}_peri_trials.npz"
        np.savez_compressed(save_path, peri_trials=peri_trials_dict)
        print(f"Saved peri-arousal trials to {save_path}")


        plt.close(fig3)
        
def get_button_press_data(subject_id, button_root):
    
    """
    Load button press data for one subject and classify events into sustained, transient, and loss.

    Returns:
        press_times (np.ndarray): all button press timestamps
        sustained (list): sustained event times
        transient (list): transient event times
        loss (list): loss event times
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
    subjects = [
        "sub-racsleep04b","sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11",
        "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", "sub-racsleep20",
        "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30",
        "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
        "sub-racsleep38"
    ]

    base_dir = "/home/vob3"
    button_root = "/orcd/data/ldlewis/001/om/hf303/for_ore"

    table_rows = []

    for subject_id in subjects:
        press_times, sustained, transient, loss = get_button_press_data(subject_id, button_root)

        row = {
            "Subject": subject_id,
            "Sustained": len(sustained),
            "Transient": len(transient),
            "Loss": len(loss)
        }
        table_rows.append(row)

    df = pd.DataFrame(table_rows)


    df.to_csv("event_counts_per_subject.csv", index=False)
    print("\nPer-subject event counts saved to event_counts_per_subject.csv")
    print(df)

   
    summary = {}
    for event_type in ["Sustained", "Transient", "Loss"]:
        counts = df[event_type].values
        summary[event_type] = {
            "Total": int(np.sum(counts)),
            "Mean": np.mean(counts),
            "StdDev": np.std(counts, ddof=1)
        }

    print("\n=== Group Summary ===")
    for event_type, stats_dict in summary.items():
        print(f"{event_type}: {stats_dict['Total']} total | "
              f"{stats_dict['Mean']:.2f} ± {stats_dict['StdDev']:.2f} per subject")



'''
"sub-racsleep04b","sub-racsleep05", "sub-racsleep08", "sub-racsleep10", "sub-racsleep11", "sub-racsleep14", "sub-racsleep16", "sub-racsleep17", "sub-racsleep19", 
  "sub-racsleep20", "sub-racsleep22", "sub-racsleep27", "sub-racsleep28", "sub-racsleep29", "sub-racsleep30", "sub-racsleep31", "sub-racsleep32", "sub-racsleep34", "sub-racsleep35", "sub-racsleep36",
 "sub-racsleep38"
'''
	
