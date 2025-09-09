"""
This script defines a plotting function for the dopamine button-pressing project.
It generates combined plots showing:
  1. Gray matter (GM) and Thalamus signals
  2. Receptor-weighted signals (D1, D2, NE)
  3. Button press behavior (all presses + Sustained/Transient/Loss arousal events)

The function is designed to be called inside other scripts.
"""


def plot_full_combined(subject_id, time_sec,
                       gm_signal_psc, thalamus_signal_psc,
                       d1_psc_with_reg, d2_psc_with_reg, norepinephrine_psc_with_reg,
                       press_times, sustained, transient, loss,
                       output_dir, zoom_range=None):
     """
    Create a combined visualization of neural signals and button press behavior.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., "sub-racsleep04b").
    time_sec : np.ndarray
        Time vector (in seconds) aligned to BOLD signals.
    gm_signal_psc : np.ndarray
        Percent signal change (PSC) for gray matter.
    thalamus_signal_psc : np.ndarray
        PSC for thalamus.
    d1_psc_with_reg : np.ndarray
        PSC for D1-weighted signal (with regression).
    d2_psc_with_reg : np.ndarray
        PSC for D2-weighted signal (with regression).
    norepinephrine_psc_with_reg : np.ndarray
        PSC for norepinephrine-weighted signal (with regression).
    press_times : list or np.ndarray
        Timestamps (in seconds) of all button presses.
    sustained : list
        Event times classified as sustained arousal.
    transient : list
        Event times classified as transient arousal.
    loss : list
        Event times classified as loss events.
    output_dir : str
        Directory to save the plot.
    zoom_range : tuple, optional
        (start_sec, end_sec) to zoom into a time window. If None, the range is
        automatically set based on event timing.

    Outputs
    -------
    Saves a PNG plot in `output_dir` with filename:
        "{subject_id}_combined_plot.png"
    """





    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Determine zoom range automatically if not given
    if zoom_range is None:
        all_events = []
        for lst in [sustained, transient, loss]:
            if lst is not None and len(lst) > 0:
                all_events.extend(lst)
        if all_events:
            min_event = min(all_events)
            max_event = max(all_events)
            margin = 50  # seconds on each side
            zoom_start = max(min_event - margin, time_sec[0])
            zoom_end = min(max_event + margin, time_sec[-1])
            zoom_range = (zoom_start, zoom_end)
        else:
            zoom_range = (time_sec[0], time_sec[-1])  # fallback to full plot

    # Mask data for zoom range
    zoom_mask = (time_sec >= zoom_range[0]) & (time_sec <= zoom_range[1])

    fig, axes = plt.subplots(3, 1, figsize=(1, 10), sharex=True)

    # --- Top row: GM & Thalamus ---
    axes[0].plot(time_sec[zoom_mask], gm_signal_psc[zoom_mask], label="GM", color="blue")
    axes[0].plot(time_sec[zoom_mask], thalamus_signal_psc[zoom_mask], label="Thalamus", color="green")
    axes[0].set_ylabel("% Signal Change")
    axes[0].set_title("GM and Thalamus Signals", fontsize=16, fontweight="bold")
    axes[0].legend()

    # --- Middle row: Receptor-Weighted Signals ---
    axes[1].plot(time_sec[zoom_mask], d1_psc_with_reg[zoom_mask], label="D1", color="purple")
    axes[1].plot(time_sec[zoom_mask], d2_psc_with_reg[zoom_mask], label="D2", color="orange")
    axes[1].plot(time_sec[zoom_mask], norepinephrine_psc_with_reg[zoom_mask], label="NE", color="cyan")
    axes[1].set_ylabel("% Signal Change")
    axes[1].set_title("Receptor-Weighted Signals", fontsize=16, fontweight="bold")
    axes[1].legend()

  
    y_center = 0.5  
    all_press_size = 25   # smaller so colored circles stand out
    arousal_size = 60    

# Apply zoom mask to press_times so only visible presses are plotted
    press_times_zoom = [pt for pt in press_times if zoom_range[0] <= pt <= zoom_range[1]]
    sustained_zoom = [pt for pt in sustained if zoom_range[0] <= pt <= zoom_range[1]] if sustained else []
    transient_zoom = [pt for pt in transient if zoom_range[0] <= pt <= zoom_range[1]] if transient else []
    loss_zoom = [pt for pt in loss if zoom_range[0] <= pt <= zoom_range[1]] if loss else []

# Plot ALL button presses in gray
    if press_times_zoom:
        axes[2].scatter(press_times_zoom, [y_center] * len(press_times_zoom),
                    color='black', s=all_press_size, alpha=0.8, zorder=1)

# Overlay arousal events in color
    if sustained_zoom:
        axes[2].scatter(sustained_zoom, [y_center] * len(sustained_zoom),
                    color='blue', s=arousal_size, zorder=2)
    if transient_zoom:
        axes[2].scatter(transient_zoom, [y_center] * len(transient_zoom),
                    color='orange', s=arousal_size, zorder=2)
    if loss_zoom:
        axes[2].scatter(loss_zoom, [y_center] * len(loss_zoom),
                    color='red', s=arousal_size, zorder=2)

# Remove y-axis
    axes[2].set_yticks([])
    axes[2].set_ylabel("")
    axes[2].set_title("Button Presses", fontsize=16, fontweight="bold")

# Remove legend
    axes[2].legend().remove()


    # X-axis
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(zoom_range)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{subject_id}_combined_plot.png"))
    plt.close()

