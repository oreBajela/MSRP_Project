""" 
full_pipeline_plot.py

This script combines the full dopamine project pipeline for a given subject.  
It combines:
  1. Extraction of receptor-informed and anatomical BOLD time series 
     using `process_subject.run_pipeline`.
  2. Loading of button press behavior data and classification of events 
     (sustained, transient, loss) using `plot_behavior.get_button_press_data`.
  3. Visualization of BOLD signals alongside behavioral events using 
     `plot_combined_signals_and_behavior.plot_full_combined`.

The output includes:
  - A CSV file with BOLD timeseries for GM, Thalamus, and receptor-weighted signals.
  - A combined plot of neural and behavioral data for each subject.

Usage:
    python run_full_pipeline_and_plot.py

Requirements:
    - fMRI preprocessed files and segmentation outputs in BIDS-like structure.
    - Behavioral .mat files containing button press events.
    - The following project scripts in the same environment:
        process_subject.py
        plot_behavior.py
        plot_combined_signals_and_behavior.py
"""



import os
import numpy as np
import pandas as pd
from pathlib import Path

# Import from your two scripts
from process_subject import run_pipeline
from plot_behavior import get_button_press_data  
from plot_combined_signals_and_behavior import plot_full_combined  

TR = 2.22

def main():  
    """
    Run the dopamine project pipeline for one or more subjects:
      1. Locate fMRI preprocessed and segmentation files.
      2. Run subject-level processing to extract BOLD signals (saves CSV).
      3. Load extracted signals and button press data.
      4. Generate combined plots of neural and behavioral signals.

    Args:
        None (subjects and directories are specified inside the function).

    Returns:
        None (saves CSVs and PNG plots to disk).
    """

    # subjects to process
    subjects = ["sub-racsleep19"]
    
    base_path = Path("/orcd/data/ldlewis/001/om/hf303/for_ore")
    data_dir = Path("neuromaps_annotations")
    data_dir.mkdir(exist_ok=True)

    button_root = base_path  # where your behavior .mat files live

    for subj in subjects:
        func_dir = base_path / subj / "func"

        #  Locate fMRI preprocessed file 
        fmri_file_001 = func_dir / f"{subj}_task-rest_run-001_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        fmri_file_1   = func_dir / f"{subj}_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

        if fmri_file_001.exists():
            fmri_path = fmri_file_001
        elif fmri_file_1.exists():
            fmri_path = fmri_file_1
        else:
            print(f"No fMRI file found for {subj}")
            continue

        # Locate segmentation file 
        seg_file_001 = func_dir / f"{subj}_task-rest_run-001_space-MNI152NLin2009cAsym_res-2_desc-aseg_dseg.nii.gz"
        seg_file_1   = func_dir / f"{subj}_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-aseg_dseg.nii.gz"

        if seg_file_001.exists():
            seg_path = seg_file_001
        elif seg_file_1.exists():
            seg_path = seg_file_1
        else:
            print(f"No segmentation file found for {subj}")
            continue

        # 1. Run process_subject to get the BOLD CSV 
        run_pipeline(subj, fmri_path, seg_path, data_dir)

        #  2. Load the BOLD CSV data 
        output_dir = Path("/home/vob3") / subj
        csv_path = output_dir / f"{subj}_bold_timeseries.csv"
        csv_df = pd.read_csv(csv_path)

        time_sec = csv_df["Time_s"].values
        gm_signal_psc = csv_df["GM_PSC"].values
        thalamus_signal_psc = csv_df["Thalamus_PSC"].values
        d2_psc_with_reg = csv_df["D2_with_reg_PSC"].values
        d1_psc_with_reg = csv_df["D1_with_reg_PSC"].values
        norepinephrine_psc_with_reg = csv_df["Norepinephrine_with_reg"].values

        # 3. Get button press + arousal classification 
        press_times, sustained, transient, loss = get_button_press_data(subj, button_root)

        #  4. Call the combined plotting function 
        plot_full_combined(
            subject_id=subj,
            time_sec=time_sec,
            gm_signal_psc=gm_signal_psc,
            thalamus_signal_psc=thalamus_signal_psc,
            d1_psc_with_reg=d1_psc_with_reg,
            d2_psc_with_reg=d2_psc_with_reg,
            norepinephrine_psc_with_reg=norepinephrine_psc_with_reg,
            press_times=press_times,
            sustained=sustained,
            transient=transient,
            loss=loss,
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()

