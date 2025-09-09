"""
summary_grid_prcoess_subject.py formerly p.y

This script extracts grey matter, thalamus, and receptor-weighted BOLD signals
from preprocessed fMRI runs in MNI152 space for subjects in the dopamine
button-pressing project. It applies anatomical masks (grey matter, thalamus,
striatum), resamples neuromaps receptor density annotations (D1, D2, NE), and
computes receptor-weighted time series both with and without regression against
grey matter signals. The script outputs:

1. CSV file of full % signal change time series (GM, Thalamus, D1, D2, NE).
2. Full and zoomed-in plots of receptor-weighted and regional time series.
3. Summary grid plot combining GM, Thalamus, and receptor-weighted signals.

Usage
-----
Run this script directly to process one or more subjects. Example:

    python extract_bold_timeseries_with_receptor_weights.py

The script will attempt to locate both fMRI and segmentation files for each
subject, run the pipeline, and save outputs in a subject-specific directory.
"""



import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img,index_img, math_img
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_epi
from nilearn import image
from nilearn.masking import apply_mask
from neuromaps import datasets
from pathlib import Path
import tempfile
from neuromaps.datasets import fetch_atlas
from nilearn.maskers import NiftiLabelsMasker
from neuromaps.datasets import fetch_annotation
from neuromaps.stats import compare_images
from neuromaps.resampling import resample_images
from neuromaps.images import load_data
from neuromaps.nulls import burt2018
from pathlib import Path
from nilearn import plotting
import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.image import resample_img, new_img_like, clean_img
from nilearn.plotting import plot_stat_map
from nilearn import datasets 
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure, gaussian_filter
from nilearn import datasets
import matplotlib as mpl
mni152= datasets.load_mni152_template(resolution=2)
TR = 2.22
grey_labels = [3, 42]
thalamus_labels = [10, 49]
chunk_size = 243
zoom_windows = [(0, 300), (1000, 1300), (3200, 3500)]

def load_grey_matter_mask(seg_path):
    """
    Load a grey matter mask from FreeSurfer aseg segmentation.

    Parameters
    ----------
    seg_path : str or Path
        Path to aseg segmentation NIfTI file.

    Returns
    -------
    grey_mask : np.ndarray (bool)
        Binary mask of grey matter voxels.
    affine : np.ndarray
        Affine transformation of the segmentation image.
    """


    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    grey_mask = np.isin(seg_data, grey_labels)
    return grey_mask.astype(bool), seg_img.affine


def load_thalamus_mask(seg_path):
     """
    Load a thalamus mask from FreeSurfer aseg segmentation.

    Parameters
    ----------
    seg_path : str or Path
        Path to aseg segmentation NIfTI file.

    Returns
    -------
    thalamus_mask : np.ndarray (bool)
        Binary mask of thalamus voxels.
    affine : np.ndarray
        Affine transformation of the segmentation image.
    """
    
    seg_img = nib.load(seg_path)
    seg_data=  seg_img.get_fdata()
    thalamus_mask = np.isin(seg_data,thalamus_labels)
    return thalamus_mask.astype(bool), seg_img.affine

def load_striatum_mask(reference_img, data_dir, dilate_iterations=3):
     """
    Load and resample striatum mask from Harvard-Oxford atlas.

    Parameters
    ----------
    reference_img : nib.Nifti1Image
        fMRI image to which the mask will be resampled.
    data_dir : str or Path
        Directory to download/store the atlas.
    dilate_iterations : int, optional
        Number of dilation steps to expand the striatum mask (default=3).

    Returns
    -------
    mask_resampled : np.ndarray (bool)
        Resampled binary striatum mask.
    """

    atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm', symmetric_split=True, data_dir=data_dir)
    atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
    atlas_data = atlas_img.get_fdata()
    labels = atlas.labels

    
    striatum_labels = ['Left Caudate', 'Right Caudate', 'Left Putamen', 'Right Putamen', 'Left Accumbens', 'Right Accumbens']
    striatum_indices = [i for i, name in enumerate(labels) if name in striatum_labels]

    # mask with striatum voxels as True
    striatum_mask = np.isin(atlas_data, striatum_indices)
    struct = generate_binary_structure(3,2)
    striatum_mask_dilated = binary_dilation(striatum_mask, structure=struct,iterations= dilate_iterations)
    atlas_resampled = resample_img(nib.Nifti1Image(striatum_mask_dilated.astype(np.int32), atlas_img.affine),
                                   target_affine=reference_img.affine,
                                   target_shape=reference_img.shape[:3],
                                   interpolation='nearest')

    # visualize the striatum mask just to make sure
    striatum_vis_img = nib.Nifti1Image(atlas_resampled.get_fdata(), reference_img.affine)
    nib.save(striatum_vis_img, os.path.join(data_dir, "striatum_mask_resampled.nii.gz"))

    plot_stat_map(
    striatum_vis_img,
    bg_img=mni152,
    title="Striatum Mask (Resampled to fMRI space)",
    cmap="Greens",
    display_mode='ortho',
    output_file=os.path.join(data_dir, "striatum_mask_visualization.png"),
    cut_coords=(0, 0, 0),  
    colorbar=False
    )
    return atlas_resampled.get_fdata().astype(bool)

def plot_neurotransmitter_maps(imgs, titles, main_title, output_path):
    fig, axes = plt.subplots(1, len(imgs), figsize=(12, 4))
    
    vmin, vmax = None, None  # Let nilearn decide, or set manually
    display_objs = []

    for ax, img, subtitle in zip(axes, imgs, titles):
        display = plot_stat_map(
            img,
            bg_img=mni152,
            cmap="turbo",
            display_mode=subtitle.lower(),  # 'axial', 'coronal', 'sagittal'
            colorbar=False,
            draw_cross=False,
            axes=ax
        )
        display_objs.append(display)
        ax.set_title("")  # Remove small titles

    # Add one shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Receptor Density", fontsize=12)
    cbar.ax.set_yticklabels(['Low Density', '', '', '', 'High Density'])

    # Big title for the neurotransmitter
    fig.suptitle(main_title, fontsize=18, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close(fig) 


    
def load_annotations(fmri_img, data_dir):
     """
    Load and resample neuromaps receptor annotations (D1, D2, NE).

    Parameters
    ----------
    fmri_img : nib.Nifti1Image
        fMRI image to resample annotations into subject space.
    data_dir : str or Path
        Directory where neuromaps annotations will be stored.

    Returns
    -------
    annotations_data : dict
        Dictionary mapping receptor name → resampled annotation data (3D array).
    """

    annotations = {
    'Alarkurtti (D2 receptor)': fetch_annotation(source='alarkurtti2015', desc='raclopride', space='MNI152', res='3mm', data_dir=data_dir),
    'kaller (D1 receptor)': fetch_annotation(source='kaller2017', desc='sch23390', space='MNI152', res='3mm', data_dir=data_dir),
    'hesse (Norepinephrine)':fetch_annotation(source='hesse2017', desc='methylreboxetine', space='MNI152', res='3mm', data_dir=data_dir) 
    } 


    annotations_data = {}
    processed_imgs = {}

    for name, path in annotations.items():
        anno_img_orig = nib.load(path)
        anno_img = resample_img(
            anno_img_orig,
            target_affine=fmri_img.affine,
            target_shape=fmri_img.shape[:3],
            interpolation='continuous',
            force_resample=True,
            copy_header=True
        )
        anno_data = anno_img.get_fdata()

        # For D1: remove striatum
        if name == 'D1 receptor':
            striatum_mask = load_striatum_mask(fmri_img, data_dir, dilate_iterations=3)
            anno_data[striatum_mask] = 0

        # Shift values to be non-negative
        min_val = np.min(anno_data)
        shifted_data = anno_data - min_val if min_val < 0 else anno_data.copy()
        shifted_img = nib.Nifti1Image(shifted_data, affine=fmri_img.affine)

        annotations_data[name] = shifted_data
        processed_imgs[name] = shifted_img

    # --- Create combined figure ---
    fig, axes = plt.subplots(
        nrows=3, ncols=3,
        figsize=(10, 10),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.15}
    )

    display_modes = ['z', 'y', 'x']  # axial, coronal, sagittal
    all_data = np.concatenate([img.get_fdata().ravel() for img in processed_imgs.values()])
    vmin, vmax = np.percentile(all_data[all_data > 0], [2, 98])

    for row_idx, (nt_name, img) in enumerate(processed_imgs.items()):
        for col_idx, mode in enumerate(display_modes):
            ax = axes[row_idx, col_idx]
            display = plot_stat_map(
                img,
                bg_img=mni152,
                cmap="turbo",
                display_mode=mode,
                colorbar=False,
                draw_cross=False,
                vmin=vmin,
                vmax=vmax,
                axes=ax
            )
            ax.set_title("")  # remove small titles

        # Big neurotransmitter label on left
        axes[row_idx, 0].set_ylabel(nt_name, fontsize=14, rotation=90, labelpad=15)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Receptor Density", fontsize=12)
    cbar.ax.set_yticklabels(['Low Density', '', '', '', 'High Density'])

    # Main title
    fig.suptitle("Neurotransmitter Receptor Density Maps", fontsize=16, y=0.98)

    # Save single PNG in data_dir
    output_path = os.path.join(data_dir, "neurotransmitter_receptor_density_maps.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return annotations_data





def run_pipeline(subject_id, fmri_path, seg_path, data_dir):
    
    """
    Run receptor-weighted time series extraction pipeline for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    fmri_path : str or Path
        Path to subject's preprocessed fMRI file in MNI152 space.
    seg_path : str or Path
        Path to subject's segmentation (aseg) file.
    data_dir : str or Path
        Directory to save outputs and store annotation data.

    Outputs
    -------
    - CSV file with BOLD time series (GM, Thalamus, D1, D2, NE).
    - PNG plots of time series (full and zoomed).
    - QC plots of anatomical masks and receptor maps.
    """

    print("Running subject:{}".format(subject_id))

    base_dir = "/home/vob3"
    output_dir = os.path.join(base_dir, subject_id)
    os.makedirs(output_dir, exist_ok=True)
    print("Saving all outputs to: {}".format(output_dir))


    
    fmri_img = nib.load(fmri_path)
    fmri_data = fmri_img.dataobj
    n_timepoints = fmri_img.shape[-1]
    thalamus_mask, _ = load_thalamus_mask(seg_path)
    gm_mask, _ = load_grey_matter_mask(seg_path)
    annotations_data = load_annotations(fmri_img, data_dir)

    results = {name: {'with_reg':[], 'without_reg':[]} for name in annotations_data}
    results['gm_signal'] = []
    results['thalamus_signal'] = []

    for start in range(0, n_timepoints, chunk_size):
        end = min(start + chunk_size, n_timepoints)
        print(" Processing timepoints {}-{}".format(start, end))

        chunk_data = np.asanyarray(fmri_data[..., start:end])
        chunk_img = new_img_like(fmri_img, chunk_data)

        gm_ts = chunk_data[gm_mask]
        gm_signal = gm_ts.mean(axis=0)
        results['gm_signal'].append(gm_signal)

        thalamus_ts = chunk_data[thalamus_mask]
        thalamus_signal = thalamus_ts.mean(axis=0)
        results['thalamus_signal'].append(thalamus_signal)

        cleaned_img = clean_img(chunk_img, confounds=gm_signal[:, np.newaxis], detrend=False, standardize=False)

        for name, anno_data in annotations_data.items():
            flat_data = anno_data[anno_data > 0]
            if flat_data.size == 0:
                continue
            thresh = np.percentile(flat_data, 90)
            mask = anno_data >= thresh
            weights = anno_data[mask]

            mask_img = nib.Nifti1Image(mask.astype(np.int32), affine=fmri_img.affine)
            masker = NiftiMasker(mask_img=mask_img, standardize=False)

            ts_orig = masker.fit_transform(chunk_img)
            ts_cleaned = masker.transform(cleaned_img)

            weighted_ts_orig = (ts_orig * weights).sum(axis=1)/ weights.sum()
            weighted_ts_cleaned = (ts_cleaned * weights).sum(axis=1) / weights.sum()

            results[name]['without_reg'].append(weighted_ts_orig)
            results[name]['with_reg'].append(weighted_ts_cleaned)


        # Plotting
        for name in annotations_data:
            ts_without = np.concatenate(results[name]['without_reg'])
            ts_with = np.concatenate(results[name]['with_reg'])
            gm_signal_full = np.concatenate(results['gm_signal'])

            ts_without_psc = (ts_without - ts_without.mean())/ ts_without.mean() * 100
            ts_with_psc = (ts_with - ts_with.mean())/ ts_with.mean() * 100
            gm_signal_psc = (gm_signal_full - gm_signal_full.mean())/gm_signal_full.mean() * 100

            time_sec = np.arange(len(ts_with_psc)) * TR


            # Full plot
            plt.figure(figsize=(12, 4))
            plt.plot(time_sec, ts_without_psc, label="Original")
            plt.plot(time_sec, ts_with_psc, label="Cleaned", linestyle="-")
            plt.plot(time_sec, gm_signal_psc, label="GM Signal", linestyle="-", color="gray")
            plt.title("{} | {}".format(subject_id, name))
            plt.xlabel("Time (s)")
            plt.ylabel("% Signal Change")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "{}_{}_full_timeseries.png".format(subject_id, name)))
            plt.close()

            # Zoomed-in windows
            for start, end in zoom_windows:
                i = (time_sec >= start) & (time_sec <= end)
                if np.any(i):
                    plt.figure(figsize=(10, 3))
                    plt.plot(time_sec[i], ts_without_psc[i], label="Original")
                    plt.plot(time_sec[i], ts_with_psc[i], label="Cleaned", linestyle="-")
                    plt.plot(time_sec[i], gm_signal_psc[i], label="GM Signal", linestyle="-", color="gray")
                    plt.title("{} | {} : {}-{}s".format(subject_id, name, start, end))
                    plt.xlabel("Time (s)")
                    plt.ylabel("% Signal Change")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "{}_{}_{}_{}s_zoom.png".format(subject_id, name, start, end)))
                    plt.close()

        thalamus_signal_full = np.concatenate(results['thalamus_signal'])
        thalamus_signal_psc = (thalamus_signal_full - thalamus_signal_full.mean())/thalamus_signal_full.mean() * 100
        time_sec = np.arange(len(thalamus_signal_psc)) * TR

        plt.figure(figsize=(12, 4))
        plt.plot(time_sec, thalamus_signal_psc, label="Thalamus Signal", color="purple")
        plt.title("{} | Thalamus Time Series".format(subject_id))
        plt.xlabel("Time (s)")
        plt.ylabel("% Signal Change")
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}_thalamus_full_timeseries.png".format(subject_id))
        plt.close()


        
      

        gm_signal_full = np.concatenate(results['gm_signal'])
        gm_signal_psc= (gm_signal_full - gm_signal_full.mean())/ gm_signal_full.mean()*100

        thalamus_signal_full = np.concatenate(results['thalamus_signal'])
        thalamus_signal_psc= (thalamus_signal_full - thalamus_signal_full.mean())/ thalamus_signal_full.mean()*100

        d2_psc_with_reg = (np.concatenate(results['Alarkurtti (D2 receptor)']['with_reg'])-np.concatenate(results['Alarkurtti (D2 receptor)']['with_reg']).mean())/np.concatenate(results['Alarkurtti (D2 receptor)']['with_reg']).mean()* 100

        d1_psc_with_reg = (np.concatenate(results['kaller (D1 receptor)']['with_reg'])-np.concatenate(results['kaller (D1 receptor)']['with_reg']).mean())/np.concatenate(results['kaller (D1 receptor)']['with_reg']).mean()* 100

        norepinephrine_psc_with_reg = (np.concatenate(results['hesse (Norepinephrine)']['with_reg'])-np.concatenate(results['hesse (Norepinephrine)']['with_reg']).mean())/np.concatenate(results['hesse (Norepinephrine)']['with_reg']).mean()* 100

        d2_psc_without_reg = (np.concatenate(results['Alarkurtti (D2 receptor)']['without_reg'])-np.concatenate(results['Alarkurtti (D2 receptor)']['without_reg']).mean())/np.concatenate(results['Alarkurtti (D2 receptor)']['without_reg']).mean()* 100

        d1_psc_without_reg = (np.concatenate(results['kaller (D1 receptor)']['without_reg'])-np.concatenate(results['kaller (D1 receptor)']['without_reg']).mean())/np.concatenate(results['kaller (D1 receptor)']['without_reg']).mean()* 100


        norepinephrine_psc_without_reg = (np.concatenate(results['hesse (Norepinephrine)']['without_reg'])-np.concatenate(results['hesse (Norepinephrine)']['without_reg']).mean())/np.concatenate(results['hesse (Norepinephrine)']['without_reg']).mean()* 100


        n_points = len(gm_signal_psc)
        time_sec = np.arange(n_points) *TR

        data_matrix = np.column_stack((time_sec, gm_signal_psc, thalamus_signal_psc, d2_psc_with_reg, d1_psc_with_reg,   norepinephrine_psc_with_reg, d2_psc_without_reg, d1_psc_without_reg, norepinephrine_psc_without_reg))

        header= "Time_s,GM_PSC,Thalamus_PSC,D2_with_reg_PSC,D1_with_reg_PSC,Norepinephrine_with_reg,D2_without_reg_PSC,D1_without_reg_PSC,Norepinephrine_without_reg"

        np.savetxt(os.path.join(output_dir, "{}_bold_timeseries.csv".format(subject_id)), data_matrix, delimiter=",", header=header, comments='')
        print("Saved full BOLD time series to {}_bold_timeseries.csv".format(subject_id))


        # Final plot combining GM and Thalamus (Top), and Receptor maps (Bottom)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top row: GM and Thalamus
        axes[0].plot(time_sec, gm_signal_psc, label="GM Signal", color="blue")
        axes[0].plot(time_sec, thalamus_signal_psc, label="Thalamus Signal", color="green")
        axes[0].set_title("GM and Thalamus Signals")
        axes[0].set_ylabel("% Signal Change")
        axes[0].legend()

        # Bottom row: D1, D2, NE with regression
        axes[1].plot(time_sec, d1_psc_with_reg, label="D1", color="purple")
        axes[1].plot(time_sec, d2_psc_with_reg, label="D2", color="orange")
        axes[1].plot(time_sec, norepinephrine_psc_with_reg, label="NE", color="cyan")
        axes[1].set_title("Receptor-Weighted Signals")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("% Signal Change")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject_id}_bold_timeseries_grid.png"))
        plt.close()
if __name__=="__main__":
    subjects = ["sub-racsleep05"]
    base_path= Path("/orcd/data/ldlewis/001/om/hf303/for_ore")
    data_dir = Path("neuromaps_annotations")
    data_dir.mkdir(exist_ok=True)

    for subj in subjects:
      func_dir = base_path / subj / "func"

      # Try possible variations
      fmri_file_001 = func_dir /"{}_task-rest_run-001_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz".format(subj)
      fmri_file_1   = func_dir / "{}_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz".format(subj)

      if fmri_file_001.exists():
          fmri_path = fmri_file_001
      elif fmri_file_1.exists():
          fmri_path = fmri_file_1
      else:
          print("No fMRI file found for {s}".format(subj))
          continue

    
      seg_file_001 = func_dir / "{}_task-rest_run-001_space-MNI152NLin2009cAsym_res-2_desc-aseg_dseg.nii.gz".format(subj)
      seg_file_1   = func_dir / "{}_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-aseg_dseg.nii.gz".format(subj)

      if seg_file_001.exists():
          seg_path = seg_file_001
      elif seg_file_1.exists():
          seg_path = seg_file_1
      else:
          print("No segmentation file for {}".format(subj))
          continue
        
    
      run_pipeline(subj, fmri_path, seg_path, data_dir)