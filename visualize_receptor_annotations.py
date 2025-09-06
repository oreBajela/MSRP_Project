"""
visualize_receptor_annotations.py

This script is part of the dopamine button press project. 
It loads neurotransmitter receptor density maps (D1, D2, NE) from 
neuromaps and overlays them on the MNI152 brain template. It also 
applies striatal masking to remove overlapping regions when needed.

Two main tasks:
1. Load and resample receptor density maps (to fMRI resolution).
2. Generate orthogonal slice visualizations (axial, coronal, sagittal) 
   for each receptor and save them as a combined grid figure.

Usage:
    python visualize_receptor_annotations.py

Inputs:
    - Reference fMRI image (for resampling target).
    - Neuromaps receptor annotations (downloaded automatically).
    - Harvard-Oxford atlas for striatal masking.

Outputs:
    - Combined receptor slice figure (PNG).
    - Dictionary of annotation data arrays (per receptor).
"""



import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps
from nilearn.image import resample_img
from neuromaps.datasets import fetch_annotation
from nilearn.plotting import plot_stat_map
from nilearn import datasets

# Load MNI template (for reference space)
mni152 = datasets.load_mni152_template(resolution=2)

def load_striatum_mask(reference_img, data_dir, dilate_iterations=3):

     """
    Creates a striatum mask from the Harvard-Oxford atlas and resamples 
    it to the reference image grid. The mask is dilated to cover 
    surrounding voxels.

    Args:
        reference_img (nib.Nifti1Image): Target image for resampling.
        data_dir (str): Directory where atlas data will be cached/downloaded.
        dilate_iterations (int): Number of dilation steps for mask expansion.

    Returns:
        np.ndarray: Boolean 3D mask array (True = striatum).
    """
 
    from nilearn import datasets
    from scipy.ndimage import binary_dilation, generate_binary_structure
    atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm', symmetric_split=True, data_dir=data_dir)
    atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
    atlas_data = atlas_img.get_fdata()
    labels = atlas.labels


     # Select striatum-related labels
    striatum_labels = ['Left Caudate', 'Right Caudate', 'Left Putamen', 'Right Putamen', 'Left Accumbens', 'Right Accumbens']
    striatum_indices = [i for i, name in enumerate(labels) if name in striatum_labels]

    # Create mask and apply dilation
    striatum_mask = np.isin(atlas_data, striatum_indices)
    struct = generate_binary_structure(3, 2)
    striatum_mask_dilated = binary_dilation(striatum_mask, structure=struct, iterations=dilate_iterations)

    # Resample mask to reference image space
    atlas_resampled = resample_img(
        nib.Nifti1Image(striatum_mask_dilated.astype(np.int32), atlas_img.affine),
        target_affine=reference_img.affine,
        target_shape=reference_img.shape[:3],
        interpolation='nearest'
    )

    return atlas_resampled.get_fdata().astype(bool)


def load_annotations(fmri_img, data_dir):

     """
    Loads and processes neurotransmitter receptor density annotations 
    (D1, D2, NE). Resamples them to fMRI resolution, applies optional 
    striatum masking for D1, and generates slice visualizations.

    Args:
        fmri_img (nib.Nifti1Image): Reference fMRI image for resampling.
        data_dir (str): Directory for caching annotation files and saving plots.

    Returns:
        dict: Dictionary of processed annotation arrays keyed by receptor name.
    """

    annotations = {
        'Alarkurtti (D2 receptor)': fetch_annotation(source='alarkurtti2015', desc='raclopride', space='MNI152', res='3mm', data_dir=data_dir),
        'kaller (D1 receptor)': fetch_annotation(source='kaller2017', desc='sch23390', space='MNI152', res='3mm', data_dir=data_dir),
        'hesse (Norepinephrine)': fetch_annotation(source='hesse2017', desc='methylreboxetine', space='MNI152', res='3mm', data_dir=data_dir)
    }

    annotations_data = {}
    all_slices = {}

      # Resample and preprocess each annotation
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

        # Apply striatum mask removal for D1 receptor map
        if name == 'kaller (D1 receptor)':
            striatum_mask = load_striatum_mask(fmri_img, data_dir, dilate_iterations=3)
            anno_data[striatum_mask] = 0

        
        # Shift values to ensure all nonnegative
        min_val = np.min(anno_data)
        shifted_anno_data = anno_data - min_val if min_val < 0 else anno_data.copy()

        # Extract mid-slices for visualization
        mid_slices = np.array(shifted_anno_data.shape) // 2
        slices = {
            'Axial': shifted_anno_data[:, :, mid_slices[2]].T,
            'Coronal': np.flip(np.rot90(shifted_anno_data[:, mid_slices[1], :])),
            'Sagittal': np.flip(np.rot90(shifted_anno_data[mid_slices[0], :, :]))
        }
        clean_name = name.split("(")[-1].replace(")", "").strip()
        all_slices[clean_name] = slices
        annotations_data[name] = shifted_anno_data

     # Visualization: receptor slices grid
    view_names = ['Axial', 'Coronal', 'Sagittal']
    receptor_names = list(all_slices.keys())
    n_rows = len(receptor_names)
    n_cols = len(view_names)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(13.95, 3 * n_rows))

    cmap = colormaps.get_cmap('turbo')
    all_data = np.concatenate([
        all_slices[rec][view].flatten()
        for rec in receptor_names for view in view_names
    ])
    vmax = np.percentile(all_data[all_data > 0], 99)
    norm = Normalize(vmin=0, vmax=vmax)
    im = None

    for i, rec in enumerate(receptor_names):
        for j, view in enumerate(view_names):
            data_slice = all_slices[rec][view]
            alpha_data = np.where(data_slice > 0.05, 1.0, 0.0).astype(np.float32)

            ax = axs[i, j] if n_rows > 1 else axs[j]
            im = ax.imshow(data_slice, cmap=cmap, norm=norm, origin='lower',
                           interpolation='none', alpha=alpha_data)
            ax.axis('off')

        mid_ax = axs[i, n_cols // 2] if n_rows > 1 else axs[n_cols // 2]
        mid_pos = mid_ax.get_position()
        fig.text(mid_pos.x0 + mid_pos.width / 2,
                 mid_pos.y1 + 0.02,
                 rec,
                 ha='center', va='bottom',
                 fontsize=14, fontweight='bold')

    # Add receptor label above each row
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, vmax])
    cbar.set_ticklabels(["Low Density", "High Density"])
    cbar.ax.tick_params(labelsize=10)

     # Finalize figure
    fig.suptitle("Neurotransmitter Receptor Density Maps", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.92])
    output_path = os.path.join(data_dir, "All_Annotations_Combined_Grid.png")
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()

    return annotations_data
