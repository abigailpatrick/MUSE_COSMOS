#!/usr/bin/env python3
"""
muse_slice_var_and_vap_mosaic.py

Create:
 - inverse-variance combined variance mosaic (from STAT HDU,)
 - background-derived variance (VAP) mosaic (from DATA HDU using photutils.Background2D)

Both mosaics are saved to FITS and optionally plotted.

"""

import os
import argparse
import csv
import time
import shutil
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_adaptive
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

start_time = time.time()


def parse_args():
    parser = argparse.ArgumentParser(description="MUSE variance slice mosaicking pipeline (variance-only)")

    parser.add_argument('--path', type=str, required=True,
                        help='Path to directory containing the cubes')
    parser.add_argument('--offsets_txt', type=str, required=True,
                        help='Text file containing cube offsets')
    parser.add_argument('--slice', type=int, required=True,
                        help='Wavelength slice to extract from each cube (0-indexed)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output FITS file path for the summed-variance mosaic (base name)')
    parser.add_argument('--mask_percent', type=float, default=1.0,
                        help='Percentage of lowest pixels to mask (default 1.0); must match white-light mask creation step')
    parser.add_argument('--plotting', action='store_true',
                        help='Enable plotting of the variance & VAP mosaics')
    parser.add_argument('--tmp_dir', type=str,
                        default='/cephfs/apatrick/musecosmos/scripts/aligned/tmp_slice_var',
                        help='Temporary directory for reprojected slices')
    parser.add_argument('--box_size', type=int, default=50,
                        help='Box size (pixels) for Background2D VAP estimation (default 50)')
    parser.add_argument('--filter_size', type=int, default=3,
                        help='Median filter size for Background2D (default 3)')
    parser.add_argument('--sigma_clip', type=float, default=3.0,
                        help='Sigma for sigma-clipping used by Background2D (default 3.0)')

    return parser.parse_args()


@dataclass
class CubeEntry:
    file_id: str       # e.g. "Autocal_3687411a_1"
    x_offset: float
    y_offset: float
    flag: str          # 'a' or 'm'
    cube_path: str = ""


def paths_ids_offsets(offsets_txt, cubes_dir):
    cubes = {}
    with open(offsets_txt, "r") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                image_path = parts[0]
                x_offset = float(parts[1])
                y_offset = float(parts[2])
                flag = parts[3]

                filename = os.path.basename(image_path)
                file_id = filename.replace("DATACUBE_FINAL_", "").replace("_ZAP_img.fits", "")
                norm_filename = f"DATACUBE_FINAL_{file_id}_ZAP_norm.fits"
                cube_path = os.path.join(cubes_dir, norm_filename)

                if os.path.exists(cube_path) and file_id not in cubes:
                    cubes[file_id] = CubeEntry(file_id=file_id, x_offset=x_offset, y_offset=y_offset,
                                               flag=flag, cube_path=cube_path)
                elif not os.path.exists(cube_path):
                    print(f"Skipping {norm_filename}, not found in {cubes_dir}")

    return list(cubes.values())


def var_slice(cubes, slice_number):
    """
    Extract a 2D variance slice from each cube (HDU 2: STAT).
    Returns dict: {file_id: {'data': 2D array, 'wcs': celestial WCS, 'wcs_e': full header}}
    """
    out = {}
    for cube in cubes:
        with fits.open(cube.cube_path) as hdul:
            data = hdul[2].data[slice_number]
            header = hdul[2].header.copy()
            wcs2d = WCS(header).celestial
            out[cube.file_id] = {'data': data, 'wcs': wcs2d, 'wcs_e': header}
    return out


def data_slice(cubes, slice_number):
    """
    Extract a 2D data slice from each cube (HDU 1: DATA).
    Returns dict: {file_id: {'data': 2D array, 'wcs': celestial WCS, 'wcs_e': full header}}
    """
    out = {}
    for cube in cubes:
        with fits.open(cube.cube_path) as hdul:
            data = hdul[1].data[slice_number]
            header = hdul[1].header.copy()
            wcs2d = WCS(header).celestial
            out[cube.file_id] = {'data': data, 'wcs': wcs2d, 'wcs_e': header}
    return out


def apply_saved_masks(aligned_slices, cubes, mask_dir, mask_percent=1.0):
    """
    Apply precomputed white-light masks to each aligned slice.
    True mask pixels are set to NaN, excluded from sum.
    """
    masked_slices = []
    for i, cube in enumerate(cubes):
        mask_filename = f"DATACUBE_FINAL_{cube.file_id}_ZAP_img_aligned_mask{int(mask_percent)}p.fits"
        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(mask_path):
            mask_data = fits.getdata(mask_path).astype(bool)
            masked_data = np.where(mask_data, np.nan, aligned_slices[i]['data'])
            masked_slices.append({
                'data': masked_data,
                'wcs': aligned_slices[i]['wcs'],
                'applied_offset': aligned_slices[i]['applied_offset']
            })
            print(f"Applied saved mask to {cube.file_id}")
        else:
            masked_slices.append(aligned_slices[i])
            print(f"WARNING: No mask found for {cube.file_id}, skipping mask")
    return masked_slices


def slice_wavelength_check(cube_path, slice_number, expected_start=4749.9, expected_step=1.25):
    """
    Validate wavelength using HDU 1 (DATA) spectral WCS (independent of variance).
    """
    with fits.open(cube_path) as hdul:
        header = hdul[1].header
        naxis3 = header['NAXIS3']

        if "CDELT3" in header and "CRVAL3" in header:
            crval3 = header['CRVAL3']
            cdelt3 = header['CDELT3']
            slice_wavelength = crval3 + cdelt3 * slice_number
        else:
            w = WCS(header)
            pix = np.arange(naxis3)
            wavelengths = w.all_pix2world(np.zeros(naxis3), np.zeros(naxis3), pix, 0)[2]
            slice_wavelength = wavelengths[slice_number] * 1e10  # convert to Å

    expected_slice_wavelength = expected_start + expected_step * slice_number
    within_tolerance = abs(slice_wavelength - expected_slice_wavelength) <= 1.0
    if not within_tolerance:
        raise ValueError(f"WARNING: {cube_path} slice {slice_number} wavelength {slice_wavelength} Å "
                         f"not within tolerance of expected {expected_slice_wavelength} Å")
    return slice_wavelength


def align_slices(slice_data, slice_wcs, offsets):
    """
    Apply pixel offsets to WCS only (data remain unchanged).
    """
    aligned = []
    for data, wcs, (dx, dy) in zip(slice_data, slice_wcs, offsets):
        new_wcs = wcs.deepcopy()
        new_wcs.wcs.crpix[0] -= dx  # RA
        new_wcs.wcs.crpix[1] += dy  # Dec
        aligned.append({'data': data, 'wcs': new_wcs, 'applied_offset': (dx, dy)})
    return aligned


def common_wcs_area(aligned_slices):
    """
    Find optimal common WCS area with optional padding.
    """
    slice_list = [(s['data'], s['wcs'].celestial) for s in aligned_slices]
    wcs_out, shape_out = find_optimal_celestial_wcs(slice_list)

    # Optional padding
    pad_y, pad_x = 100, 100
    shape_out = (shape_out[0] + pad_y, shape_out[1] + pad_x)
    wcs_out.wcs.crpix[0] += pad_x // 2
    wcs_out.wcs.crpix[1] += pad_y // 2
    return wcs_out, shape_out


def reproject_and_save_single(i, slice_dict, wcs_out, shape_out, output_dir):
    """
    Reproject a single aligned slice (variance or data) onto common WCS and save.
    """
    os.makedirs(output_dir, exist_ok=True)
    data, wcs = slice_dict['data'], slice_dict['wcs'].celestial

    array, _ = reproject_adaptive((data, wcs), output_projection=wcs_out,
                                  shape_out=shape_out, conserve_flux=False)

    header = wcs_out.to_header()
    fname = os.path.join(output_dir, f"reproj_slice_{i:03d}_pid{os.getpid()}.fits")
    fits.writeto(fname, array, header=header, overwrite=True)
    return fname


def _map_reproject_and_save(args):
    return reproject_and_save_single(*args)


def reproject_and_save_slices(aligned_slices, wcs_out, shape_out, output_dir, max_workers=8):
    """
    Parallel reprojection of all slices. Returns list of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    args_list = [(i, s, wcs_out, shape_out, output_dir) for i, s in enumerate(aligned_slices)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        file_list = list(executor.map(_map_reproject_and_save, args_list))
    return file_list


def variance_sum_from_files(file_list):
    """
    Combine reprojected variance slices using inverse-variance summation (1/sum(1/var)).
    Returns combined_var (2D array) and WCS built from the first file header.
    """
    sum_inv = None
    first_header = None

    for i, f in enumerate(file_list):
        data = fits.getdata(f, memmap=True)  # reprojected variance slice

        # Mask out bad values: NaN, non-finite, or non-positive variances don't contribute
        good = np.isfinite(data) & (data > 0.0)

        # compute inverse variance safely: set inv where good, else zero
        inv = np.zeros_like(data, dtype=float)
        if np.any(good):
            inv[good] = 1.0 / data[good]

        if sum_inv is None:
            sum_inv = np.zeros_like(inv, dtype=float)
            first_header = fits.getheader(f)

        # accumulate inverse-variance
        sum_inv += inv

    # convert sum_inv -> combined variance
    combined_var = np.full_like(sum_inv, np.nan, dtype=float)
    positive = sum_inv > 0.0
    combined_var[positive] = 1.0 / sum_inv[positive]

    wcs = WCS(first_header) if first_header is not None else None
    return combined_var, wcs


def background_variance_map(data, box_size=(30, 30), filter_size=(5, 5), sigma=3.0):
    """
    Estimate a 2D background-based variance map for a given MUSE slice
    using photutils.Background2D (sigma-clipped, no explicit source mask).

    Input:
      data : 2D image (flux) array
      box_size : int or tuple (box y, box x)
      filter_size : int or tuple for median filtering of the low-res grid
      sigma : sigma for SigmaClip

    Returns:
      var_map : 2D numpy.ndarray (variance = background_rms^2, same shape as data)
      bkg_rms : 2D numpy.ndarray (background RMS)
      bkg_obj : Background2D object (for diagnostics)
    """
    # Replace NaNs with zeros for Background2D input (ignored in stats)
    indata = np.array(data, dtype=float)
    indata = np.nan_to_num(indata, nan=0.0, posinf=0.0, neginf=0.0)

    sigma_clip = SigmaClip(sigma=sigma, maxiters=10)
    bkg_estimator = MedianBackground()

    bkg = Background2D(
        indata,
        box_size=box_size,
        filter_size=(filter_size, filter_size) if isinstance(filter_size, int) else filter_size,
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
        edge_method='pad'
    )

    bkg_rms = bkg.background_rms
    var_map = bkg_rms ** 2
    return var_map, bkg_rms, bkg


def save_variance_mosaic(mosaic, mosaic_wcs, output_file, file_ids, offsets, a_m, wcs_e):
    """
    Save the summed-variance mosaic to FITS with merged headers and provenance.
    """
    mosaic_header = mosaic_wcs.to_header()

    # Merge non-spectral header keys from the variance HDU header object
    for card in wcs_e.cards:
        key = card.keyword
        value = card.value

        if key.endswith('3'):
            continue
        if key == 'COMMENT' and value is not None:
            mosaic_header.add_comment(str(value))
        elif key == 'HISTORY' and value is not None:
            mosaic_header.add_history(str(value))
        elif key not in mosaic_header:
            mosaic_header[key] = value

    mosaic_header.add_history("Variance mosaic created by summing STAT (HDU 2) slices after reprojection.")
    mosaic_header['VARCOMB'] = ('INVVAR', 'Variance combined via inverse-variance weighting (1/sum(1/var))')

    hdu = fits.PrimaryHDU(data=mosaic, header=mosaic_header)
    for i, (fid, off, typ) in enumerate(zip(file_ids, offsets, a_m), 1):
        hdu.header[f'FILE{i}'] = fid
        hdu.header[f'OFF{i}'] = str(off)
        hdu.header[f'TYPE{i}'] = typ

    hdu.writeto(output_file, overwrite=True)
    print(f"Saved variance mosaic to {output_file}")


def save_vap_mosaic(vap_map, mosaic_wcs, output_file, file_ids, offsets, a_m, wcs_e):
    """
    Save the background-derived variance (VAP) map to FITS with provenance.
    """
    mosaic_header = mosaic_wcs.to_header()

    # Merge non-spectral header keys from original header
    for card in wcs_e.cards:
        key = card.keyword
        value = card.value
        if key.endswith('3'):
            continue
        if key == 'COMMENT' and value is not None:
            mosaic_header.add_comment(str(value))
        elif key == 'HISTORY' and value is not None:
            mosaic_header.add_history(str(value))
        elif key not in mosaic_header:
            mosaic_header[key] = value

    mosaic_header.add_history("Background-derived variance (VAP) map computed with photutils.Background2D.")
    mosaic_header['VARCOMB'] = ('BKG', 'Variance derived from empirical background noise (RMS^2).')

    hdu = fits.PrimaryHDU(data=vap_map, header=mosaic_header)
    for i, (fid, off, typ) in enumerate(zip(file_ids, offsets, a_m), 1):
        hdu.header[f'FILE{i}'] = fid
        hdu.header[f'OFF{i}'] = str(off)
        hdu.header[f'TYPE{i}'] = typ

    hdu.writeto(output_file, overwrite=True)
    print(f"Saved VAP mosaic to {output_file}")


def plot_mosaic(mosaic, slice_wavelength, output_path=None, title_extra=None, fname_suffix="var"):
    """
    Plot a mosaic (variance or vap) and save PNG.
    """
    if output_path is None:
        output_path = "/cephfs/apatrick/musecosmos/reduced_cubes/slices"
    os.makedirs(output_path, exist_ok=True)

    norm = simple_norm(mosaic, 'sqrt', percent=99.5)

    plt.figure(figsize=(10, 8))
    plt.imshow(mosaic, origin='lower', cmap='viridis', norm=norm)
    title = f"{'Variance' if fname_suffix=='var' else 'Background-derived Variance '} Mosaic of slice {round(slice_wavelength,1)} Å"
    if title_extra:
        title += f" - {title_extra}"
    plt.title(title)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label='Variance')
    plt.tight_layout()

    output_png = os.path.join(output_path, f"mosaic_{fname_suffix}_slice_{round(slice_wavelength, 1)}_full.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_png}")
    return output_png

def plot_ratio_histogram(vap_mosaic, var_mosaic, slice_wavelength, output_dir,
                         bins=100, logx=False, clip_percentiles=(1, 99),
                         fname_tag="ratio_vap_over_var", slice_index=None):
    """
    Plot a 1D line-only histogram of the pixel-wise ratio R = VAP / VAR.

    Parameters
    ----------
    vap_mosaic : 2D ndarray
    var_mosaic : 2D ndarray
    slice_wavelength : float
    output_dir : str
    bins : int
    logx : bool
    clip_percentiles : (float, float) or None
    fname_tag : str
    slice_index : int or None
        Optional slice index to include in filenames for easier aggregation.

    Returns
    -------
    result : dict with stats and file paths
    """

    os.makedirs(output_dir, exist_ok=True)

    var = np.array(var_mosaic, dtype=float)
    vap = np.array(vap_mosaic, dtype=float)

    valid = np.isfinite(var) & np.isfinite(vap) & (var > 0.0) & (vap > 0.0)
    if not np.any(valid):
        raise RuntimeError("No valid pixels to compute ratio (check NaNs/zeros).")

    R = vap[valid] / var[valid]

    # Robust stats on the linear ratio R
    median_R = float(np.nanmedian(R))
    mad_R = float(1.4826 * np.nanmedian(np.abs(R - median_R)))  # robust sigma
    p16, p50, p84 = [float(q) for q in np.nanpercentile(R, [16, 50, 84])]
    N = int(R.size)

    # Transform for plotting axis
    if logx:
        X = np.log10(R)
        xlabel = "log10(VAP / VAR)"
        v_med = np.log10(median_R) if median_R > 0 else np.nan
        v_p16 = np.log10(p16) if p16 > 0 else np.nan
        v_p84 = np.log10(p84) if p84 > 0 else np.nan
    else:
        X = R
        xlabel = "VAP / VAR"
        v_med, v_p16, v_p84 = median_R, p16, p84

    # Percentile clipping for the plotted histogram (does not affect stats)
    if clip_percentiles is not None:
        lo_p, hi_p = np.nanpercentile(X, clip_percentiles)
        Xplot = X[(X >= lo_p) & (X <= hi_p)]
        clipped_range = (float(lo_p), float(hi_p))
    else:
        lo_p, hi_p = float(np.nanmin(X)), float(np.nanmax(X))
        Xplot = X
        clipped_range = (lo_p, hi_p)

    # Compute histogram counts and edges with density normalization
    counts, edges = np.histogram(Xplot, bins=bins, range=(lo_p, hi_p), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Plot as a line-only histogram (step line)
    plt.figure(figsize=(8, 5))
    plt.step(centers, counts, where='mid', color='steelblue', lw=2, label=f"median={median_R:.3g}")
    # Overplot median line in the same x-space for reference
    if np.isfinite(v_med):
        plt.axvline(v_med, color='crimson', ls='-', lw=1.5)

    title = f"Histogram of VAP / VAR for slice {round(slice_wavelength,1)} Å  (N={N})"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Probability density")
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()

    wave_tag = f"{round(slice_wavelength,1)}"
    idx_tag = f"_slice{int(slice_index):04d}" if slice_index is not None else ""
    suffix = "_log" if logx else ""
    hist_png = os.path.join(output_dir, f"{fname_tag}{idx_tag}_{wave_tag}{suffix}.png")
    plt.savefig(hist_png, dpi=200, bbox_inches='tight')
    plt.close()

    # Save histogram data for downstream overlay plotting
    hist_npz = os.path.join(output_dir, f"{fname_tag}{idx_tag}{suffix}.npz")
    np.savez_compressed(
        hist_npz,
        centers=centers,
        counts=counts,
        edges=edges,
        slice_wavelength=slice_wavelength,
        slice_index=(int(slice_index) if slice_index is not None else -1),
        logx=int(logx),
        clip_lo=clipped_range[0],
        clip_hi=clipped_range[1],
        N=N,
        median_ratio=median_R,
        mad_ratio=mad_R,
        p16=p16, p50=p50, p84=p84
    )

    # Append stats CSV
    stats_csv = os.path.join(output_dir, f"{fname_tag}_stats.csv")
    header_needed = not os.path.exists(stats_csv)
    with open(stats_csv, "a") as f:
        if header_needed:
            f.write("slice_index,slice_wavelength_A,N,median_ratio,mad_ratio,p16,p50,p84,clip_lo,clip_hi,logx\n")
        f.write(f"{(slice_index if slice_index is not None else -1)},{slice_wavelength:.3f},{N},"
                f"{median_R:.6g},{mad_R:.6g},{p16:.6g},{p50:.6g},{p84:.6g},"
                f"{clipped_range[0]:.6g},{clipped_range[1]:.6g},{int(logx)}\n")

    print(f"[Ratio] N={N} median(VAP/VAR)={median_R:.4g} MAD~={mad_R:.4g} (p16,p50,p84)=({p16:.4g},{p50:.4g},{p84:.4g})")
    print(f"Suggested scale factor to bring VAR -> VAP: multiply VAR by {median_R:.4g}")
    print(f"Saved line-histogram to {hist_png} and histogram data to {hist_npz}")

    return {
        'N': N,
        'median_ratio': median_R,
        'mad_ratio': mad_R,
        'p16': p16,
        'p50': p50,
        'p84': p84,
        'hist_png': hist_png,
        'stats_csv': stats_csv,
        'hist_npz': hist_npz
    }


def main():
    args = parse_args()

    cubes_dir = args.path
    cubes = paths_ids_offsets(args.offsets_txt, cubes_dir)

    # Ensure unique by file_id
    unique = {}
    for c in cubes:
        unique.setdefault(c.file_id, c)
    cubes = list(unique.values())

    print(f"Found {len(cubes)} unique cubes with offsets.")
    if len(cubes) == 0:
        raise RuntimeError(f"No matching _norm cubes found in {cubes_dir}. Check your paths.")

    # Validate wavelengths and collect slice_wavelength (use first cube for display)
    slice_wavelength = None
    for cube in cubes:
        cube.cube_path = os.path.join(cubes_dir, f"DATACUBE_FINAL_{cube.file_id}_ZAP_norm.fits")
        this_wave = slice_wavelength_check(cube.cube_path, args.slice)
        if slice_wavelength is None:
            slice_wavelength = this_wave
        # write per-cube csv of wavelengths
        csv_filename = f"{int(args.slice)}_wave.csv"
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            if not file_exists:
                csvfile.write("file_id,slice,slice_wavelength\n")
            csvfile.write(f"{cube.file_id},{args.slice},{this_wave:.4f}\n")
    print(f"Completed writing {int(args.slice)}_wave.csv")

    # Extract variance (HDU2) and data (HDU1) slices
    var_slices = var_slice(cubes, args.slice)
    data_slices = data_slice(cubes, args.slice)
    print(f"Extracted {len(var_slices)} variance slices (HDU2) and corresponding data slices (HDU1).")

    # Keep one STAT header for provenance
    wcs_e = var_slices[cubes[0].file_id]['wcs_e']

    # Align via offsets
    offsets = [(c.x_offset, c.y_offset) for c in cubes]

    slice_var_data = [var_slices[c.file_id]['data'] for c in cubes]
    slice_var_wcs = [var_slices[c.file_id]['wcs'] for c in cubes]
    aligned_var = align_slices(slice_var_data, slice_var_wcs, offsets)
    print("Applied pixel offsets to variance slices.")

    slice_img_data = [data_slices[c.file_id]['data'] for c in cubes]
    slice_img_wcs = [data_slices[c.file_id]['wcs'] for c in cubes]
    aligned_data = align_slices(slice_img_data, slice_img_wcs, offsets)
    print("Applied pixel offsets to data slices.")

    # Apply saved masks to variance slices (keep data unmasked; VAP uses data)
    mask_dir = '/cephfs/apatrick/musecosmos/scripts/aligned/masks'
    aligned_var = apply_saved_masks(aligned_var, cubes, mask_dir, args.mask_percent)
    print(f"Applied {args.mask_percent}% masks to variance slices (masks set to NaN).")

    # Find common WCS
    wcs_out, shape_out = common_wcs_area(aligned_var)

    # Reproject all slices to common grid (variance and data)
    tmp_dir = args.tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    print("Reprojecting and saving aligned variance slices")
    var_file_list = reproject_and_save_slices(aligned_var, wcs_out, shape_out, tmp_dir)

    print("Reprojecting and saving aligned data slices")
    data_file_list = reproject_and_save_slices(aligned_data, wcs_out, shape_out, tmp_dir)

    # Combine variance across reprojected slices (inverse-variance)
    mosaic_data, mosaic_wcs = variance_sum_from_files(var_file_list)
    print(f"Created variance-sum mosaic from {len(var_file_list)} reprojected slices.")

    # Save and plot inverse-variance mosaic
    file_ids = [c.file_id for c in cubes]
    a_m = [c.flag for c in cubes]
    save_variance_mosaic(mosaic_data, mosaic_wcs, args.output_file, file_ids, offsets, a_m, wcs_e)
    if getattr(args, 'plotting', False):
        print(f"Plotting variance mosaic for slice {round(slice_wavelength, 1)} Å")
        plot_mosaic(mosaic_data, slice_wavelength, fname_suffix="var", output_path=os.path.dirname(args.output_file))

    # Build background-derived variance (VAP) mosaic from the reprojected data slices:
    # Combine reprojected data slices into a single image (median to mitigate residual sources)
    print("Building background (VAP) mosaic from reprojected data slices…")
    # load arrays from reprojected data files and stack
    reprojected_data_stack = []
    for f in data_file_list:
        arr = fits.getdata(f, memmap=True)
        # treat infinite/NaN
        arr = np.array(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        reprojected_data_stack.append(arr)
    if len(reprojected_data_stack) == 0:
        raise RuntimeError("No reprojected data slices found to build VAP mosaic.")

    # median combine along stack axis (ignore NaNs)
    stacked = np.nanmedian(np.stack(reprojected_data_stack, axis=0), axis=0)

    # estimate VAP map (RMS^2) using Background2D
    vap_map, vap_rms, bkg_obj = background_variance_map(
        stacked,
        box_size=(args.box_size, args.box_size),
        filter_size=(args.filter_size, args.filter_size),
        sigma=args.sigma_clip
    )

    # Save VAP mosaic to file (use output_file base name)
    vap_output_file = args.output_file.replace('.fits', '_vap.fits')
    save_vap_mosaic(vap_map, mosaic_wcs, vap_output_file, file_ids, offsets, a_m, wcs_e)

    # Plot VAP if requested
    if getattr(args, 'plotting', False):
        print(f"Plotting VAP mosaic for slice {round(slice_wavelength, 1)} Å")
        plot_mosaic(vap_map, slice_wavelength, fname_suffix="vap", output_path=os.path.dirname(args.output_file))

    # Histogram of scaling between VAP and VAR (ratio = VAP / VAR)
    out_dir = os.path.dirname(args.output_file)
    try:
        plot_ratio_histogram(vap_map, mosaic_data, slice_wavelength, out_dir,
                             bins=120, logx=False, clip_percentiles=(1, 99),
                             slice_index=args.slice)
        # Optional: also save a log10 histogram for readability
        plot_ratio_histogram(vap_map, mosaic_data, slice_wavelength, out_dir,
                             bins=120, logx=True, clip_percentiles=(1, 99),
                             slice_index=args.slice)
    except Exception as e:
        print(f"WARNING: Failed to make ratio histogram: {e}")


    # Cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Temporary directory {tmp_dir} removed.")

    print(f"Finished variance + VAP mosaics for slice {args.slice} (λ = {round(slice_wavelength,1)} Å)")

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\nCompleted in {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()


"""
to run:
python mosaic_chunk_var.py --path /cephfs/apatrick/musecosmos/reduced_cubes/norm/ --offsets_txt /cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt --slice 123 --output_file /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full/var_mosaic_slice123.fits --mask_percent 1.0 --plotting

"""