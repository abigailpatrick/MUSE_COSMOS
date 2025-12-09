#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import inspect
from glob import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for batch jobs
import matplotlib.pyplot as plt

from astropy.io import fits
from photutils.segmentation import SegmentationImage

import muse_origin
from muse_origin import ORIGIN
from mpdaf.obj import Cube

# FSFModel is provided by MPDAF, not muse_origin, in many environments
FSFModel = None
try:
    from mpdaf.muse import FSFModel as _FSFModel
    FSFModel = _FSFModel
except Exception:
    try:
        from mpdaf.MUSE import FSFModel as _FSFModel  # older MPDAF capitalization
        FSFModel = _FSFModel
    except Exception:
        FSFModel = None

# Catalog and Source imports (tolerate missing modules)
try:
    from mpdaf.sdetect import Catalog
except Exception:
    Catalog = None

try:
    from mpdaf.sdetect.source import Source
except Exception:
    Source = None


def open_source(path):
    if Source is None:
        raise RuntimeError("MPDAF Source class not available")
    # Try common constructors across MPDAF versions
    try:
        return Source.from_file(path)
    except AttributeError:
        pass
    except TypeError:
        pass
    try:
        return Source(filename=path)
    except Exception:
        with fits.open(path, memmap=False) as hdul:
            return Source(hdulist=hdul)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run ORIGIN pipeline on a specific source subcube.")
    parser.add_argument(
        "--source-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)),
        help="Source index (1..27). Defaults to SLURM_ARRAY_TASK_ID if present.",
    )
    args = parser.parse_args()
    SRC_ID = args.source_id

    # Build paths and names
    DATACUBE = f"/cephfs/apatrick/musecosmos/dataproducts/extractions/source_{SRC_ID}_subcube_20.0_3681.fits"
    NAME = f"source_{SRC_ID}"

    # Per-source base output directory to avoid collisions
    BASE_OUT = os.path.join("/home/apatrick/ORIGIN", NAME)
    os.makedirs(BASE_OUT, exist_ok=True)

    # Informative prints
    print(os.path.dirname(inspect.getfile(muse_origin)))
    print(muse_origin.__version__)
    print(f"Running ORIGIN on: {DATACUBE}")
    print(f"Project NAME: {NAME}")
    print(f"Base output directory: {BASE_OUT}")

    # Inspect cube
    Cube(DATACUBE).info()

    # Load cube
    cube = Cube(DATACUBE)
    # scale variance down to boost S/N 
    cube._var = np.nan_to_num(cube._var, nan=1e10, posinf=1e10, neginf=1e10)
    cube._var *= 0.6
    cube.write('/tmp/cleaned_varscaled_cube.fits', savemask=False)
    DATACUBE = '/tmp/cleaned_varscaled_cube.fits'

    # Clean NaNs / Infs to safe values
    cube._data = np.nan_to_num(cube._data, nan=0.0, posinf=0.0, neginf=0.0)
    cube._var = np.nan_to_num(cube._var, nan=1e10, posinf=1e10, neginf=1e10)

    # Write to a unique temp cleaned cube (avoid collisions in job arrays)
    tmp_clean = f"/tmp/cleaned_cube_{NAME}.fits"
    cube.write(tmp_clean, savemask=False)
    DATACUBE = tmp_clean
    print(f"Cleaned cube written to: {DATACUBE}")

    # FSF model (optional; depends on MPDAF availability)
    if FSFModel is not None:
        try:
            fsfmodel = FSFModel.read(DATACUBE)
            hdr = fsfmodel.to_header()
            print("FSFModel header:")
            print(hdr)
        except Exception as e:
            print(f"FSFModel present but failed to read/print header: {e}")
    else:
        print("FSFModel not available in this environment; skipping FSF header print.")

    # Initialize ORIGIN
    orig = ORIGIN.init(DATACUBE, name=NAME, loglevel="DEBUG", logcolor=True)
    orig.set_loglevel("INFO")

    # Step 01: preprocessing and basic images
    orig.step01_preprocessing()

    # Quick visualization: white, cont (dct), std
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(21, 4))
    images = [orig.ima_white, orig.ima_dct, orig.ima_std]
    titles = ["white image", "cont image (dct)", "std image: (raw-dct)/std"]
    for ax, im, title in zip(axes.flat, images, titles):
        im.plot(ax=ax, title=title, zscale=True)
    out_path = os.path.join(BASE_OUT, "images.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Segmentation maps
    seg_fig = orig.plot_segmaps(figsize=(7, 4))
    fig_segmaps = seg_fig if hasattr(seg_fig, "savefig") else plt.gcf()
    out_path = os.path.join(BASE_OUT, "segmaps.png")
    fig_segmaps.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig_segmaps)
    print(f"Segmaps figure saved to {out_path}")

    # Persist
    orig.write()

    # Reload and status
    orig = ORIGIN.load(NAME)
    orig.status()

    # Step 02: areas
    orig.step02_areas(minsize=50, maxsize=None)  # Note: test changing this

    # Step 03–04: PCA thresholds and greedy PCA
    orig.step03_compute_PCA_threshold(pfa_test=0.01)
    orig.step04_compute_greedy_PCA()

    # Persist
    orig.write()

    # Reload and status
    orig = ORIGIN.load(NAME)
    orig.status()

    # PCA plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    orig.plot_mapPCA(ax=ax1)
    ima_faint = orig.cube_faint.max(axis=0)
    ima_faint.plot(ax=ax2, colorbar="v", zscale=True, title="White image for cube_faint")
    out_path = os.path.join(BASE_OUT, "PCA.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Step 05: TGLR
    pcut = 1e-08
    orig.step05_compute_TGLR(ncpu=1, pcut=pcut)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    orig.maxmap.plot(ax=ax1, title="maxmap", cmap="Spectral_r")
    (-1 * orig.minmap).plot(ax=ax2, title="minmap", cmap="Spectral_r")
    out_path = os.path.join(BASE_OUT, "TGLR.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Persist
    orig.write()

    # Step 06: purity thresholds
    orig.step06_compute_purity_threshold(purity=0.6, purity_std=0.7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    orig.plot_purity(ax=ax1)
    orig.plot_purity(ax=ax2, comp=True)
    out_path = os.path.join(BASE_OUT, "purity.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True, sharex=True)
    orig.plot_min_max_hist(ax=ax1)
    orig.plot_min_max_hist(ax=ax2, comp=True)
    out_path = os.path.join(BASE_OUT, "hist.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Show thresholds (optional side effects)
    _ = (orig.threshold_correl, orig.threshold_std)

    # Persist
    orig.write()

    # Reload and status
    orig = ORIGIN.load(NAME)
    orig.status()

    # Convert MPDAF Image to numpy array for segmentation, then to Photutils segmap
    segmap_array = orig.segmap_merged.data
    segmap = SegmentationImage(segmap_array)
    orig.segmap_merged = segmap
    print("Segmap type before deblending:", type(orig.segmap_merged))

    # Step 07: detection
    orig.step07_detection()

    # Wrap catalogs if Catalog is available
    if Catalog is not None:
        cat0 = Catalog(orig.Cat0)
        cat1 = Catalog(orig.Cat1)
        print(type(cat0))
        print(type(cat1))
    else:
        cat0, cat1 = orig.Cat0, orig.Cat1
        print("Catalog class not available; using raw tables.")

    # Save catalogs (avoid collisions)
    out_cat0 = os.path.join(BASE_OUT, "Cat0.fits")
    out_cat1 = os.path.join(BASE_OUT, "Cat1.fits")
    orig.Cat0.write(out_cat0, overwrite=True)
    orig.Cat1.write(out_cat1, overwrite=True)

    # Segmap overlay figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    orig.maxmap.plot(ax=ax1, zscale=True, colorbar="v", cmap="gray", scale="asinh", title="maxmap")
    orig.ima_white.plot(ax=ax2, zscale=True, colorbar="v", cmap="gray", scale="asinh", title="white image")
    # Plot detections (only Cat1 as in your snippet)
    try:
        if Catalog is not None and len(cat1) > 0:
            cat1.plot_symb(ax1, orig.maxmap.wcs, ecol="r", esize=1.0, ra="ra", dec="dec")
            cat1.plot_symb(ax2, orig.maxmap.wcs, ecol="r", esize=1.0, ra="ra", dec="dec")
    except Exception as e:
        print(f"Plotting catalog symbols failed: {e}")

    out_path = os.path.join(BASE_OUT, "segmap7.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    # Steps 08–10: spectra, clean, masks
    orig.step08_compute_spectra()
    orig.step09_clean_results()
    orig.step10_create_masks()
    orig.write()

    # Reload and status
    orig = ORIGIN.load(NAME)
    orig.status()

    # Catalog 3 info and save
    print(f"Catalog 3 Sources: {len(orig.Cat3_sources)}")
    print(f"Catalog 3 Lines: {len(orig.Cat3_lines)}")
    print("Catalog 3 Sources:")
    print("Cat3_sources (all columns):")
    orig.Cat3_sources['ra'].info.format = '%.8f'
    orig.Cat3_sources['dec'].info.format = '%.8f'
    orig.Cat3_sources.pprint_all()

    print("Cat3_lines (all columns):")
    orig.Cat3_lines.pprint_all()


    out_lines = os.path.join(BASE_OUT, "Cat3_lines.fits")
    out_sources = os.path.join(BASE_OUT, "Cat3_sources.fits")
    orig.Cat3_lines.write(out_lines, overwrite=True)
    orig.Cat3_sources.write(out_sources, overwrite=True)

    # Reload and status
    orig = ORIGIN.load(NAME)
    orig.status()

    # Step 11: save sources
    orig.step11_save_sources("0.1", n_jobs=1)

    # Finalize
    orig.write()
    orig.stat()
    orig.timestat()

    # =========================
    # Simple per-source quicklook (one figure per source)
    # Uses the REFSPEC line for images and spectrum
    # =========================
    if Source is None:
        print("[Quicklook] MPDAF Source class not available; skipping quicklook generation.")
        return

    src_dir = os.path.join(orig.outpath, "sources")
    out_dir = os.path.join(orig.outpath, "quicklook")
    os.makedirs(out_dir, exist_ok=True)

    src_files = sorted(glob(os.path.join(src_dir, "source-*.fits")))
    print(f"[Quicklook] Found {len(src_files)} sources in {src_dir}")

    for sf in src_files:
        try:
            src = open_source(sf)
        except Exception as e:
            print(f"[Quicklook] Could not open {sf}: {e}")
            continue

        base = os.path.basename(sf)
        m = re.search(r"source-(\d+)\.fits$", base)
        sid = int(m.group(1)) if m else -1

        refspec = src.header.get("REFSPEC", None)

        ref_num = None
        if isinstance(refspec, str):
            m = re.search(r"ORI_CORR_(\d+)", refspec)
            if m:
                ref_num = int(m.group(1))

        lbda = None
        if getattr(src, "lines", None) is not None and len(src.lines) > 0 and ref_num is not None:
            rows = src.lines[src.lines["NUM_LINE"] == ref_num]
            if len(rows) > 0:
                row = rows[0]
                if "LBDA_OBS" in row.colnames:
                    lbda = float(row["LBDA_OBS"])
                elif "LBDA" in row.colnames:
                    lbda = float(row["LBDA"])

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        ax_im1, ax_im2 = axs[0]
        ax_sp = axs[1, 0]
        axs[1, 1].axis("off")

        tag_corr = f"ORI_CORR_{ref_num}" if ref_num is not None else None
        if tag_corr and tag_corr in src.images:
            try:
                src.images[tag_corr].plot(ax=ax_im1, title=tag_corr, zscale=True, colorbar="v")
            except Exception:
                src.images[tag_corr].plot(ax=ax_im1, title=tag_corr)
        else:
            ax_im1.text(0.5, 0.5, f"{tag_corr or 'ORI_CORR_#'} not found", ha="center", va="center")
            ax_im1.set_title("ORI_CORR")

        tag_nb = f"NB_LINE_{ref_num}" if ref_num is not None else None
        if tag_nb and tag_nb in src.images:
            try:
                src.images[tag_nb].plot(ax=ax_im2, title=tag_nb, zscale=True, colorbar="v")
            except Exception:
                src.images[tag_nb].plot(ax=ax_im2, title=tag_nb)
        else:
            ax_im2.text(0.5, 0.5, f"{tag_nb or 'NB_LINE_#'} not found", ha="center", va="center")
            ax_im2.set_title("NB_LINE")

        plotted = False
        if isinstance(refspec, str) and refspec in src.spectra:
            src.spectra[refspec].plot(ax=ax_sp, color="k", label=refspec)
            plotted = True
        elif "DATA_SKYSUB" in src.spectra:
            src.spectra["DATA_SKYSUB"].plot(ax=ax_sp, color="k", label="DATA_SKYSUB")
            plotted = True
        elif "DATA" in src.spectra:
            src.spectra["DATA"].plot(ax=ax_sp, color="gray", label="DATA")
            plotted = True

        if plotted:
            if lbda is not None:
                ax_sp.axvline(lbda, color="r", ls="--", alpha=0.7)
                try:
                    w = (
                        src.spectra[refspec].wave.coord()
                        if isinstance(refspec, str) and refspec in src.spectra
                        else src.spectra["DATA_SKYSUB"].wave.coord()
                        if "DATA_SKYSUB" in src.spectra
                        else src.spectra["DATA"].wave.coord()
                    )
                    pad = 100.0
                    ax_sp.set_xlim(max(w.min(), lbda - pad), min(w.max(), lbda + pad))
                except Exception:
                    pass
            ax_sp.set_title(f"Spectrum ({refspec or 'DATA_SKYSUB/DATA'})")
            ax_sp.set_xlabel("Wavelength (Å)")
            ax_sp.legend(loc="best", fontsize=8)
        else:
            ax_sp.text(0.5, 0.5, "No spectrum found", ha="center", va="center")
            ax_sp.set_title("Spectrum")

        fig.suptitle(f"Source {sid:05d}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = os.path.join(out_dir, f"source-{sid:05d}.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[Quicklook] Wrote {out_png}")

    print(f"[Quicklook] Done. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
