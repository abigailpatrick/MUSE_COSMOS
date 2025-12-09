import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Image, Cube
from mpdaf.sdetect import Catalog

import muse_origin
from muse_origin import ORIGIN
import photutils
from mpdaf.MUSE import FSFModel


import numpy as np
import types
from photutils.segmentation import SegmentationImage
import muse_origin.source_creation as sc

import muse_origin
import inspect
import os
import photutils


print(os.path.dirname(inspect.getfile(muse_origin)))


print(muse_origin.__version__)

#DATACUBE = "/home/apatrick/MUSE/DATACUBE_UDF-MOSAIC.fits"
DATACUBE =  "/cephfs/apatrick/musecosmos/dataproducts/extractions/source_7_subcube_20.0_3681.fits"

NAME = 'source_13'
#'UDF10-example'

Cube(DATACUBE).info()

cube = Cube(DATACUBE)
# scale variance down to boost S/N 
cube._var = np.nan_to_num(cube._var, nan=1e10, posinf=1e10, neginf=1e10)
cube._var *= 0.6
cube.write('/tmp/cleaned_varscaled_cube.fits', savemask=False)
DATACUBE = '/tmp/cleaned_varscaled_cube.fits'

cube._data = np.nan_to_num(cube._data, nan=0.0, posinf=0.0, neginf=0.0)
cube._var  = np.nan_to_num(cube._var,  nan=1e10, posinf=1e10, neginf=1e10)

cube.write('/tmp/cleaned_cube.fits', savemask=False)

DATACUBE = "/tmp/cleaned_cube.fits"


fsfmodel = FSFModel.read(DATACUBE)
fsfmodel.to_header()
# print the new header
print(fsfmodel.to_header())

orig = ORIGIN.init(DATACUBE, name=NAME, loglevel='DEBUG', logcolor=True)


orig.set_loglevel('INFO')


orig.step01_preprocessing()

orig.ima_white, orig.ima_dct, orig.ima_std


fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(21, 4))
images = [orig.ima_white, orig.ima_dct, orig.ima_std]
titles = ['white image', 'cont image (dct)', 'std image: (raw-dct)/std']
for ax, im, title in zip(axes.flat, images, titles):
    im.plot(ax=ax, title=title, zscale=True)

fig.savefig("/home/apatrick/ORIGIN/images.png", dpi=300, bbox_inches="tight")
print(f"Figure saved to /home/apatrick/ORIGIN/images.png")

# Plot and save the segmentation maps
seg_fig = orig.plot_segmaps(figsize=(7, 4))

# plot_segmaps may return a Figure or nothing; normalize to a Figure handle
if hasattr(seg_fig, "savefig"):
    fig_segmaps = seg_fig
else:
    fig_segmaps = plt.gcf()

fig_segmaps.savefig("/home/apatrick/ORIGIN/segmaps.png", dpi=300, bbox_inches="tight")
print("Segmaps figure saved to /home/apatrick/ORIGIN/segmaps.png")
plt.close(fig_segmaps)  # option
orig.write()



orig = ORIGIN.load(NAME)
orig.status()



orig.step02_areas(minsize=80, maxsize=120)  ## where is 80 -120 coming from?


orig.step03_compute_PCA_threshold(pfa_test=0.01)
orig.step04_compute_greedy_PCA()

orig.write()



orig = ORIGIN.load(NAME)
orig.status()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
orig.plot_mapPCA(ax=ax1)
ima_faint = orig.cube_faint.max(axis=0)
ima_faint.plot(ax=ax2, colorbar='v',zscale=True, title='White image for cube_faint')

fig.savefig("/home/apatrick/ORIGIN/PCA.png", dpi=300, bbox_inches="tight")
print(f"Figure saved to /home/apatrick/ORIGIN/PCA.png")

pcut = 1e-8

orig.step05_compute_TGLR(ncpu=1, pcut=pcut)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14, 4))
orig.maxmap.plot(ax=ax1, title='maxmap', cmap='Spectral_r')
(-1*orig.minmap).plot(ax=ax2, title='minmap', cmap='Spectral_r')

fig.savefig("/home/apatrick/ORIGIN/TGLR.png", dpi=300, bbox_inches="tight")
print(f"Figure saved to /home/apatrick/ORIGIN/TGLR.png")


orig.write()


orig.step06_compute_purity_threshold(purity=0.8, purity_std=0.95)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14, 4))
orig.plot_purity(ax=ax1)
orig.plot_purity(ax=ax2, comp=True)

fig.savefig("/home/apatrick/ORIGIN/purity.png", dpi=300, bbox_inches="tight")
print(f"Figure saved to /home/apatrick/ORIGIN/purity.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True, sharex=True)
orig.plot_min_max_hist(ax=ax1)
orig.plot_min_max_hist(ax=ax2, comp=True)

fig.savefig("/home/apatrick/ORIGIN/hist.png", dpi=300, bbox_inches="tight")
print(f"Figure saved to /home/apatrick/ORIGIN/hist.png")

orig.threshold_correl, orig.threshold_std

orig.write()



orig = ORIGIN.load(NAME)
orig.status()



# Convert MPDAF Image to a NumPy array
segmap_array = orig.segmap_merged.data  # Extract data from MPDAF Image

# Convert to SegmentationImage
segmap = SegmentationImage(segmap_array)

# Assign the converted segmap back to orig
orig.segmap_merged = segmap  # Explicitly assign it

# Check type again
print("Segmap type before deblending:", type(orig.segmap_merged))



orig.step07_detection()

cat0 = Catalog(orig.Cat0)
cat1 = Catalog(orig.Cat1)
print(type(cat0))
print(type(cat1))

orig.Cat0.write('/home/apatrick/ORIGIN/Cat0.fits', overwrite=True)
orig.Cat1.write('/home/apatrick/ORIGIN/Cat1.fits', overwrite=True)

#cat0 = Catalog(orig.Cat1[orig.Cat1['comp']==0])
#cat1 = Catalog(orig.Cat1[orig.Cat1['comp']==1])

fig,(ax1, ax2) = plt.subplots(1,2,figsize=(20,10))

orig.maxmap.plot(ax=ax1, zscale=True, colorbar='v', cmap='gray', scale='asinh', title='maxmap')
orig.ima_white.plot(ax=ax2, zscale=True, colorbar='v', cmap='gray', scale='asinh', title='white image')

for ax in (ax1, ax2):
    #cat0.plot_symb(ax, orig.maxmap.wcs, ecol='g', esize=1.0, ra='ra', dec='dec')
    if len(cat1) > 0:
        cat1.plot_symb(ax, orig.maxmap.wcs, ecol='r', esize=1.0, ra='ra', dec='dec')
     
fig.savefig("/home/apatrick/ORIGIN/segmap7.png", dpi=300, bbox_inches="tight")
print(f"Figure saved to /home/apatrick/ORIGIN/segmap7.png")


orig.step08_compute_spectra()
orig.step09_clean_results() 
orig.step10_create_masks()
orig.write()




orig = ORIGIN.load(NAME)
orig.status()
orig.Cat3_sources
orig.Cat3_lines
print(f"Catalog 3 Sources: {len(orig.Cat3_sources)}")
print(f"Catalog 3 Lines: {len(orig.Cat3_lines)}")
print("Catalog 3 Sources:")
orig.Cat3_sources['ra'].info.format = '%.8f'
orig.Cat3_sources['dec'].info.format = '%.8f'
orig.Cat3_sources.pprint(max_lines=-1)
orig.Cat3_sources.pprint_all()
orig.Cat3_lines.pprint(max_lines=-1)
orig.Cat3_lines.write('/home/apatrick/ORIGIN/Cat3_lines.fits', overwrite=True)
orig.Cat3_sources.write('/home/apatrick/ORIGIN/Cat3_sources.fits', overwrite=True)



orig = ORIGIN.load(NAME)
orig.status()


orig.step11_save_sources('0.1', n_jobs=1)

orig.write()
orig.stat()
orig.timestat()

# =========================
# Simple per-source quicklook (one figure per source)
# =========================
import os
import re
from glob import glob
import matplotlib.pyplot as plt
from mpdaf.sdetect.source import Source
from astropy.io import fits

def open_source(path):
    # Try the common constructors across MPDAF versions
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

    refspec = src.header.get('REFSPEC', None)

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
            src.images[tag_corr].plot(ax=ax_im1, title=tag_corr, zscale=True, colorbar='v')
        except Exception:
            src.images[tag_corr].plot(ax=ax_im1, title=tag_corr)
    else:
        ax_im1.text(0.5, 0.5, f"{tag_corr or 'ORI_CORR_#'} not found", ha="center", va="center")
        ax_im1.set_title("ORI_CORR")

    tag_nb = f"NB_LINE_{ref_num}" if ref_num is not None else None
    if tag_nb and tag_nb in src.images:
        try:
            src.images[tag_nb].plot(ax=ax_im2, title=tag_nb, zscale=True, colorbar='v')
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
                w = (src.spectra[refspec].wave.coord()
                     if isinstance(refspec, str) and refspec in src.spectra
                     else src.spectra["DATA_SKYSUB"].wave.coord()
                     if "DATA_SKYSUB" in src.spectra
                     else src.spectra["DATA"].wave.coord())
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
