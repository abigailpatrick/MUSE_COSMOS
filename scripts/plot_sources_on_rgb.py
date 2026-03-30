import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


def plot_sources_on_rgb(
    rgb_fits_path,
    csv_path,
    output_png,
    marker_size=60,
    text_offset=5
):
    # --------------------------------------------------
    # 1. Load RGB FITS
    # --------------------------------------------------
    with fits.open(rgb_fits_path) as hdul:
        data = hdul[0].data  # shape should be (3, ny, nx)
        header = hdul[0].header

    if data.shape[0] != 3:
        raise ValueError("RGB FITS must have shape (3, ny, nx)")

    # Convert to (ny, nx, 3) for imshow
    rgb = np.moveaxis(data, 0, -1)

    # --------------------------------------------------
    # 2. Load WCS (celestial only)
    # --------------------------------------------------
    wcs = WCS(header).celestial

    # --------------------------------------------------
    # 3. Load CSV
    # --------------------------------------------------
    df = pd.read_csv(csv_path)

    # Required columns
    required_cols = ["ra_used", "dec_used", "lya_detect_flag", "ID"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    # --------------------------------------------------
    # 4. Convert RA/Dec → pixel coordinates
    # --------------------------------------------------
    coords = SkyCoord(
        ra=df["ra_used"].values * u.deg,
        dec=df["dec_used"].values * u.deg,
        frame="icrs"
    )

    xpix, ypix = wcs.world_to_pixel(coords)

    # --------------------------------------------------
    # 5. Plot
    # --------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)

    ax.imshow(rgb, origin="lower")

    for i in range(len(df)):
        flag = df["lya_detect_flag"].iloc[i]
        sid = df["ID"].iloc[i]

        # Colour logic
        if flag in [1, 2]:
            color = "lime"
        else:
            color = "red"

        # Plot marker
        ax.scatter(
            xpix[i],
            ypix[i],
            s=marker_size,
            edgecolor=color,
            facecolor="none",
            linewidth=1.5
        )

        # Plot ID label
        ax.text(
            xpix[i] + text_offset,
            ypix[i] + text_offset,
            str(sid),
            color=color,
            fontsize=6,
            weight="bold"
        )

    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title("MUSE RGB with Lyα Sources")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved overlay image to {output_png}")


if __name__ == "__main__":
    rgb_fits = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_rgb.fits"
    csv_file = "/cephfs/apatrick/musecosmos/scripts/sample_select/outputs/lya_flux_ap0p6.csv"
    output_png = "/cephfs/apatrick/musecosmos/dataproducts/extractions/MEGA_CUBE_rgb_with_sources.png"

    plot_sources_on_rgb(rgb_fits, csv_file, output_png)