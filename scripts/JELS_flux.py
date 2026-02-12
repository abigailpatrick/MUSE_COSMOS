"""
Open both the photomtery files and the sed files
combine the df's into one 
limit to just the sources I want -remove the agn ? (my brightest source noooooo)
calculate and add in fluxes and luminosities uncorrected
Aperture correct 
Dust correct

"""
#!/usr/bin/env python3
import numpy as np
from astropy.table import Table, Column
from astropy.cosmology import Planck18
import astropy.units as u

# --- File paths ---
input_fits  = "/home/apatrick/P1/JELSDP/combined_selected_sources.fits"
output_fits = "/home/apatrick/P1/JELSDP/combined_selected_sources_with_LHa.fits"

# ----------------------------
# Utilities for safe extraction
# ----------------------------
def get_col_as_array(t, name, default=np.nan):
    n = len(t)
    if name in t.colnames:
        data = t[name]
        arr = np.array(data, dtype=float)
        if getattr(data, 'mask', None) is not None:
            mask = np.array(data.mask, dtype=bool)
            arr[mask] = np.nan
        return arr
    else:
        return np.full(n, default, dtype=float)

def first_finite(*arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    out = np.full_like(arrays[0], np.nan, dtype=float)
    for a in arrays:
        need = ~np.isfinite(out)
        take = need & np.isfinite(a)
        out[take] = a[take]
    return out

# ----------------------------
# Cosmology and redshift
# ----------------------------
def choose_redshift(t):
    z_spec    = get_col_as_array(t, "z_spec")
    z1_median = get_col_as_array(t, "z1_median")
    return z1_median # change if I want to try zspec

def luminosity_distance_cm(z):
    dL = Planck18.luminosity_distance(np.asarray(z, dtype=float))  # Mpc
    return dL.to(u.cm).value

# ----------------------------
# H-alpha flux selection
# ----------------------------
def select_halpha_flux_and_error(t):
    src = np.array(t['SOURCE_CAT'], dtype='U8')

    f = np.full(len(t), np.nan)
    e = np.full(len(t), np.nan)
    fsrc = np.full(len(t), "", dtype='U32')

    m466 = src == "F466N"
    m470 = src == "F470N"

    if np.any(m466):
        f[m466] = get_col_as_array(t, "F_line_F444W_F466N")[m466]
        e[m466] = get_col_as_array(t, "F_line_err_F444W_F466N")[m466]
        fsrc[m466] = "F_line_F444W_F466N"

    if np.any(m470):
        f[m470] = get_col_as_array(t, "F_line_F444W_F470N")[m470]
        e[m470] = get_col_as_array(t, "F_line_err_F444W_F470N")[m470]
        fsrc[m470] = "F_line_F444W_F470N"

    bad = ~np.isfinite(f) | (f <= 0)
    fsrc[bad] = "none"

    return f, e, fsrc


# ----------------------------
# Aperture correction factor
# ----------------------------
def compute_aperture_correction_factor(t):
    src = np.array(t['SOURCE_CAT'], dtype='U8')

    ratio = np.ones(len(t), dtype=float)
    rsrc  = np.full(len(t), "none", dtype='U8')

    m466 = src == "F466N"
    m470 = src == "F470N"

    if np.any(m466):
        a = get_col_as_array(t, "F466N_auto_flux")[m466]
        r = get_col_as_array(t, "F466N_600mas_flux")[m466]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio[m466] = a / r
        rsrc[m466] = "F466N"

    if np.any(m470):
        a = get_col_as_array(t, "F470N_auto_flux")[m470]
        r = get_col_as_array(t, "F470N_600mas_flux")[m470]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio[m470] = a / r
        rsrc[m470] = "F470N"

    bad = ~np.isfinite(ratio) | (ratio <= 0)
    ratio[bad] = 1.0
    rsrc[bad]  = "none"

    return ratio, rsrc


# ----------------------------
# Calzetti attenuation
# ----------------------------
def calzetti_k_lambda(wavelength_angstrom, R_V=4.05):
    lam_um = wavelength_angstrom / 1e4
    if lam_um <= 0.63:
        k = 2.659 * (-2.156 + 1.509/lam_um - 0.198/lam_um**2 + 0.011/lam_um**3) + R_V
    else:
        k = 2.659 * (-1.857 + 1.040/lam_um) + R_V
    return k

def dust_correction_coeffs(wavelength_angstrom=6562.81, nebular_to_cont=2.27, R_V=4.05):
    k = calzetti_k_lambda(wavelength_angstrom, R_V=R_V)
    a_cont = (k / R_V)
    a_line = a_cont * nebular_to_cont
    return a_cont, a_line

# ----------------------------
# Luminosity calculations
# ----------------------------
def compute_uncorrected_luminosity(f_flux, dL_cm):
    return 4.0 * np.pi * (dL_cm**2) * f_flux

def propagate_uncorr_L_error(f_err, dL_cm):
    return 4.0 * np.pi * (dL_cm**2) * f_err

def apply_aperture_correction(L_uncorr, L_uncorr_err, ratio):
    L_ap = L_uncorr * ratio
    L_ap_err = L_uncorr_err * ratio
    return L_ap, L_ap_err

def apply_dust_correction(L, L_err, Av, Av_err, a_coeff):
    factor = 10.0**(0.4 * a_coeff * Av)
    L_corr = L * factor
    term_flux = (factor * L_err)**2
    term_av   = (L * np.log(10.0) * 0.4 * a_coeff * factor * Av_err)**2
    L_corr_err = np.sqrt(term_flux + term_av)
    return L_corr, L_corr_err

# ----------------------------
# FITS helpers
# ----------------------------
def _is_string_like(x):
    return isinstance(x, (str, bytes, np.str_, np.bytes_))

def _stringify_cell(x):
    """Convert one cell to a plain string: join lists/arrays of strings, else str(x)."""
    if x is None:
        return ""
    if _is_string_like(x):
        return str(x)
    # Lists/arrays
    if isinstance(x, (list, tuple, np.ndarray)):
        # If all elements are string-like, join by space
        try:
            if all(_is_string_like(xx) for xx in x):
                return " ".join(str(xx) for xx in x)
        except Exception:
            pass
        # Otherwise, just stringify the container
        return str(x)
    # Fallback
    return str(x)

def sanitize_table_for_fits(t):
    """
    Return a copy of table with:
      - object dtype columns converted to strings
      - any list/array-of-strings cells flattened to a single string
      - unicode converted to bytes for FITS
    """
    t2 = t.copy(copy_data=True)
    converted_cols = []

    for name in list(t2.colnames):
        col = t2[name]
        # If this is a mixin column, skip unless it's object-like
        kind = getattr(getattr(col, 'dtype', None), 'kind', None)

        if kind == 'O':  # object dtype: convert to strings
            arr = np.array([_stringify_cell(v) for v in col], dtype='U256')
            t2.replace_column(name, Column(arr, name=name))
            converted_cols.append(name)
        else:
            # Also catch vector columns of strings (ndim > 1) – convert to 1D strings
            data = np.asarray(col)
            if data.ndim > 1 and data.dtype.kind in ('U', 'S'):
                flat = np.array([" ".join(map(str, row)) for row in data], dtype='U256')
                t2.replace_column(name, Column(flat, name=name))
                converted_cols.append(name)

    # Ensure all unicode string columns are bytes for FITS
    try:
        t2.convert_unicode_to_bytestring()
    except Exception:
        # Older astropy might not need/want this; safe to ignore
        pass

    if converted_cols:
        print("Sanitized columns for FITS (converted to simple strings):")
        print("  " + ", ".join(converted_cols))
    else:
        print("No FITS sanitation needed.")

    return t2

# ----------------------------
# Main workflow
# ----------------------------
def main():
    t = Table.read(input_fits)
    print(t.colnames)

    ids = t['ID'] if 'ID' in t.colnames else np.arange(len(t))

    # Redshift and distance
    z = choose_redshift(t)
    dL_cm = luminosity_distance_cm(z)

    # Hα flux selection
    f_flux, f_err, f_src = select_halpha_flux_and_error(t)

    # Uncorrected luminosity and error
    L_uncorr = compute_uncorrected_luminosity(f_flux, dL_cm)
    L_uncorr_err = propagate_uncorr_L_error(f_err, dL_cm)

    # Aperture correction
    ap_ratio, ap_src = compute_aperture_correction_factor(t)
    L_ap, L_ap_err = apply_aperture_correction(L_uncorr, L_uncorr_err, ap_ratio)

    # Dust correction inputs
    Av50 = get_col_as_array(t, "Av_50", default=0.0)
    Av16 = get_col_as_array(t, "Av_16", default=np.nan)
    Av84 = get_col_as_array(t, "Av_84", default=np.nan)
    Av_err = np.where(np.isfinite(Av16) & np.isfinite(Av84), 0.5*(Av84 - Av16), 0.0)

    # Dust correction coefficients at Hα
    a_cont, a_line = dust_correction_coeffs()

    # Apply dust to aperture-corrected luminosity
    L_ap_dust_cont, L_ap_dust_cont_err = apply_dust_correction(L_ap, L_ap_err, Av50, Av_err, a_cont)
    L_ap_dust_line, L_ap_dust_line_err = apply_dust_correction(L_ap, L_ap_err, Av50, Av_err, a_line)

    # Add columns to table (ensure new string columns are plain strings)
    def add(name, data):
        if isinstance(data, np.ndarray) and data.dtype.kind == 'U':
            col = Column(np.array(data, dtype='U256'), name=name)
        elif isinstance(data, np.ndarray) and data.dtype.kind == 'O':
            # force to unicode
            col = Column(np.array([str(x) for x in data], dtype='U256'), name=name)
        else:
            col = Column(data=data, name=name)
        if name not in t.colnames:
            t.add_column(col)
        else:
            t[name] = col

    add("z_used", z)
    add("DL_cm", dL_cm)

    add("Ha_flux", f_flux)
    add("Ha_flux_err", f_err)
    add("Ha_flux_source", np.array(f_src, dtype='U64'))

    add("L_Ha_uncorr", L_uncorr)
    add("L_Ha_uncorr_err", L_uncorr_err)

    add("apcorr_factor", ap_ratio)
    add("apcorr_source", np.array(ap_src, dtype='U16'))

    add("L_Ha_apcorr", L_ap)
    add("L_Ha_apcorr_err", L_ap_err)

    # For transparency, also add A_Ha (continuum and line)
    k_ha = calzetti_k_lambda(6562.81, R_V=4.05)
    A_Ha_cont = (k_ha / 4.05) * Av50
    A_Ha_line = A_Ha_cont * 2.27
    add("A_Ha_cont_mag", A_Ha_cont)
    add("A_Ha_line_mag", A_Ha_line)

    add("L_Ha_ap_dustcorr_cont", L_ap_dust_cont)
    add("L_Ha_ap_dustcorr_cont_err", L_ap_dust_cont_err)

    add("L_Ha_ap_dustcorr_line", L_ap_dust_line)
    add("L_Ha_ap_dustcorr_line_err", L_ap_dust_line_err)

    # Sanitize for FITS and write
    t_out = sanitize_table_for_fits(t)
    t_out.write(output_fits, overwrite=True)

    # Print concise summary
    print(f"\nRead:  {input_fits}")
    print(f"Wrote: {output_fits}\n")
    print("Per-source summary:")
    hdr = ("ID", "z", "Ha_flux", "L_uncorr", "apfac", "L_ap", "Av50", "L_dust_cont", "L_dust_line", "flux_src", "ap_src")
    print("{:>6} {:>7} {:>11} {:>12} {:>6} {:>12} {:>6} {:>12} {:>12} {:>18} {:>7}".format(*hdr))
    for i in range(len(t)):
        print("{:>6} {:7.3f} {:11.3e} {:12.3e} {:6.3f} {:12.3e} {:6.2f} {:12.3e} {:12.3e} {:>18} {:>7}".format(
            int(t['ID'][i]) if 'ID' in t.colnames and np.isfinite(t['ID'][i]) else -1,
            z[i] if np.isfinite(z[i]) else np.nan,
            f_flux[i] if np.isfinite(f_flux[i]) else np.nan,
            L_uncorr[i] if np.isfinite(L_uncorr[i]) else np.nan,
            ap_ratio[i] if np.isfinite(ap_ratio[i]) else 1.0,
            L_ap[i] if np.isfinite(L_ap[i]) else np.nan,
            Av50[i] if np.isfinite(Av50[i]) else 0.0,
            L_ap_dust_cont[i] if np.isfinite(L_ap_dust_cont[i]) else np.nan,
            L_ap_dust_line[i] if np.isfinite(L_ap_dust_line[i]) else np.nan,
            str(f_src[i]),
            str(ap_src[i]),
        ))

if __name__ == "__main__":
    main()
