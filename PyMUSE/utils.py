"""Utilities for MuseCube"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools import utils as ltu
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from astropy import units as u


def plot_two_spec(sp1, sp2, text1=None, text2=None, renorm2=1.0):
    """Plot two spectra for comparison purposes"""

    plt.figure()
    plt.plot(sp1.wavelength, sp1.flux, 'k', drawstyle='steps-mid', label=text1)
    plt.plot(sp1.wavelength, sp1.sig, 'g', drawstyle='steps-mid')
    plt.plot(sp2.wavelength, renorm2 * sp2.flux, 'b', drawstyle='steps-mid', label=text2)
    plt.plot(sp2.wavelength, renorm2 * sp2.sig, 'y', drawstyle='steps-mid')
    plt.legend()
    # print stats
    print("<SN1> = {}".format(np.median(sp1.flux / sp1.sig)))
    print("<SN2> = {}".format(np.median(sp2.flux / sp2.sig)))
    print("<FL_IVAR1> = {}".format(np.median(sp1.flux / sp1.sig ** 2)))
    print("<FL_IVAR2> = {}".format(np.median(sp2.flux / sp2.sig ** 2)))
    print("<FL1>/<FL2> = {}".format(np.median(sp1.flux / sp2.flux)))

def indexOf(array,element):
    for i,j in enumerate(array):
        if j==element:
            return i
    return -1


def spec_to_redmonster_format(spec, fitsname, n_id=None, mag=None):
    """
    Function used to create a spectrum in the REDMONSTER software format
    :param spec: XSpectrum1D object
    :param mag: List containing 2 elements, the first is the keyword in the header that will contain the magnitud saved in the second element
    :param fitsname:  Name of the fitsfile that will be created
    :return:
    """
    from scipy import interpolate
    wave = spec.wavelength.value
    wave_log = np.log10(wave)
    n = len(wave)
    spec.wavelength = wave_log * u.angstrom
    new_wave_log = np.arange(wave_log[1], wave_log[n - 2], 0.0001)
    spec_rebined = spec.rebin(new_wv=new_wave_log * u.angstrom)
    flux = spec_rebined.flux.value
    f = interpolate.interp1d(wave_log, spec.sig.value)
    sig = f(new_wave_log)
    inv_sig = 1. / np.array(sig) ** 2
    inv_sig = np.where(np.isinf(inv_sig), 0, inv_sig)
    inv_sig = np.where(np.isnan(inv_sig), 0, inv_sig)
    hdu1 = fits.PrimaryHDU([flux])
    hdu2 = fits.ImageHDU([inv_sig])
    hdu1.header['COEFF0'] = new_wave_log[0]
    hdu1.header['COEFF1'] = new_wave_log[1] - new_wave_log[0]
    if n_id != None:
        hdu1.header['ID'] = n_id
    if mag != None:
        hdu1.header[mag[0]] = mag[1]
    hdulist_new = fits.HDUList([hdu1, hdu2])
    hdulist_new.writeto(fitsname, clobber=True)

def get_template(redmonster_file, n_template):  # n_template puede ser 1,2 o 3 o 4 o 5
    hdulist = fits.open(redmonster_file)
    templates = hdulist[2]
    table = hdulist[1].data
    Z = 'Z' + str(n_template)
    Z_ERR = 'Z_ERR' + str(n_template)
    MINRCHI2 = 'MINRCHI2' + str(n_template)
    CLASS = 'CLASS' + str(n_template)
    z_template = table[Z][0]
    z_err_template = table[Z_ERR][0]
    class_template = table[CLASS][0]
    minchi_template = table[MINRCHI2][0]
    index_template = n_template - 1
    template = hdulist[2].data[0][index_template]
    n = len(template)
    COEFF0 = hdulist[0].header['COEFF0']
    COEFF1 = hdulist[0].header['COEFF1']
    NAXIS1 = n
    WAVE_END = COEFF0 + (NAXIS1 - 1) * COEFF1
    wave_log = np.linspace(COEFF0, WAVE_END, n)
    wave = 10 ** wave_log
    return wave, template, z_template, class_template, minchi_template, z_err_template


def get_spec(specfit):
    hdulist = fits.open(specfit)
    flux = hdulist[0].data[0]
    er = hdulist[1].data[0]
    n = len(flux)
    COEFF0 = hdulist[0].header['COEFF0']
    COEFF1 = hdulist[0].header['COEFF1']
    NAXIS1 = hdulist[0].header['NAXIS1']
    WAVE_END = COEFF0 + (NAXIS1 - 1) * COEFF1
    wave_log = np.linspace(COEFF0, WAVE_END, n)
    wave = 10 ** wave_log
    return wave, flux, er


def get_rm_spec(rm_spec_name, rm_out_file=None, rm_fit_number=1):
    """
    Function to plot and examine a spectrum which is in a readable format to redmonster software.
    Optionally, the user can inster a redmonster output file and compare the input spectrum with the
    template contained in the fit given by rm_fit_number
    :param rm_spec_name: Name of the fits file which contain the spectrum in redmonster format
    :param rm_out_file: Optional. Default = None. Output file from redmonster
    :param rm_fit_number: Optional. Default = None. Number of the fit in rm_out_file (1 to 5)
    :return:
    """
    plt.figure()
    w_spec, f_spec, er_spec = get_spec(rm_spec_name)
    plt.plot(w_spec, f_spec, color='Blue', label=rm_spec_name)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux (arbitrary units)')
    mean = np.nanmean(f_spec)
    med = np.nanmedian(f_spec)
    std = np.nanstd(f_spec)
    ymin = mean - 3 * std
    ymax = mean + 4 * std
    plt.ylim(ymin, ymax)
    plt.xlim(min(w_spec), max(w_spec))
    if rm_out_file != None and rm_fit_number != None:
        w_temp, flux_temp, z_temp, class_temp, minchi_temp, z_err_temp = get_template(rm_out_file, rm_fit_number)
        label_str = 'Templ. Class = {}, z={:.5f} +- {:.5f}\nchisqr_dof={:.3f}'.format(class_temp, z_temp, z_err_temp,
                                                                                      minchi_temp)
        plt.plot(w_temp, flux_temp, color='Red', label=label_str)
        plt.legend(fontsize=9, bbox_to_anchor=(1.05, -0.13), loc=1, borderaxespad=0.)
        plt.show()
        return w_spec, f_spec, w_temp, flux_temp
    plt.legend()
    plt.show()
    return w_spec, f_spec, er_spec


def calculate_empirical_rms(spec, test=False):
    fl = spec.flux.value
    wv = spec.wavelength.value
    min_cond = ltu.is_local_minima(fl)
    max_cond = ltu.is_local_maxima(fl)
    min_local_inds = np.where(min_cond)[0]
    max_local_inds = np.where(max_cond)[0]

    interpolated_max = interp1d(wv[max_local_inds], fl[max_local_inds], kind='linear', bounds_error=False, fill_value=0)
    interpolated_min = interp1d(wv[min_local_inds], fl[min_local_inds], kind='linear', bounds_error=False, fill_value=0)
    # these are the envelopes
    fl_max = interpolated_max(wv)
    fl_min = interpolated_min(wv)
    # take the mid value
    fl_mid = 0.5 * (fl_max + fl_min) # reference flux

    # the idea here is that these will be the intrinsic rms per pixel (both are the same though)
    max_mean_diff = np.abs(fl_mid - fl_max)
    min_mean_diff = np.abs(fl_mid - fl_min)
    sigma = 0.5 * (max_mean_diff + min_mean_diff)  # anyways these two differences are the same by definition

    if test:
        # fluxes
        wv_mins = wv[min_local_inds]
        wv_maxs = wv[max_local_inds]
        plt.figure()
        plt.plot(wv, fl, drawstyle='steps-mid')
        plt.plot(wv_mins, fl[min_local_inds], marker='o', color='r', label='Local minimum')
        plt.plot(wv_maxs, fl[max_local_inds], marker='o', color='green', label='Local maximum')
        plt.plot(wv, fl_mid, color='black', label='flux_mid')

        # sigmas
        plt.plot(wv, sigma, 'o', color='pink', label='Empirical sigma')
        plt.plot(wv, spec.sig.value, color='yellow', label='Original sigma')
        plt.legend()
        plt.show()
    return XSpectrum1D.from_tuple((wv, fl, sigma))
