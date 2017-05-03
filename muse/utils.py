"""Utilities for MuseCube"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def plot_two_spec(sp1, sp2, text1=None, text2=None, renorm2=1.0):
    """Plot two spectra for comparison purposes"""

    plt.figure()
    plt.plot(sp1.wavelength, sp1.flux, 'k', drawstyle='steps-mid', label=text1)
    plt.plot(sp1.wavelength, sp1.sig, 'g', drawstyle='steps-mid')
    plt.plot(sp2.wavelength, renorm2*sp2.flux, 'b', drawstyle='steps-mid', label=text2)
    plt.plot(sp2.wavelength, renorm2*sp2.sig, 'y', drawstyle='steps-mid')
    plt.legend()
    # print stats
    print("<SN1> = {}".format(np.median(sp1.flux/sp1.sig)))
    print("<SN2> = {}".format(np.median(sp2.flux/sp2.sig)))
    print("<FL_IVAR1> = {}".format(np.median(sp1.flux/sp1.sig**2)))
    print("<FL_IVAR2> = {}".format(np.median(sp2.flux/sp2.sig**2)))
    print("<FL1>/<FL2> = {}".format(np.median(sp1.flux/sp2.flux)))


def get_template(redmonster_file,n_template):  #n_template puede ser 1,2 o 3 o 4 o 5
    hdulist=fits.open(redmonster_file)
    templates=hdulist[2]
    table=hdulist[1].data
    Z='Z'+str(n_template)
    Z_ERR='Z_ERR'+str(n_template)
    MINRCHI2='MINRCHI2'+str(n_template)
    CLASS='CLASS'+str(n_template)
    z_template=table[Z][0]
    z_err_template=table[Z_ERR][0]
    class_template=table[CLASS][0]
    minchi_template=table[MINRCHI2][0]
    index_template=n_template-1
    template=hdulist[2].data[0][index_template]
    n=len(template)
    COEFF0=hdulist[0].header['COEFF0']
    COEFF1=hdulist[0].header['COEFF1']
    NAXIS1=n
    WAVE_END=COEFF0+(NAXIS1-1)*COEFF1
    wave_log=np.linspace(COEFF0,WAVE_END,n)
    wave=10**wave_log
    return wave,template,z_template,class_template,minchi_template,z_err_template


def get_spec(specfit):
    hdulist=fits.open(specfit)
    flux=hdulist[0].data[0]
    n=len(flux)
    COEFF0=hdulist[0].header['COEFF0']
    COEFF1=hdulist[0].header['COEFF1']
    NAXIS1=hdulist[0].header['NAXIS1']
    WAVE_END=COEFF0+(NAXIS1-1)*COEFF1
    wave_log=np.linspace(COEFF0,WAVE_END,n)
    wave=10**wave_log
    return wave,flux

def get_rm_spec(rm_spec_name,rm_out_file = None,rm_fit_number = None):
    w_spec,f_spec = get_spec(rm_spec_name)
    plt.plot(w_spec,f_spec,color ='Blue',label = rm_spec_name)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux (arbitrary units)')
    mean = np.nanmean(f_spec)
    med = np.nanmedian(f_spec)
    std = np.nanstd(f_spec)
    ymin = mean - 3 * std
    ymax = mean + 4 * std
    plt.ylim(ymin, ymax)
    plt.xlim(min(w_spec), max(w_spec))
    if rm_out_file!=None and rm_fit_number!=None:
        w_temp,flux_temp,z_temp,class_temp,minchi_temp,z_err_temp = get_template(rm_out_file,rm_fit_number)
        label_str = 'Templ. Class = {}, z={:.5f} +- {:.5f}\nchisqr_dof={:.3f}'.format(class_temp, z_temp, z_err_temp, minchi_temp)
        plt.plot(w_temp,f_temp,color='Red',label=label_str)
        plt.legend(fontsize=9, bbox_to_anchor=(1.05, -0.13), loc=1, borderaxespad=0.)
        plt.show()
        return w_spec,f_spec,w_temp,flux_temp
    plt.legend()
    plt.show()
    return w_spec,f_spec


def is_local_minima(a):
    """For a given array a, it returns true for local minima
    (copied from pyntejos.utils)."""
    a = np.array(a)
    mask = []
    for i in range(1, len(a) - 1):
        cond = (a[i] < a[i - 1]) and (a[i] < a[i + 1])
        if cond:
            mask += [1]
        else:
            mask += [0]
    mask = np.array(mask)
    mask = np.append(0, mask)
    mask = np.append(mask, 0)
    return mask == 1


def is_local_maxima(a):
    """For a given array a, returns true for local maxima
    (copied from pyntejos.utils)"""
    a = np.array(a)
    mask = []
    for i in range(1, len(a) - 1):
        cond = (a[i] > a[i - 1]) and (a[i] > a[i + 1])
        if cond:
            mask += [1]
        else:
            mask += [0]
    mask = np.array(mask)
    mask = np.append(0, mask)
    mask = np.append(mask, 0)
    return mask == 1

