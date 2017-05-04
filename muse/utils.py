"""Utilities for MuseCube"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from linetools.spectra.xspectrum1d import XSpectrum1D

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
    """
    Function to plot and examine a spectrum which is in a readable format to redmonster software.
    Optionally, the user can inster a redmonster output file and compare the input spectrum with the
    template contained in the fit given by rm_fit_number
    :param rm_spec_name: Name of the fits file which contain the spectrum in redmonster format
    :param rm_out_file: Optional. Default = None. Output file from redmonster
    :param rm_fit_number: Optional. Default = None. Number of the fit in rm_out_file (1 to 5)
    :return:
    """
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


def calculate_empirical_rms(spec,test = False):
    fl = spec.flux.value
    wv = spec.wavelength.value
    min_cond = is_local_minima(fl)
    max_cond = is_local_maxima(fl)
    min_local_inds = np.where(min_cond)[0]
    max_local_inds = np.where(max_cond)[0]
    ########  Nueva version con interpolacion del espectro
    no_minmax_inds = np.where(~min_cond & ~max_cond)[0]
    wv_nominmax = wv[no_minmax_inds]
    fl_nominmax = fl[no_minmax_inds]
    interpolated_nominmax=interp1d(wv_nominmax,fl_nominmax,kind='linear')
    new_fl_nominmax=interpolated_nominmax(wv)


    #Segunda implementacion
    max_mean_diferences=np.abs(fl[max_local_inds] - new_fl_nominmax[max_local_inds])
    min_mean_diferences = np.abs(fl[min_local_inds] - new_fl_nominmax[min_local_inds])



    #Primera Implementacion
    #max_mean_diferences = np.abs(fl[max_local_inds] - (fl[max_local_inds+1]+fl[max_local_inds-1])/2.)
    #min_mean_diferences = np.abs(fl[min_local_inds] - (fl[min_local_inds+1]+fl[min_local_inds-1])/2.)

    wv_mins=wv[min_local_inds]
    wv_maxs=wv[max_local_inds]
    if test:
        plt.figure()
        plt.plot(wv,fl)
        plt.plot(wv_mins,fl[min_local_inds],marker='o',color='r',label='Local minimum')
        plt.plot(wv_maxs,fl[max_local_inds],marker='o',color='green',label='Local maximum')
        plt.plot(wv, new_fl_nominmax, color ='black',label='nor_min_max_interpolation')
    wv_all_index = list(np.concatenate((wv_mins,wv_maxs)))
    all_mean_diferences=list(np.concatenate((min_mean_diferences,max_mean_diferences)))
    wv_all_index_sorted,all_mean_diferences_sorted = zip(*sorted(zip(wv_all_index,all_mean_diferences)))
    wv_all_index_sorted, all_mean_diferences_sorted=list(wv_all_index_sorted),list(all_mean_diferences_sorted)
    if wv_all_index_sorted[0]!=wv[0]:
        wv_all_index_sorted.insert(0,wv[0])
        value = all_mean_diferences_sorted[0]
        all_mean_diferences_sorted.insert(0,value)
    n = len(wv)
    m=len(wv_all_index_sorted)
    if wv_all_index_sorted[m-1]!=wv[n-1]:
        wv_all_index_sorted.insert(m, wv[n-1])
        value = all_mean_diferences_sorted[m-1]
        all_mean_diferences_sorted.insert(m, value)
    wv_all_index_sorted, all_mean_diferences_sorted = np.array(wv_all_index_sorted),np.array(all_mean_diferences_sorted)
    interpolator = interp1d(wv_all_index_sorted, all_mean_diferences_sorted,kind = 'linear')
    sigma = interpolator(wv)
    if test:
        plt.plot(wv_all_index_sorted,all_mean_diferences_sorted,marker = 'o',color = 'black',label = 'mean diferences')
        plt.plot(wv,sigma,label = 'interpolated sigma',color = 'pink')
        plt.plot(wv,spec.sig.value,color='yellow',label = 'Original sigma')
        plt.legend()
        plt.show()
    return XSpectrum1D.from_tuple((wv, fl, sigma))



