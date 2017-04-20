"""Utilities for MuseCube"""

import matplotlib.pyplot as plt
import numpy as np

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



