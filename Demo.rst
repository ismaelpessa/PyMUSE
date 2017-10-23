Focus demo ADASS XXVII
======================

Multi Unit Spectroscopic Explorer (MUSE) is a panoramic integral-field spectrograph composed of 24 identical IFU modules.
It has a field of view of ~1'x1' spatially sampled at 0.2" per pixel. This implies ~90000 spectra obtained in a single exposure.
We introduce here PyMUSE, which is a Python package designed to help the users in the task of perform a complete analysis to these giants
datasets.

PyMUSE provides a set of potentially useful tools focused on versatility at the moment of extracting a spectrum or creating an image.

Initializing
------------
Initializing is easy You must be in "ipython --pylab" enviroment.::

        from PyMUSE.musecube import MuseCube
        cube = MuseCube('example_cube.fits', 'example_white.fits')

If for any reason you do not have the original white image you can still initialize a MuseCube::

        cube_ = MuseCube('example_cube.fits')

This will create a new white image by collapsing the wavelength axis of the cube.

Extraction of a spectrum
-------------------------

