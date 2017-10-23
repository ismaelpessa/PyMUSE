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

The simplest case is the extraction of an arbitrarily defined circular region::

    spec = cube.get_spec_from_ellipse_params(137,56,10,color='Blue')

Of course, the region can be defined by a set of elliptical parameters [a,b,theta] instead of a single radius::

    spec = cube.get_spec_from_ellipse_params(137,56,[10,5,45],color='green')

And the regions can also be declared in degrees::

    spec = cube.get_spec_from_ellipse_params(212.6656,23.0744,[0.0005,0.00025,-45],coord_system='wcs',color='red')

The apertures of extraction can also be defined by a DS9 region string::

    spec = cube.get_spec_from_region_string('physical;ellipse(137,56,10,5,45) # color = cyan')

And even from a DS9 region file::

    spec=cube.get_spec_from_ds9regfile('example.reg',i=0)
    spec=cube.get_spec_from_ds9regfile('example.reg',i=1)
    spec=cube.get_spec_from_ds9regfile('example.reg',i=2)

Some extra features regarding the spectrum extraction are::

    spec= cube.get_spec_and_image((137,56),halfsize=12)
That returns not only the 1-D spectrum but also the 2-D image of the selected source, and if you initialized in
"--pylab qt enviroment" you are able to interactively define a polygonal region::

    spec = cube.get_spec_from_interactive_polygon_region()


Modes of extraction
^^^^^^^^^^^^^^^^^^^

Once the aperture of extraction is defined, the combination of the spaxels inside it to calculate the total flux per
wavelength bin can be done in a variety of manners:

              * ``ivar`` - Inverse variance weighting, variance is taken only spatially, from a "white variance" images.
              * ``sum`` - Sum of total flux.
              * ``gaussian`` - Weighted mean. Weights are obtained from a 2D gaussian fit of the bright profile
              * ``wwm`` - 'White Weighted Mean'. Weighted mean, weights are obtained from the white image, optionally smoothed using a gaussian filter.
              * ``ivarwv`` - Weighted mean, the weight of every pixel is given by the inverse of it's variance.
              * ``mean``  -  Mean of the total flux
              * ``median`` - Median of the total flux
              * ``wwm_ivarwv`` - Weights given by both, ``ivarwv`` and ``wwm``
              * ``wwm_ivar`` - Weights given by both, ``wwm`` and ``ivar``
              * ``wfrac`` - It only takes the fraction ``frac`` of brightest spaxels (white) in the region.
                         (e.g. frac=0.1 means 10% brightest) with equal weights.


Imaging
-------

PyMUSE also offer to the users a set of features to produce different types of images.

Masking images is possible, just define a DS9 region file with the region that you want to mask out::

    cube.get_image(wv_input=cube.wavelength,maskfile='example2.reg',save=True,inverse_mask=False)

Or mask in::

    cube.get_image(wv_input=cube.wavelength,maskfile='example2.reg',save=True,inverse_mask=True)


The parameter wv_input can ve either an iterable that contains the wavelengths that you want to collapse or a wavelength range::

    cube.get_image(wv_input=[[4750,6000]],maskfile='example2.reg',save=True,inverse_mask=False)

Filtered images are also supported. PyMUSE has the feature of convolve the MUSE datacube with photometric filters (SDSS and Johnson filters are available)
Given the MUSE wavelength range PyMUSE can create r,i,R,V images::

    cube.get_filtered_image(_filter='r', custom_filter=None)

You can also define your own filter, for example if we define a Gaussian transmission curve::

    import numpy as np
    from astropy.modeling import models
    Gauss=models.Gaussian1D(mean=5400,stddev=200,amplitude=1)
    w=np.arange(5000,6000,1)
    tc=Gauss(w)
    plt.figure()
    plt.plot(w,tc)

We can use::

    cube.get_filtered_image(custom_filter=[w,tc])

To create the new image.

To get an smoothed image, the method::

    cube.get_smoothed_white(npix=1)

will create a new smoothed white image. The smooth is done by a Gaussian filter with standard deviation given by npix.










