Getting started
---------------

Inicializar cube
++++++++++++++++

Initializing is easy:
You must be in "ipython --pylab" enviroment::

    from muse.musecube import MuseCube
    cube = MuseCube(filename_cube, filename_white)


Get a spectrum
++++++++++++++

You can get an spectrum of a geometrical region by using::

    sp1 = cube.get_spec_from_ellipse_params(134, 219, 5, mode='wwm')

This ``sp1`` is an ``XSpectrum1D`` object of the spaxels within a circle of radius 5 at position xy=(134, 219).

You can also define an elliptical aperture by using instead::

    sp1 = cube.get_spec_from_ellipse_params(134,219,[10,5,35], mode='wwm')

where [10,5,35] corresponds to the semimajor axis, semiminor axis and rotation angle respectively:


You also may want to get the spectrum of a region defined by a single string line in DS9 format (e.g. see http://ds9.si.edu/doc/ref/region.html)
To do this, you can use the function::

    sp1 = cube.get_spec_from_region_string(region_string, mode = 'wwm')

In both of the get_spec() functions you can set ``save = True`` to save the spectrum to the hard_disk

Another extra feature is given by the  function::

    sp1 = cube.get_spec_image(center,halfsize,mode='wwm')

This code will, in addition of extract the spectrum given by center = (x,y) and halfsize either the radius of a circular
region or a set of [a,b,theta] parameters defining an ellipse, will plot the spectrum and will show the source that is being analysed in a  subplot.

If you want to insert the input positions in wcs space, you can set the coord_system parameter to wcs by adding

``coord_system = 'wcs'``

Finally, you are able to get the spectrum of a single spaxel of the cube by using::

    sp1 = cube.get_spec_spaxel(x,y,coord_system ='pix')

Again, you can set coord_system = 'wcs' if you want to insert an xy coordinate in degrees.

Get a spectrum interactively
++++++++++++++++++++++++++++
To use this feature, the class must have been initialized in a "ipython --pylab qt" enviroment
It's also needed the package roipoly. Installation instructions and LICENSE in:
https://github.com/jdoepfert/roipoly.py/

This feature allows the user to interactively define a region in the canvas as a polygon. To do this::

    spec=cube.get_spec_from_interactive_polygon_region(mode='wwm')

This will turn interactive the canvas. To select the spaxel that will be the vertices of the region, just press left click on them.
When you have finished, just press right click and then enter to continue. The las vertex that you selected will link the first one to define the contour of the region.



Get the spectrum of a region defined in a DS9 .reg file
+++++++++++++++++++++++++++++++++++++++++++++++++++++++
You also can define a region in a ds9 .reg file.
The only thing needed is that the .reg file MUST be saved in physical coordinates. Once this is done, you can get the spectrum::

    spec = cube.get_spec_from_ds9regfile(regfile,mode='wwm')

Modes of spectrum extraction
++++++++++++++++++++++++++++

As you have noted, all the diferent `get_spec_` functions have the keyword argument "mode". The mode availables to combine the spectrum of the diferent spaxels in a region are:

              * `ivar` - Inverse variance weighting, variance is taken only spatially, from a "white variance image"
              * `sum` - Sum of total flux
              * `gaussian` - Weighted mean. Weights are obtained from a 2D gaussian fit of the bright profile
              * `wwm` - 'White Weighted Mean'. Weigted mean, weights are obtained from the white image, smoothed using a gaussian filter of sigma = npix. If npix=0, no smooth is done
              * `ivarwv` - Weighted mean, the weight of every pixel is given by the inverse of it's variance
              * `mean`  -  Mean of the total flux
              * `median` - Median of the total flux
              * `wwm_ivarwv` - Weights given by both, `ivarwv` and `wwm`
              * `wwm_ivar` - Weghts given by both, `wwm` and `ivar`

Note: The gaussian method is not available in `get_spec_from_ds9regfile()` nor `get_spec_from_interactive_polygon_region()`

Other keyword parameters
------------------------
Also, all the `get_spec_` function have the keyword arguments `npix` , `empirical_std`, `n_figure` and `save`.

Some modes of extraction require a npix value (default = 0). This value correspond to the sigma of the gaussian function
that will smooth the white image, where the bright profile will be obtained. If npix = 0, no smooth is done.

If `empirical_std = True` (default = False) the uncertainties of the spectrum will be calculated empirically

`n_figure` is the number of the figure that will display the new_spectrum

if `save` = True (default = False) The new spectrum extracted will be saved to the hard disk


Use a SExtractor output file as an input
++++++++++++++++++++++++++++++++++++++++

The software allows the extraction and save of a serie of sources detected in a SExtractor output file.
To do this, you should have at least the next parameters in the SExtractor output file:
    * X_IMAGE.
    * Y_IMAGE.
    * A_IMAGE.
    * B_IMAGE.
    * THETA_IMAGE.
    * FLAGS.
    * NUMBER.
    * MAG_AUTO

First, to plot your regions, you can use::

    cube.plot_sextractor_regions('sextractor_filename',flag_threshold = 32,a_min = 3.5)

Where sextractor_filename is the name of the SExtractor's output. Every source with a SExtractor flag higher
than flag_threshold will be marked in red.

The a_min value correspond to the minimum number of pixels that will have the semimajor axis of a region.
The original (a/b) ratio will be constant, but this set a minimum size for the elliptical apertures.

Once you are satisfied with the regions that will be extracted, you can run::

    cube.save_sextractor_spec(sextractor_filename, flag_threshold=32, redmonster_format=True,a_min=3.5, n_figure=2,
                              mode='wwm', mag_kwrd='mag_r', npix=0)
This will save in the hard disk the spectra of all the sources defined in the sextractor_filename which flags be lower or
equal than flag_threshold using the specified mode.

If `redmonster_format = True`, the spectra will be saved in a format redeable for redmonster software.

Also, you can set the parameter ``mag_kwrd`` which by default is ``'mag_r'`` to the keyword in the new fits_image that will
contain the SExtractor's MAG_AUTO value


Estimate seeing
+++++++++++++++

The method::

    cube.determinate_seeing_from_white(x_center,y_Center,halfsize)
Will allow  you to estimate the seeing using the white image. The user must insert as the input the xy coordinates in spaxel space
of a nearly puntual source expanded by the seeing. The method will fit a 2D gaussian to the bright profile and will associate
the FWHM of the profile with the seeing. The halfsize parameter  indicates the radius size in spaxels of the source that will be fited.


Compose a filtered image
++++++++++++++++++++++++

If you want to do a photometric analysis from the Muse Cube, you would need to convolute your data with a sdss photometric filter
and compose a new filtered image. To do this, you can use the method::

    cube.get_filtered_image(_filter = 'r')

_filter can be any of ugriz. This method will write a new filtered image that will be usefull to photometry analysis


Create Videos
+++++++++++++

As an extra analysis to your data, the MuseCube Class allows the user to create 2 types of videos (need the cv2 package)

::

    cube.create_movie_redshift_range(z_ini,z_fin_dz)
Will create a video which frames will be, at each redshifts, the sum of all wavelengths that would fall at strong emission lines
(Ha,Hb,OII,OIII)

::

    cube_create_movie_wavelength_range(w_ini,w_end,width)

Will create a movie that goes from wavelength = w_ini suming a number of wavelength values given by width, to wavelength = w_end








