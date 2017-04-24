Getting started
---------------

Inicializar cube
++++++++++++++++

Initializing is easy::

    from muse.musecube import MuseCube
    cube = MuseCube(filename_cube, filename_white)


Get a spectrum
++++++++++++++

You can get an spectrum of a geometrical region by using::

    sp1 = cube.get_spec_from_ellipse_params(134, 219, 5, mode='mean')

This ``sp1`` is an ``XSpectrum1D`` object of the spaxels within a circle of radius 5 at position xy=(134, 219).

You can also define an elliptical aperture by using instead::

    sp1 = cube.get_spec_from_ellipse_params(134,219,[10,5,35], mode='mean')

where [10,5,35] corresponds to the semimajor axis, semiminor axis and rotation angle respectively:

If you want to insert the input positions in wcs space, you can set the coord_system parameter to wcs by adding
``coord_system = 'wcs'``


You also may want to get the spectrum of a region defined by a single string line in DS9 format (e.g. see http://ds9.si.edu/doc/ref/region.html)
To do this, you can use the function::

    sp1 = get_spec_from_region_string(region_string)

In both of the get_spec() functions you can set ``save = True`` to save the spectrum to the hard_disk

Another extra feature is given by the  function::

    sp1 = get_spec_image(center,halfsize)

This code will, in addition of extract the spectrum given by center = (x,y) and halfsize either the radius of a circular
region or a set of [a,b,theta] parameters defining an ellipse, will plot the spectrum and will show the source that is being analysed in a  subplot.


Finally, you are able to get the spectrum of a single point of the cube by using::

    sp1 = get_spec_point(x,y,coord_system ='pix')

Again, you can set coord_system = 'wcs' if you want to insert an xy coordinate in degrees.

Use a SExtractor output file as an input
++++++++++++++++++++++++++++++++++++++++

The software allows the extraction and save of a serie of sources detected in a SExtractor output file.
To do this, you should have at least the next parameters in the SExtractor output file:
X_IMAGE
Y_IMAGE
A_IMAGE
B_IMAGE
THETA_IMAGE
FLAGS
NUMBER
MAG_AUTO

First, to plot your regions, you can use::

    cube.plot_sextractor_regions('sextractor_filename',flag_threshold = 32)

Where sextractor_filename is the name of the SExtractor's output. Every source with a SExtractor flag higher
than flag_threshold will be marked in red.

Once you are satisfied with the regions that will be extracted, you can run::

    cube.save_sextractor_spec('sextractor_filename',flag_threshold = 32)
This will save in the hard disk the spectra of all the sources defined in the sextractor_filename which flags be lower or
equal than flag_threshold.

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








