Tutorial
=========


Initializing
^^^^^^^^^^^^
Initializing is easy You must be in "ipython --pylab" enviroment.::

        from PyMUSE.musecube import MuseCube)
        cube = MuseCube(filename_cube, filename_white)

If for any reason you do not have the white image, you can still initialize the cube just typing.::
        
        cube = MuseCube(filename_cube)
Get a spectrum
^^^^^^^^^^^^^^
You can get an spectrum of a geometrical region by using::

    spectrum = cube.get_spec_from_ellipse_params(134, 219, 5, mode='wwm')

This ``spectrum`` is an ``XSpectrum1D`` object of the spaxels within a circle of radius 5 at position xy=(134, 219)/

You can also define an elliptical aperture by using instead::

    spectrum = cube.get_spec_from_ellipse_params(134,219,[10,5,35], mode='wwm')

where [10,5,35] corresponds to the semimajor axis, semiminor axis and rotation angle respectively/
You also may want to get the spectrum of a region defined by a single string line in DS9 format (e.g. see http//ds9.si.edu/doc/ref/region.html/
To do this, you can use the functioni::

    spectrum = cube.get_spec_from_region_string(region_string, mode = 'wwm')

In both of the get_spec() functions you can set ``save = True`` to save the spectrum to the hard_dis/

Another extra feature is given by the  function::

    spectrum = cube.get_spec_and_image(center,halfsize,mode='wwm')

This code will, in addition of extract the spectrum given by center = (x,y) and halfsize either the radius of a circula/
region or a set of [a,b,theta] parameters defining an ellipse, will plot the spectrum and will show the source that is being analysed in a  subplot/

If you want to insert the input positions and semi-axes in degrees, you can set the coord_system parameter to wcs by adding::

coord_system = 'wcs/

Finally, you are able to get the spectrum of a single spaxel of the cube by using::

    spectrum = cube.get_spec_spaxel(x,y,coord_system ='pix')

Again, you can set coord_system = 'wcs' if you want to insert an xy coordinate in degrees/


