Getting started
---------------

Initialize the cube
++++++++++++++++

Initializing is easy:
You must be in "ipython --pylab" enviroment::

    from PyMUSE.musecube import MuseCube
    cube = MuseCube(filename_cube, filename_white)

If for any reason you do not have the white image, you can still initialize the cube just typing::

    cube = MuseCube(filename_cube)

This will create and save to the hard disk a the new white image. You may need to change the default visualization parameters if you do this.

Other initialization parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            * ``pixelsize``: Default = 0.2 * u.arcsec.
            * ``n_fig`` Default = 1. Figure to display the GUI.
            * ``flux_units`` Default = 1E-20 * u.erg / u.s / u.cm ** 2 / u.angstrom.
            * ``vmin`` Default = 0.
            * ``vmax`` Default = 5.
            * ``wave_cal`` Default = 'air'. Possible values: 'air','vac'. If 'air', all spectra obtained will be automatically calibrated to vacuum.




Visualization
+++++++++++++

In any moment, you can clean the primary canvas by using::

    cube.reload_canvas()

You can update the vmin and vmax parameters of the dynamical range of the visualization by setting::

    cube.reload_canvas(vmin=10,vmax=200)
for example.

If you want to use an alternative matplotlib color map for the visualization, just use::

    cube.color_gui(cmap='BuPu')

where cmap can be any of the matplotlib color maps.




Get a spectrum
++++++++++++++

You can get an spectrum of a geometrical region by using::

    spectrum = cube.get_spec_from_ellipse_params(134, 219, 5, mode='wwm')

This ``spectrum`` is an ``XSpectrum1D`` object of the spaxels within a circle of radius 5 at position xy=(134, 219).

You can also define an elliptical aperture by using instead::

    spectrum = cube.get_spec_from_ellipse_params(134,219,[10,5,35], mode='wwm')

where [10,5,35] corresponds to the semimajor axis, semiminor axis and rotation angle respectively:


You also may want to get the spectrum of a region defined by a single string line in DS9 format (e.g. see http://ds9.si.edu/doc/ref/region.html)
To do this, you can use the function::

    spectrum = cube.get_spec_from_region_string(region_string, mode = 'wwm')

In both of the get_spec() functions you can set ``save = True`` to save the spectrum to the hard_disk

Another extra feature is given by the  function::

    spectrum = cube.get_spec_and_image(center,halfsize,mode='wwm')

This code will, in addition of extract the spectrum given by center = (x,y) and halfsize either the radius of a circular
region or a set of [a,b,theta] parameters defining an ellipse, will plot the spectrum and will show the source that is being analysed in a  subplot.

If you want to insert the input positions and semi-axes in degrees, you can set the coord_system parameter to wcs by adding

``coord_system = 'wcs'``

Finally, you are able to get the spectrum of a single spaxel of the cube by using::

    spectrum = cube.get_spec_spaxel(x,y,coord_system ='pix')

Again, you can set coord_system = 'wcs' if you want to insert an xy coordinate in degrees.

Get a spectrum interactively
++++++++++++++++++++++++++++
To use this feature, the class must have been initialized in a "ipython --pylab qt" enviroment
It's also needed the package roipoly. Installation instructions and LICENSE in:
https://github.com/jdoepfert/roipoly.py/

This feature allows the user to interactively define a region in the canvas as a polygon. To do this::

    spectrum=cube.get_spec_from_interactive_polygon_region(mode='wwm')

This will turn interactive the canvas. To select the spaxel that will be the vertices of the region, just press left click on them.
When you have finished, just press right click and then enter to continue. The last vertex that you selected will link the first one to define the contour of the region.



Get the spectrum of a region defined in a DS9 .reg file
+++++++++++++++++++++++++++++++++++++++++++++++++++++++
You also can define a region in a ds9 .reg file.
The only thing needed is that the .reg file MUST be saved in physical coordinates. Once this is done, you can get the spectrum::

    spectrum = cube.get_spec_from_ds9regfile(regfile,mode='wwm')

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
              * `wfrac` - It only takes the fraction `frac` of brightest spaxels (white) in the region
                         (e.g. frac=0.1 means 10% brightest) with equal weight.
Note: The gaussian method is not available in `get_spec_from_ds9regfile()` nor `get_spec_from_interactive_polygon_region()`

Other keyword parameters
------------------------
Also, all the `get_spec_` function have the keyword arguments `npix` , `empirical_std`, `n_figure` and `save`, `frac`

Some modes of extraction require a npix value (default = 0). This value correspond to the sigma of the gaussian function
that will smooth the white image, where the bright profile will be obtained. If npix = 0, no smooth is done.

The parameter `frac` (default = 0.1) will be used in mode = `wfrac`, and it defines the fraction of brightest spaxels that will be considered in the sum of the flux.

If `empirical_std = True` (default = False) the uncertainties of the spectrum will be calculated empirically

`n_figure` is the number of the figure that will display the new_spectrum

if `save` = True (default = False) The new spectrum extracted will be saved to the hard disk

Read a spectrum saved by get_spec_methods
+++++++++++++++++++++++++++++++++++++++++

If you used the ``save``= True option, you saved the spectrum to the hard-disk as a fits file. To access the data you can use::

    from linetools.spectra.io import readspec
    spectrum = readspec('spectrum_fitsname')

This will create a ``XSpectrum1D`` object from the fits file. You can access to the spectrum wavelength, flux and sigma
typing spectrum.wavelength, spectrum.flux and spectrum.sig. Additional information on the ``XSpectrum1D`` Class can be found in: https://github.com/linetools/linetools/blob/master/linetools/spectra/xspectrum1d.py



Use a SExtractor output file as an input
++++++++++++++++++++++++++++++++++++++++

The software allows the extraction and save of a set of sources detected in a SExtractor output file.
To do this, you should have at least the next parameters in the SExtractor output file:
    * X_IMAGE.
    * Y_IMAGE.
    * A_IMAGE.
    * B_IMAGE.
    * THETA_IMAGE.
    * FLAGS.
    * NUMBER.
    * MAG_AUTO

(Assuming that you ran SExtractor in the white image or any image with the same dimensions and astrometry of the cube)
First, to plot your regions, you can use::

    cube.plot_sextractor_regions('sextractor_filename', flag_threshold=32, a_min=3.5)

Where sextractor_filename is the name of the SExtractor's output. Every source with a SExtractor flag higher
than flag_threshold will be marked in red.

The a_min value correspond to the minimum number of spaxels that will have the semimajor axis of a region.
The original (a/b) ratio will be constant, but this set a minimum size for the elliptical apertures.

Once you are satisfied with the regions that will be extracted, you can run::

    cube.save_sextractor_spec('sextractor_filename', flag_threshold=32, redmonster_format=True, a_min=3.5, n_figure=2,
                              mode='wwm', mag_kwrd='mag_r', npix=0, frac = 0.1)
This will save in the hard disk the spectra of all the sources defined in the sextractor_filename which flags be lower or
equal than flag_threshold using the specified mode.

If `redmonster_format = True`, the spectra will be saved in a format redeable for redmonster software (http://www.sdss.org/dr13/algorithms/redmonster-redshift-measurement-and-spectral-classification/).

You can access to the data of a file writen in this format doing the next::

    import PyMUSE.utils as mcu
    wv,fl,er = mcu.get_rm_spec(rm_spec_name)
where rm_spec_name is the name of the fits file.

Also, you can set the parameter ``mag_kwrd`` which by default is ``'mag_r'`` to the keyword in the new fits_image that will
contain the SExtractor's MAG_AUTO value

It is possible the usage of a different image as an input for SExtractor. If this is the case, you should not use the
X_IMAGE, Y_IMAGE, A_IMAGE, B_IMAGE given by SExtractor (although they still must be included in the parameters list), because the spaxel-wcs conversion in the
image given to SExtractor will be probably different to the conversion in the MUSE cube.  You may want to include the parameters:
    * X_WORLD.
    * Y_WORLD
    * A_WORLD
    * B_WORLD
You also may want to be sure that the astrometry between the 2 images in consistent (on the other hand, the regions defined by SExtractor in the image will be shifted in the cube)
Once you included them in the parameters list, you should set the parameter `wcs_coords = True` in both functions::

    cube.plot_sextractor_regions('sextractor_filename', flag_threshold=32, a_min=3.5, wcs_coords=True)

to plot the regions and::

    cube.save_sextractor_spec('sextractor_filename', flag_threshold=32, redmonster_format=True, a_min=3.5, n_figure=2,
                              mode='wwm', mag_kwrd='mag_r', npix=0, frac = 0.1, wcs_coords = True)
to save them.

Save a set of spectra defined by a multi regionfile DS9 .reg file
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
You can save all the spectra of regions defined by a DS9 region file to the hard disk. Just use::

    cube.save_ds9regfile_specs(regfile,mode='wwm',frac=0.1,npix=0,empirical_std=False,redmonster_format=True,id_start=1, coord_name = False)

Again, you can select between all available modes (except gaussian). The different spectra in the file will be identified by an id,
starting from id_start (default = 1). The coord_name variable will determine how the different spectra are named. If is False, The spectra will be named as
ID_regfile.fits. If True, The name will depend of the first (X,Y) pair of each region. This is particularly good for ellipses and circles, but not as exact in polygons.

Save a set of spectra defined by a MUSELET output fits table
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MUSELET (for MUSE Line Emission Tracker)  is an emission line galaxy detection tool based on SExtractor from MPDAF (MUSE Python Data Analysis Framework) Python package (http://mpdaf.readthedocs.io/en/latest/muselet.html)
PyMUSE allow the user te extraction of a set spectra given a MUSELET output fits table. The method::

    cube.save_muselet_specs(self, filename, mode='wwm', params=4, frac=0.1, npix=0, empirical_std=False, redmonster_format=True, ids='all')

Will do it easily. Most of the keyword parameters are related to the extraction modes. The important parameters are ``params`` and ``ids``:
``params`` by default is set to 4 and correspond to the elliptical parameter of the extraction for ALL the sources in the catalog. It can be either a int or a iterable [a,b, theta] (in spaxel units)
``ids`` by default is set to 'all'. This means that ``save_muselet_specs()`` will extract all the sources in the MUSELET catalog. If you set ids = [1,5,23] for example, the function will extract only the sources with that IDs in the MUSELET catalog.

Saving a single spectrum to the hard disk
+++++++++++++++++++++++++++++++++++++++++

To do this you can use the ``XSpectrum1D`` functions::

    spectrum.write_to_ascii(outfile_name)
    spectrum.write_to_fits(outfile_name)
You also may want to save the spectrum in a fits redeable for redmonster. In that case use the MuseCube function::

    mcu.spec_to_redmonster_format(spectrum, fitsname, n_id=None, mag=None)
If `n_id` is not  `None`, the new fitsfile will contain a ID keyword with n_id in it.
If `mag` is not `None`, must be a  tuple with two elements. The first one must contain the keyword that will be in the header (example: mag_r) and the second one must contain the value that will be in that keyword on the header of the new fitsfile



Estimate seeing
+++++++++++++++

The method::

    cube.determinate_seeing_from_white(x_center,y_center,halfsize)
Will allow  you to estimate the seeing using the white image. The user must insert as the input the xy coordinates in spaxel space
of a nearly point source expanded by the seeing. The method will fit a 2D gaussian to the bright profile and will associate
the FWHM of the profile with the seeing. The halfsize parameter  indicates the radius size in spaxels of the source that will be fited.

Image creation
--------------

Create image collapsing the Cube
+++++++++++++++++++++++++++++++++

You can create a 2D image by collapsing some wavelength slices of the cube using the method::

    cube.get_image(wv_input, fitsname='new_collapsed_cube.fits', type='sum', n_figure=2, save=False, stat=False)

IMPORTANT!!: wv_input must be list. The list can contain either individual wavelength values (e.g [5000,5005,5010]) or
a wavelength range (defined as [[5000,6000]] to collapse all wavelength between 5000 and 6000 angstroms).
If save is True, the new image will be saved to the hard disk as ``fitsname``. The ``type`` of collapse can be either 'sum'
or 'median'. n_figure is the figure's number  to display the image if ``save`` = True. Finally, if stat = True, the collapse will
be done in the stat extension of the MUSE cube.
If you want to directly create a new "white" just use::

    cube.create_white(new_white_fitsname='white_from_colapse.fits', stat=False, save=True)

This will sum all wavelengths and the new image will be saved in a fits file named by ``new_white_fitsname``. If ``stat``=True, the new
image will be created from the stat extension, as the sum of the variances along the wavelength range.

Maybe yo want to collapse more than just one wavelength range (for example, the range of several emission lines)
To do that, you may want to use the method::

    cube.get_image_wv_ranges(wv_ranges, substract_cont=True, fitsname='new_collapsed_cube.fits', save=False, n_figure=3)

wv_ranges must be a list of ranges (for example [[4000,4100],[5000,5100],[5200,5300]]). You can use the method::

    cube.create_ranges(z,width=10)

To define the ranges that correspond to the [OII, Hb, OIII 4959,OIII 5007, Ha].  This method will return the list of the range
of these transitions at redshift z, and the width given (in angstroms). The method will only return those ranges that
remains inside the MUSE wavelength range.
Finally, if ``substract_cont`` is True, the flux level around the ranges given by wv_ranges will be substracted from the image.

Create a smoothed white image
++++++++++++++++++++++++++++++

The method::

    cube.get_smoothed_white(npix=2, save=True, **kwargs)

returns a smoothed version of the white image. ``npix`` defines the sigma of the gaussian filter. **kwargs are passed to
scipy.ndimage.gaussian_filter(). The method ``cube.spatial_smooth(npix, output="smoothed.fits", **kwargs)`` do the same for the whole cube, and saves
the new MUSE Cube under the name given by ``output`` (The STAT extension is not touched)



Compose a filtered image
++++++++++++++++++++++++

If you want to do a photometric analysis from the Muse Cube, you would need to convolve your data with a photometric filter
and compose a new filtered image. To do this, you can use the method::

    cube.get_filtered_image(_filter = 'r')

This method will write a new filtered image that will be useful to photometry analysis
Available filters: u,g,r,i,z,V,R (The Johnson filters V and R have been slightly reduced  in order to fit the MUSE spectral range)


Compute kinematics
++++++++++++++++++++++++++

An useful thing to do with a MuseCube is a kinematic analysis of an extended source. The function::

    cube.compute_kinematics(x_c,y_c,params,wv_line_vac, wv_range_size=35, type='abs', z=0)

estimates de kinematics of the elliptical region defined by (x_c,y_c,params) in spaxels. The method extract the 1-D spectrum of every spaxel within
the region and fit a gaussian + linear model, in order to fit and emi/abs line and the continuum. The required paramters are:
    * x_c
    * y_c
    * params
That define the elliptical region.
    * wv_line_vac: wavelength of the transition in vacuum.
    * wv_range_size: Angstroms. Space at each side of the line in the spectrum. Set this parameter in order to fit the complete transition but do not include near additional lines
    * type: 'abs' or 'emi'. Type of transition to fit. 'abs' for absorption and 'emi' for emission.
    * z: redshift of the galaxy.
This function returns the kinematic image of the region, and saves the image in a .fits file.
IMPORTANT: Select strong lines that be spatially extended.


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








