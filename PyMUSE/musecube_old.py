import copy
import gc
import glob
import os
import warnings

import aplpy
import linetools.utils as ltu
import numpy as np
import numpy.ma as ma
import pyregion
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.ascii.sextractor import SExtractor
from astropy.modeling import models, fitting
from astropy.table import Table
from astropy.utils import isiterable
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.utils import name_from_coord
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import ndimage

import PyMUSE.utils as mcu


class MuseCube:
    """
    Class to handle VLT/MUSE data

    """

    def __init__(self, filename_cube, filename_white=None, pixelsize=0.2 * u.arcsec, n_fig=1,
                 flux_units=1E-20 * u.erg / u.s / u.cm ** 2 / u.angstrom, vmin=None, vmax=None, wave_cal='air'):
        """
        Parameters
        ----------
        filename_cube: string
            Name of the MUSE datacube .fits file
        filename_white: string
            Name of the MUSE white image .fits file
        pixel_size : float or Quantity, optional
            Pixel size of the datacube, if float it assumes arcsecs.
            Default is 0.2 arcsec
        n_fig : int, optional
            XXXXXXXX
        flux_units : Quantity
            XXXXXXXXXX

        """

        # init
        self.color = False
        self.cmap = ""
        self.flux_units = flux_units
        self.n = n_fig
        plt.close(self.n)
        self.wave_cal = wave_cal


        self.filename = filename_cube
        self.filename_white = filename_white
        self.load_data()
        self.white_data = fits.open(self.filename_white)[1].data
        self.hdulist_white = fits.open(self.filename_white)
        self.white_data = np.where(self.white_data < 0, 0, self.white_data)

        if not vmin:
            self.vmin=np.nanpercentile(self.white_data,0.25)
        else:
            self.vmin = vmin
        if not vmax:
            self.vmax=np.nanpercentile(self.white_data,98.)
        else:
            self.vmax = vmax
        self.gc2 = aplpy.FITSFigure(self.filename_white, figure=plt.figure(self.n))
        self.gc2.show_grayscale(vmin=self.vmin, vmax=self.vmax)

        # self.gc = aplpy.FITSFigure(self.filename, slices=[1], figure=plt.figure(20))
        self.pixelsize = pixelsize
        gc.enable()
        # plt.close(20)
        print("MuseCube: Ready!")

    def load_data(self):
        hdulist = fits.open(self.filename)
        print("MuseCube: Loading the cube fluxes and variances...")

        # import pdb; pdb.set_trace()
        self.cube = ma.MaskedArray(hdulist[1].data)
        self.stat = ma.MaskedArray(hdulist[2].data)

        print("MuseCube: Defining master masks (this may take a while but it is for the greater good).")
        # masking
        self.mask_init = np.isnan(self.cube) | np.isnan(self.stat)
        self.cube.mask = self.mask_init
        self.stat.mask = self.mask_init

        # for ivar weighting ; consider creating it in init ; takes long
        # self.flux_over_ivar = self.cube / self.stat

        self.header_1 = hdulist[1].header  # Necesito el header para crear una buena copia del white.
        self.header_0 = hdulist[0].header

        if self.filename_white is None:
            print("MuseCube: No white image given, creating one.")

            w_data = copy.deepcopy(self.create_white(save=False).data)

            w_header_0 = copy.deepcopy(self.header_0)
            w_header_1 = copy.deepcopy(self.header_1)

            # These loops remove the third dimension from the header's keywords. This is neccesary in order to
            # create the white image and preserve the cube astrometry
            for i in w_header_0.keys():
                if '3' in i:
                    del w_header_0[i]
            for i in w_header_1.keys():
                if '3' in i:
                    del w_header_1[i]

            # prepare the header
            hdu = fits.HDUList()
            hdu_0 = fits.PrimaryHDU(header=w_header_0)
            hdu_1 = fits.ImageHDU(data=w_data, header=w_header_1)
            hdu.append(hdu_0)
            hdu.append(hdu_1)
            hdu.writeto('new_white.fits', clobber=True)
            self.filename_white = 'new_white.fits'
            print("MuseCube: `new_white.fits` image saved to disk.")

    def color_gui(self, cmap):
        """
        Function to change the cmap of the canvas
        :param cmap: string. matplotlib's color map. cmap = 'none' to gray scale again
        :return:
        """
        if cmap == 'none':
            self.color = False
            self.cmap = ""
        else:
            self.color = True
            self.cmap = cmap
        self.reload_canvas()

    def get_smoothed_white(self, npix=2, save=True, show=False, **kwargs):
        """Gets an smoothed version (Gaussian of sig=npix)
        of the white image. If save is True, it writes a file
        to disk called `smoothed_white.fits`.
        **kwargs are passed down to scipy.ndimage.gaussian_filter()
        """
        hdulist = self.hdulist_white
        im = self.white_data
        if npix > 0:
            smooth_im = ndimage.gaussian_filter(im, sigma=npix, **kwargs)
        else:
            smooth_im = im
        if save:
            hdulist[1].data = smooth_im
            prihdr = hdulist[0].header
            comment = 'Spatially smoothed with a Gaussian kernel of sigma={} spaxels (by MuseCube)'.format(npix)
            # print(comment)
            prihdr['history'] = comment
            hdulist.writeto('smoothed_white.fits', clobber=True)
            if show:
                fig = aplpy.FITSFigure('smoothed_white.fits', figure=plt.figure())
                fig.show_grayscale(vmin=self.vmin,vmax=self.vmax)

        return smooth_im

    def spec_to_vacuum(self, spectrum):
        spectrum_vac = spectrum
        if self.wave_cal == 'air':
            spectrum_vac.meta['airvac'] = 'air'
            spectrum_vac.airtovac()
            return spectrum_vac
        else:
            return spectrum_vac

    def spatial_smooth(self, npix, output="smoothed.fits", test=False, **kwargs):
        """Applies Gaussian filter of std=npix in both spatial directions
        and writes it to disk as a new MUSE Cube.
        Notes: the STAT cube is not touched.

        Parameters
        ----------
        npix : int
            Std of Gaussian kernel in spaxel units.
        output : str, optional
            Name of the output file
        test : bool, optional
            Whether to check for flux being conserved

        **kwargs are passed down to scipy.ndimage.gaussian_filter()

        Return
        ------
        Writes a new file to disk.

        """
        if not isinstance(npix, int):
            raise ValueError("npix must be integer.")

        cube_new = copy.deepcopy(self.cube)
        ntot = len(self.cube)
        for wv_ii in range(ntot):
            print('{}/{}'.format(wv_ii + 1, ntot))
            image_aux = self.cube[wv_ii, :, :]
            smooth_ii = ma.MaskedArray(ndimage.gaussian_filter(image_aux, sigma=npix, **kwargs))
            smooth_ii.mask = image_aux.mask | np.isnan(smooth_ii)

            # test the fluxes are conserved
            if test:
                gd_pix = ~smooth_ii.mask
                try:
                    med_1 = np.nansum(smooth_ii[gd_pix])
                    med_2 = np.nansum(image_aux[gd_pix])
                    print(med_1, med_2, (med_1 - med_2) / med_1)
                    np.testing.assert_allclose(med_1, med_2, decimal=4)
                except AssertionError:
                    import pdb
                    pdb.set_trace()
            cube_new[wv_ii, :, :] = smooth_ii
            # import pdb; pdb.set_trace()

        hdulist = fits.open(self.filename)
        hdulist[1].data = cube_new.data
        prihdr = hdulist[0].header
        comment = 'Spatially smoothed with a Gaussian kernel of sigma={} spaxels (by MuseCube)'.format(npix)
        print(comment)
        prihdr['history'] = comment
        hdulist.writeto(output, clobber=True)
        print("MuseCube: new smoothed cube written to {}".format(output))

    def get_mini_image(self, center, halfsize=15):

        """

        :param center: tuple of coordinates, in pixels
        :param size: length of the square around center
        :return: ndarray which contain the image
        """
        side = 2 * halfsize + 1
        image = [[0 for x in range(side)] for y in range(side)]
        data_white = fits.open(self.filename_white)[1].data
        center_x = center[0]
        center_y = center[1]
        for i in xrange(center_x - halfsize - 1, center_x + halfsize):
            for j in xrange(center_y - halfsize - 1, center_y + halfsize):
                i2 = i - (center_x - halfsize)
                j2 = j - (center_y - halfsize)
                image[j2][i2] = data_white[j - 1][i - 1]
        return image

    def get_gaussian_seeing_weighted_spec(self, x_c, y_c, radius, seeing=4):
        """
        Function to extract the spectrum of a circular aperture defined by x_c, y_c and radius in spaxel space.
        The spectrum is weighted by a 2d gaussian centered at the center of the aperture, with a std = seeing in spaxels
        :param x_c: x coordinate of the center of the aperture (spaxel)
        :param y_c: y coordiante of the center of the aperture (spaxel)
        :param radius: radius of the circular aperture
        :param seeing: standard deviation of the gaussian in spaxels
        :return: XSpectrum1D object
        """
        import scipy.ndimage.filters as fi
        new_3dmask = self.get_mini_cube_mask_from_ellipse_params(x_c, y_c, radius)
        w = self.wavelength
        n = len(w)
        fl = np.zeros(n)
        sig = np.zeros(n)
        self.cube.mask = new_3dmask
        for wv_ii in range(n):
            mask = new_3dmask[wv_ii]
            center = np.zeros(mask.shape)  ###Por alguna razon no funciona si cambio la asignacion a np.zeros_like(mask)
            center[y_c][x_c] = 1
            weigths = ma.MaskedArray(fi.gaussian_filter(center, seeing))
            weigths.mask = mask
            weigths = weigths / np.sum(weigths)
            fl[wv_ii] = np.sum(self.cube[wv_ii] * weigths)
            sig[wv_ii] = np.sqrt(np.sum(self.stat[wv_ii] * (weigths ** 2)))
        self.cube.mask = self.mask_init
        return XSpectrum1D.from_tuple((w, fl, sig))

    def get_spec_spaxel(self, x, y, coord_system='pix', n_figure=2, empirical_std=False, save=False):
        """
        Gets the spectrum of a single spaxel (xy) of the MuseCube
        :param x: x coordinate of the spaxel
        :param y: y coordinate of the spaxel
        :param coord_system: 'pix' or 'wcs'
        :return: spec: XSpectrum1D object
        """
        if coord_system == 'wcs':
            x_c, y_c = self.w2p(x, y)
            x_world, y_world = x, y
        else:
            x_c, y_c = x, y
            x_world, y_world = self.p2w(x, y)
        region_string = self.ellipse_param_to_ds9reg_string(x_c, y_c, 1, 1, 0, coord_system='pix')
        self.draw_pyregion(region_string)
        w = self.wavelength
        n = len(w)
        spec = np.zeros(n)
        sigma = np.zeros(n)
        for wv_ii in range(n):
            spec[wv_ii] = self.cube.data[wv_ii][int(y_c)][int(x_c)]
            sigma[wv_ii] = np.sqrt(self.stat.data[wv_ii][int(y_c)][int(x_c)])
        spec = XSpectrum1D.from_tuple((self.wavelength, spec, sigma))
        if empirical_std:
            spec = mcu.calculate_empirical_rms(spec)
        spec = self.spec_to_vacuum(spec)
        plt.figure(n_figure)
        plt.plot(spec.wavelength, spec.flux)
        coords = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
        name = name_from_coord(coords)
        plt.title(name)
        plt.xlabel('Angstroms')
        plt.ylabel('Flux (' + str(self.flux_units) + ')')
        if save:
            spec.write_to_fits(name + '.fits')
        return spec

    def get_spec_from_ellipse_params(self, x_c, y_c, params, coord_system='pix', mode='wwm', npix=0, frac=0.1,
                                     n_figure=2, empirical_std=False, save=False, color='green'):
        """
        Obtains a combined spectrum of spaxels within a geometrical region defined by
        x_c, y_c, param
        :param x_c: x coordinate of the center of the ellipse
        :param y_c: y coordinate of the center of the ellipse
        :param params: Either a float that will be interpreted as a radius, or an iterable [a,b,theta] with the ellipse parameters
        :param coord_system: str. Default = 'pix'.
            If coord_system = 'wcs' the coordinates will be considered as degrees
        :param mode: str
            Mode for combining spaxels:
              * `ivar` - Inverse variance weighting, variance is taken only spatially, from a "white variance image"
              * `sum` - Sum of total flux
              * `gaussian` - Weighted mean. Weights are obtained from a 2D gaussian fit of the bright profile
              * `wwm` - 'White Weighted Mean'. Weigted mean, weights are obtained from the white image, smoothed using a gaussian filter of sigma = npix. If npix=0, no smooth is done
              * `ivarwv` - Weighted mean, the weight of every pixel is given by the inverse of it's variance
              * `mean`  -  Mean of the total flux
              * `median` - Median of the total flux
              * `wwm_ivarwv' - Weights given by both, `ivarwv` and `wwm`
              * `wwm_ivar` - Weights given by both, `wwm` and `ivar`
              * `wfrac` - It only takes the fraction `frac` of brightest spaxels (white) in the region
                         (e.g. frac=0.1 means 10% brightest) with equal weight.
        :param frac. FLoat, default = 0.1
                     Parameter needed for wfrac mode
        :param npix: int. Default = 0
            Standard deviation of the gaussian filter to smooth (Only in wwm methods)
        :param n_figure: int. Default = 2. Figure to display the spectrum
        :param empirical_std: boolean. Default = False.
            If True, the errors of the spectrum will be determined empirically
        :param save: boolean. Default = False
            If True, the spectrum will be saved in hard_disk
        :return: spec: XSpectrum1D object
        """
        if mode == 'gaussian':
            spec = self.get_gaussian_profile_weighted_spec(x_c=x_c, y_c=y_c, params=params)
        else:
            new_mask = self.get_mini_cube_mask_from_ellipse_params(x_c, y_c, params, coord_system=coord_system,color=color)
            spec = self.spec_from_minicube_mask(new_mask, mode=mode, npix=npix, frac=frac)

        if empirical_std:
            spec = mcu.calculate_empirical_rms(spec)
        spec = self.spec_to_vacuum(spec)
        plt.figure(n_figure)
        plt.plot(spec.wavelength, spec.flux)
        if coord_system == 'wcs':
            x_world, y_world = x_c, y_c
        else:
            x_world, y_world = self.p2w(x_c, y_c)
        coords = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
        name = name_from_coord(coords)
        plt.title(name)
        plt.xlabel('Angstroms')
        plt.ylabel('Flux (' + str(self.flux_units) + ')')
        if save:
            spec.write_to_fits(name + '.fits')
        return spec

    def get_spec_from_interactive_polygon_region(self, mode='wwm', npix=0, frac=0.1,
                                                 n_figure=2,
                                                 empirical_std=False, save=False):
        """
        Function used to interactively define a region and extract the spectrum of that region

        To use this function, the class must have been initialized in a "ipython --pylab qt" enviroment
        It's also needed the package roipoly. Installation instructions and LICENSE in:
        https://github.com/jdoepfert/roipoly.py/
        :param mode: str, default = wwm
            Mode for combining spaxels:
              * `ivar` - Inverse variance weighting, variance is taken only spatially, from a "white variance image"
              * `sum` - Sum of total flux
              * `wwm` - 'White Weighted Mean'. Weigted mean, weights are obtained from the white image, smoothed using a gaussian filter of sigma = npix. If npix=0, no smooth is done
              * `ivarwv` - Weighted mean, the weight of every pixel is given by the inverse of it's variance
              * `mean`  -  Mean of the total flux
              * `median` - Median of the total flux
              * `wwm_ivarwv' - Weights given by both, `ivarwv` and `wwm`
              * `wwm_ivar` - Weghts given by both, `wwm` and `ivar`
              * `wfrac` - It only takes the fraction `frac` of brightest spaxels (white) in the region
                         (e.g. frac=0.1 means 10% brightest) with equal weight.
        :param frac. FLoat, default = 0.1
                     Parameter needed for wfrac mode
        :param npix: int. Default = 0
            Standard deviation of the gaussian filter to smooth (Only in wwm methods)
        :param n_figure: int. Default = 2. Figure to display the spectrum
        :param empirical_std: boolean.  Default = False
            If True, the errors of the spectrum will be determined empirically
        :param save: boolean. Default = False
            If True, the spectrum will be saved in hard_disk
        :return: spec: XSpectrum1D object

        """
        from roipoly import roipoly
        current_fig = plt.figure(self.n)
        MyROI = roipoly(roicolor='r', fig=current_fig)
        raw_input("MuseCube: Please select points with left click. Right click and Enter to continue...")
        print("MuseCube: Calculating the spectrum...")
        mask = MyROI.getMask(self.white_data)
        mask_inv = np.where(mask == 1, 0, 1)
        complete_mask = self.mask_init + mask_inv
        new_3dmask = np.where(complete_mask == 0, False, True)
        spec = self.spec_from_minicube_mask(new_3dmask, mode=mode, npix=npix, frac=frac)
        self.reload_canvas()
        plt.figure(n_figure)
        plt.plot(spec.wavelength, spec.flux)
        plt.ylabel('Flux (' + str(self.flux_units) + ')')
        plt.xlabel('Wavelength (Angstroms)')
        plt.title('Polygonal region spectrum ')
        plt.figure(self.n)
        MyROI.displayROI()
        if empirical_std:
            spec = mcu.calculate_empirical_rms(spec)
        spec = self.spec_to_vacuum(spec)
        if save:
            spec.write_to_fits('Poligonal_region_spec.fits')
        return spec

    def params_from_ellipse_region_string(self, region_string, deg=False):
        """
        Function to get the elliptical parameters of a region_string.
        If deg is True, only will be returned the center in degrees.
        Otherwise, all parameters will be returned in pixels
        :param region_string: Region defined as string using ds9 format
        :param deg: If True, only the center of the ellipse will be returned, in degrees.
        :return: x_center,y_center,params, parameter of the ellipse defined in region_string
        """
        r = pyregion.parse(region_string)
        if deg:
            x_c, y_c = r[0].coord_list[0], r[0].coord_list[1]
            if r[0].coord_format == 'physical' or r[0].coord_format == 'image':
                x_world, y_world = self.p2w(x_c - 1, y_c - 1)
            else:
                x_world, y_world = x_c, y_c
            return x_world, y_world
        else:
            if r[0].coord_format == 'physical' or r[0].coord_format == 'image':
                x_c, y_c, params = r[0].coord_list[0], r[0].coord_list[1], r[0].coord_list[2:5]
            else:
                x_world = r[0].coord_list[0]
                y_world = r[0].coord_list[1]
                par = r[0].coord_list[2:5]
                x_c, y_c, params = self.ellipse_params_to_pixel(x_world, y_world, params=par)
            return x_c - 1, y_c - 1, params

    def get_spec_from_region_string(self, region_string, mode='wwm', npix=0., frac=0.1, empirical_std=False, n_figure=2,
                                    save=False):
        """
        Obtains a combined spectrum of spaxels within geametrical region defined by the region _string, interpretated by ds9
        :param region_string: str
            Region defined by a string, using ds9 format (ellipse only in gaussian method)
            example: region_string = 'physical;ellipse(100,120,10,5,35) # color = green'
        :param mode: str
            Mode for combining spaxels:
              * `ivar` - Inverse variance weighting, variance is taken only spatially, from a "white variance image"
              * `sum` - Sum of total flux
              * `gaussian` - Weighted mean. Weights are obtained from a 2D gaussian fit of the bright profile (for elliptical regions only)
              * `wwm` - 'White Weighted Mean'. Weigted mean, weights are obtained from the white image, smoothed using a gaussian filter of sigma = npix. If npix=0, no smooth is done
              * `ivarwv` - Weighted mean, the weight of every pixel is given by the inverse of it's variance
              * `mean`  -  Mean of the total flux
              * `median` - Median of the total flux
              * `wwm_ivarwv' - Weights given by both, `ivarwv` and `wwm`
              * `wwm_ivar` - Weghts given by both, `wwm` and `ivar`
              * `wfrac` - It only takes the fraction `frac` of brightest spaxels (white) in the region
                         (e.g. frac=0.1 means 10% brightest) with equal weight.
        :param frac. Float, default = 0.1
                     Parameter needed for wfrac mode
        :param npix: int. Default = 0
            Standard deviation of the gaussian filter to smooth (Only in wwm methods)
        :param n_figure: int. Default = 2. Figure to display the spectrum
        :param empirical_std: boolean. Default = False.
            If True, the errors of the spectrum will be determined empirically
        :param save: boolean. Default = False
            If True, the spectrum will be saved in hard_disk
        :return: spec: XSpectrum1D object
        """

        if mode == 'gaussian':
            spec = self.get_gaussian_profile_weighted_spec(region_string_=region_string)
        else:
            new_mask = self.get_mini_cube_mask_from_region_string(region_string)
            spec = self.spec_from_minicube_mask(new_mask, mode=mode, npix=npix, frac=frac)
        if empirical_std:
            spec = mcu.calculate_empirical_rms(spec)
        self.draw_pyregion(region_string)
        spec = self.spec_to_vacuum(spec)
        plt.figure(n_figure)
        plt.plot(spec.wavelength, spec.flux)
        x_world, y_world = self.params_from_ellipse_region_string(region_string, deg=True)
        coords = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
        name = name_from_coord(coords)
        plt.title(name)
        plt.xlabel('Angstroms')
        plt.ylabel('Flux (' + str(self.flux_units) + ')')
        if save:
            spec.write_to_fits(name + '.fits')
        return spec

    def draw_ellipse_params(self, xc, yc, params, color='green'):
        """
        Function to draw in the interface the contour of the elliptical region defined by (xc,yc,params)
        :param xc: x coordinate of the center of the ellipse
        :param yc: y coordinate of the center of the ellipse
        :param params: either a single radius or [a,b,theta] iterable
        :param color: color to draw
        :return:
        """
        if isinstance(params, (float, int)):
            params = [params, params, 0]
        region_string = self.ellipse_param_to_ds9reg_string(xc, yc, params[0], params[1], params[2], color=color)
        self.draw_pyregion(region_string)

    def draw_pyregion(self, region_string):
        """
        Function used to draw in the interface the contour of the region defined by region_string
        :param region_string: str. Region defined by a string using ds9 format
        :return: None
        """
        hdulist = self.hdulist_white
        r = pyregion.parse(region_string).as_imagecoord(hdulist[1].header)
        fig = plt.figure(self.n)
        ax = fig.axes[0]
        patch_list, artist_list = r.get_mpl_patches_texts(origin=0)
        patch = patch_list[0]
        ax.add_patch(patch)

    def spec_from_minicube_mask(self, new_3dmask, mode='wwm', npix=0, frac=0.1):
        """Given a 3D mask, this function provides a combined spectrum
        of all non-masked voxels.

        Parameters
        ----------
        new_3dmask : np.array of same shape as self.cube
            The 3D mask
        mode : str
            Mode for combining spaxels:
              * `ivar` - Inverse variance weighting, variance is taken only spatially, from a "white variance image"
              * `sum` - Sum of total flux
              * `wwm` - 'White Weighted Mean'. Weigted mean, weights are obtained from the white image, smoothed using a gaussian filter of sigma = npix. If npix=0, no smooth is done
              * `ivarwv` - Weighted mean, the weight of every pixel is given by the inverse of it's variance
              * `mean`  -  Mean of the total flux
              * `median` - Median of the total flux
              * `wwm_ivarwv' - Weights given by both, `ivarwv` and `wwm`
              * `wwm_ivar` - Weghts given by both, `wwm` and `ivar`
              * `wfrac` - It only takes the fraction `frac` of brightest spaxels (white) in the region
                         (e.g. frac=0.1 means 10% brightest) with equal weight.
        Returns
        -------
        An XSpectrum1D object (from linetools) with the combined spectrum.

        """
        if mode not in ['ivarwv', 'ivar', 'mean', 'median', 'wwm', 'sum', 'wwm_ivarwv', 'wwm_ivar', 'wfrac']:
            raise ValueError("Not ready for this type of `mode`.")
        if np.shape(new_3dmask) != np.shape(self.cube.mask):
            raise ValueError("new_3dmask must be of same shape as the original MUSE cube.")

        n = len(self.wavelength)
        fl = np.zeros(n)
        er = np.zeros(n)
        if mode == 'ivar':
            var_white = self.create_white(stat=True, save=False)

        elif mode in ['wwm', 'wwm_ivarwv', 'wwm_ivar', 'wfrac']:
            smoothed_white = self.get_smoothed_white(npix=npix, save=False)
            if mode == 'wwm_ivar':
                var_white = self.create_white(stat=True, save=False)
            elif mode == 'wfrac':
                mask2d = new_3dmask[1]
                self.wfrac_show_spaxels(frac=frac, mask2d=mask2d, smoothed_white=smoothed_white)
        warn = False
        for wv_ii in xrange(n):
            mask = new_3dmask[wv_ii]  # 2-D mask
            im_fl = self.cube[wv_ii][~mask]  # this is a 1-d np.array()
            im_var = self.stat[wv_ii][~mask]  # this is a 1-d np.array()

            if len(im_fl) == 0:
                fl[wv_ii] = 0
                er[wv_ii] = 99
            elif mode == 'wwm':
                im_weights = smoothed_white[~mask]
                n_weights = len(im_weights)
                im_weights = np.where(np.isnan(im_weights), 0, im_weights)
                if np.sum(im_weights) == 0:
                    im_weights[:] = 1. / n_weights
                    warn = True
                im_weights = im_weights / np.sum(im_weights)
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
            elif mode == 'ivar':
                im_var_white = var_white[~mask]
                im_weights = 1. / im_var_white
                n_weights = len(im_weights)
                im_weights = np.where(np.isnan(im_weights), 0, im_weights)
                if np.sum(im_weights) == 0:
                    im_weights[:] = 1. / n_weights
                    warn = True
                im_weights = im_weights / np.sum(im_weights)
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
            elif mode == 'ivarwv':
                im_weights = 1. / im_var
                n_weights = len(im_weights)
                im_weights = np.where(np.isnan(im_weights), 0, im_weights)
                if np.sum(im_weights) == 0:
                    im_weights[:] = 1. / n_weights
                    warn = True
                im_weights = im_weights / np.sum(im_weights)
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
            elif mode == 'wwm_ivarwv':
                im_white = smoothed_white[~mask]
                im_weights = im_white / im_var
                n_weights = len(im_weights)
                im_weights = np.where(np.isnan(im_weights), 0, im_weights)
                if np.sum(im_weights) == 0:
                    im_weights[:] = 1. / n_weights
                    warn = True
                im_weights = im_weights / np.sum(im_weights)
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
            elif mode == 'wwm_ivar':
                im_white = smoothed_white[~mask]
                im_var_white = var_white[~mask]
                im_weights = im_white / im_var_white
                n_weights = len(im_weights)
                im_weights = np.where(np.isnan(im_weights), 0, im_weights)
                if np.sum(im_weights) == 0:
                    im_weights[:] = 1. / n_weights
                    warn = True
                im_weights = im_weights / np.sum(im_weights)
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
            elif mode == 'sum':
                im_weights = 1.
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
            elif mode == 'mean':
                im_weights = 1. / len(im_fl)
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
            elif mode == 'median':
                fl[wv_ii] = np.median(im_fl)
                er[wv_ii] = 1.2533 * np.sqrt(np.sum(im_var)) / len(im_fl)  # explain 1.2533
            elif mode == 'wfrac':
                if (frac > 1) or (frac < 0):
                    raise ValueError('`frac` must be value within (0,1)')
                im_white = smoothed_white[~mask]
                fl_limit = np.percentile(im_white, (1. - frac) * 100.)
                im_weights = np.where(im_white >= fl_limit, 1., 0.)
                n_weights = len(im_weights)
                im_weights = np.where(np.isnan(im_weights), 0., im_weights)
                if np.sum(im_weights) == 0:
                    im_weights[:] = 1. / n_weights
                    warn = True
                im_weights = im_weights / np.sum(im_weights)
                fl[wv_ii] = np.sum(im_fl * im_weights)
                er[wv_ii] = np.sqrt(np.sum(im_var * (im_weights ** 2)))
        if warn:
            warnings.warn(
                'Some wavelengths could not be combined using the selected mode (a mean where used only on those cases)')

        if mode not in ['sum', 'median', 'mean', 'wfrac']:  # normalize to match total integrated flux
            spec_sum = self.spec_from_minicube_mask(new_3dmask, mode='sum')
            fl_sum = spec_sum.flux.value
            norm = np.sum(fl_sum) / np.sum(fl)
            if norm < 0:
                warnings.warn(
                    "Normalization factor is Negative!! (This probably means that you are extracting the spectrum where flux<0)")
            fl = fl * norm
            er = er * abs(norm)
            print('normalization factor relative to total flux = ' + str(norm))

        return XSpectrum1D.from_tuple((self.wavelength, fl, er))

    def get_spec_and_image(self, center, halfsize=15, n_figure=3, mode='wwm', coord_system='pix', npix=0, frac=0.1,
                           save=False, empirical_std=False):

        """
        Function to Get a spectrum and an image of the selected source.

        :param center: Tuple. Contain the coordinates of the source.
        :param halfsize: flot or list. If int, is the halfsize of the image box and the radius of a circular aperture to get the spectrum
                                      If list, contain the [a,b,theta] parameter for an eliptical aperture. The box will be a square with the major semiaxis
        :param n_fig: Figure number to display the spectrum and the image
        :param mode: str
            Mode for combining spaxels:
              * `ivar` - Inverse variance weighting, variance is taken only spatially, from a "white variance image"
              * `sum` - Sum of total flux
              * `gaussian` - Weighted mean. Weights are obtained from a 2D gaussian fit of the bright profile
              * `wwm` - 'White Weighted Mean'. Weigted mean, weights are obtained from the white image, smoothed using a gaussian filter of sigma = npix. If npix=0, no smooth is done
              * `ivarwv` - Weighted mean, the weight of every pixel is given by the inverse of it's variance
              * `mean`  -  Mean of the total flux
              * `median` - Median of the total flux
              * `wwm_ivarwv' - Weights given by both, `ivarwv` and `wwm`
              * `wwm_ivar` - Weghts given by both, `wwm` and `ivar`
              * `wfrac` - It only takes the fraction `frac` of brightest spaxels (white) in the region
                         (e.g. frac=0.1 means 10% brightest) with equal weight.
        :param frac. Float, default = 0.1
                     Parameter needed for wfrac mode
        :param npix: int. Default = 0
            Standard deviation of the gaussian filter to smooth (Only in wwm methods)m
        :param empirical_std: boolean. Default = False.
            If True, the errors of the spectrum will be determined empirically
        :param save: boolean. Default = False
            If True, the spectrum will be saved in hard_disk
        :param coord_system: str. Default = 'pix'.
            If coord_system = 'wcs' the coordinates will be considered as degrees
        :return: spec: XSpectrum1D object
        """
        spec = self.get_spec_from_ellipse_params(x_c=center[0], y_c=center[1], params=halfsize,
                                                 coord_system=coord_system, mode=mode, frac=frac, npix=npix,
                                                 empirical_std=empirical_std)
        spec = self.spec_to_vacuum(spec)
        if isinstance(halfsize, (int, float)):
            halfsize = [halfsize, halfsize, 0]
        if coord_system == 'wcs':
            x_c, y_c, halfsize = self.ellipse_params_to_pixel(center[0], center[1], params=halfsize)
            center_ = (x_c, y_c)
        else:
            center_ = center
        aux = [halfsize[0], halfsize[1]]
        halfsize = max(aux)
        mini_image = self.get_mini_image(center=center_, halfsize=halfsize)
        plt.figure(n_figure, figsize=(17, 5))
        ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        if coord_system == 'pix':
            x_world, y_world = self.p2w(center[0], center[1])
        else:
            x_world, y_world = center[0], center[1]
        coord = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
        spec_name = name_from_coord(coord)
        if save:
            spec.write_to_fits(spec_name + '.fits')
        plt.title(spec_name)
        w = spec.wavelength.value
        f = spec.flux.value
        ax1.plot(w, f)
        plt.ylabel('Flux (' + str(self.flux_units) + ')')
        plt.xlabel('Wavelength (Angstroms)')
        n = len(w)
        ave = np.nanmean(f)
        std = np.nanstd(f)
        ymin = ave - 3 * std
        ymax = ave + 4 * std
        plt.ylim([ymin, ymax])
        plt.xlim([w[0], w[n - 1]])
        ax2 = plt.subplot2grid((1, 4), (0, 3), colspan=1)
        ax2.imshow(mini_image, cmap='gray', vmin=self.vmin, vmax=self.vmax)
        plt.ylim([0, 2 * halfsize])
        plt.xlim([0, 2 * halfsize])
        return spec

    def draw_region(self, r):
        fig = plt.figure(self.n)
        ax = fig.axes[0]
        patch_list, artist_list = r.get_mpl_patches_texts(origin=0)
        patch = patch_list[0]
        ax.add_patch(patch)

    def region_2dmask(self, r):
        from pyregion.region_to_filter import as_region_filter
        im_aux = np.ones_like(self.white_data)
        hdu_aux = fits.open(self.filename_white)[1]
        hdu_aux.data = im_aux
        shape = hdu_aux.data.shape
        region_filter = as_region_filter(r, origin=0)
        mask_new = region_filter.mask(shape)
        mask_new_inverse = np.where(~mask_new, True, False)
        mask2d = mask_new_inverse
        return mask2d

    def region_3dmask(self, r):
        mask2d = self.region_2dmask(r)
        complete_mask_new = mask2d + self.mask_init
        complete_mask_new = np.where(complete_mask_new != 0, True, False)
        mask3d = complete_mask_new
        return mask3d

    def compute_kinematics(self, x_c, y_c, params, wv_line_vac, wv_range_size=35, type='abs', debug=False, z=0,
                           cmap='seismic'):
        ##Get the integrated spec fit, and estimate the 0 velocity wv from there
        wv_line = wv_line_vac * (1 + z)
        dwmax = 10
        spec_total = self.get_spec_from_ellipse_params(x_c, y_c, params, mode='wwm')
        wv_t = spec_total.wavelength.value
        fl_t = spec_total.flux.value
        sig_t = spec_total.sig.value
        sig_eff = sig_t[np.where(np.logical_and(wv_t >= wv_line - wv_range_size, wv_t <= wv_line + wv_range_size))]
        wv_eff = wv_t[np.where(np.logical_and(wv_t >= wv_line - wv_range_size, wv_t <= wv_line + wv_range_size))]
        fl_eff = fl_t[np.where(np.logical_and(wv_t >= wv_line - wv_range_size, wv_t <= wv_line + wv_range_size))]
        fl_left = fl_eff[:3]
        fl_right = fl_eff[-3:]
        intercept_init = (np.sum(fl_right) + np.sum(fl_left)) / (len(fl_left) + len(fl_right))
        if type == 'abs':
            a_init = np.min(fl_eff) - intercept_init
        if type == 'emi':
            a_init = np.max(fl_eff) - intercept_init
        slope_init = 0
        sigma_init = wv_range_size / 3.
        mean_init = wv_line
        gaussian = models.Gaussian1D(amplitude=a_init, mean=mean_init, stddev=sigma_init)
        line = models.Linear1D(slope=slope_init, intercept=intercept_init)
        model_init = gaussian + line
        fitter = fitting.LevMarLSQFitter()
        model_fit = fitter(model_init, wv_eff, fl_eff, weights=sig_eff / np.sum(sig_eff))
        mean_total = model_fit[0].mean.value
        sigma_total = model_fit[0].stddev.value
        z_line = (mean_total / wv_line_vac) - 1.
        if isinstance(params, (int, float)):
            params = [params, params, 0]

        region_string = self.ellipse_param_to_ds9reg_string(x_c, y_c, params[0], params[1], params[2])
        mask2d = self.get_new_2dmask(region_string)
        ##Find center guessing parameters
        spec_c = self.get_spec_spaxel(x_c, y_c)
        fl_c = spec_c.flux.value
        wv_c = spec_c.wavelength.value
        sig_c = spec_total.sig.value
        sig_eff = sig_c[np.where(np.logical_and(wv_c >= wv_line - wv_range_size, wv_c <= wv_line + wv_range_size))]
        wv_eff = wv_c[np.where(np.logical_and(wv_c >= wv_line - wv_range_size, wv_c <= wv_line + wv_range_size))]
        fl_eff = fl_c[np.where(np.logical_and(wv_c >= wv_line - wv_range_size, wv_c <= wv_line + wv_range_size))]

        #### Define central gaussian_mean
        wv_c_eff = wv_eff
        fl_c_eff = fl_eff
        fl_left = fl_eff[:3]
        fl_right = fl_eff[-3:]
        intercept_init = (np.sum(fl_right) + np.sum(fl_left)) / (len(fl_left) + len(fl_right))
        if type == 'abs':
            a_init = np.min(fl_eff) - intercept_init
        if type == 'emi':
            a_init = np.max(fl_eff) - intercept_init
        slope_init = 0
        sigma_init = sigma_total
        mean_init = wv_line
        gaussian = models.Gaussian1D(amplitude=a_init, mean=mean_init, stddev=sigma_init)
        line = models.Linear1D(slope=slope_init, intercept=intercept_init)
        model_init = gaussian + line
        fitter = fitting.LevMarLSQFitter()
        model_fit = fitter(model_init, wv_eff, fl_eff, weights=sig_eff / np.sum(sig_eff))
        mean_center = model_fit[0].mean.value
        a_center = model_fit[0].amplitude.value
        sigma_center = model_fit[0].stddev.value

        ##get spaxel in mask2d
        y, x = np.where(~mask2d)
        n = len(x)
        kine_im = np.where(self.white_data == 0, np.nan, np.nan)
        sigma_im = np.where(self.white_data == 0, np.nan, np.nan)

        for i in xrange(n):
            print(str(i + 1) + '/' + str(n))
            spec = self.get_spec_spaxel(x[i], y[i])
            wv = spec.wavelength.value
            fl = spec.flux.value
            sig = spec_total.sig.value
            sig_eff = sig[np.where(np.logical_and(wv >= wv_line - wv_range_size, wv <= wv_line + wv_range_size))]
            wv_eff = wv[np.where(np.logical_and(wv >= wv_line - wv_range_size, wv <= wv_line + wv_range_size))]
            fl_eff = fl[np.where(np.logical_and(wv >= wv_line - wv_range_size, wv <= wv_line + wv_range_size))]
            fl_left = fl_eff[:3]
            fl_right = fl_eff[-3:]
            intercept_init = (np.sum(fl_right) + np.sum(fl_left)) / (len(fl_left) + len(fl_right))
            if type == 'abs':
                a_init = np.min(fl_eff) - intercept_init
            if type == 'emi':
                a_init = np.max(fl_eff) - intercept_init
            slope_init = 0
            sigma_init = sigma_center
            mean_init = mean_center
            gaussian = models.Gaussian1D(amplitude=a_init, mean=mean_init, stddev=sigma_init)
            line = models.Linear1D(slope=slope_init, intercept=intercept_init)
            model_init = gaussian + line
            fitter = fitting.LevMarLSQFitter()
            model_fit = fitter(model_init, wv_eff, fl_eff, weights=sig_eff / np.sum(sig_eff))
            m = fitter.fit_info['param_cov']
            residual = model_fit(wv_eff) - fl_eff
            noise = np.std(residual)
            if debug:
                plt.figure()
                plt.plot(wv_c_eff, fl_c_eff, drawstyle='steps-mid', color='grey')
                plt.plot(wv_eff, fl_eff, drawstyle='steps-mid')
                plt.plot(wv_eff, model_fit(wv_eff))
                plt.plot(wv_eff, residual, color='red')
                plt.plot(wv_eff, sig_eff, color='yellow', drawstyle='steps-mid')
                m = fitter.fit_info['param_cov']
                if m != None:
                    print('Display Cov Matrix')
                    plt.figure()
                    plt.imshow(m, interpolation='none', vmin=0, vmax=15)
                    plt.colorbar()
                else:
                    print('Cov Matrix undefined')
            mean = model_fit[0].mean.value
            amp = model_fit[0].amplitude.value
            if abs(amp) >= 2. * noise and (a_center * amp > 0) and abs(mean_center - mean) <= dwmax:
                if debug:
                    print('Fit Aceptado')
                    print(str(x[i]) + ',' + str(y[i]))
                units = u.km / u.s
                vel = ltu.dv_from_z((mean / wv_line_vac) - 1, z_line).to(units).value
                kine_im[y[i]][x[i]] = vel
            else:
                if debug:
                    print('Fit Negado')
                    print(str(x[i]) + ',' + str(y[i]))
            if debug:
                print('value of wv_dif = ' + str(mean_center - mean))
                print('amplitude = ' + str(amp))
                print('noise = ' + str(noise))
                raw_input('Enter to continue...')

        hdulist = self.hdulist_white
        hdulist[1].data = kine_im
        hdulist.writeto('kinematics.fits', clobber=True)
        fig = aplpy.FITSFigure('kinematics.fits', figure=plt.figure())
        fig.show_colorscale(cmap=cmap)
        fig.add_colorbar()
        fig.colorbar.set_axis_label_text('V (km s$^{-1}$)')
        xw, yw = self.p2w(x_c, y_c)
        if isinstance(params, (int, float)):
            r = params * self.pixelsize
        else:
            r = params[0] * self.pixelsize
        r = r.to(u.deg)
        fig.recenter(xw, yw, r.value)
        return kine_im

    def save_muselet_specs(self, filename, mode='sum', params=4, frac=0.1, npix=0, empirical_std=False,
                           redmonster_format=True, ids='all'):
        """

        :param filename: string, Name of the MUSELET output fits table
        :param mode: string, mode of extractor for the spectra
        :param params: int or iterable. Default = 4. Elliptical parameters for the extraction of the spectra in spaxel units
        :param frac: float. Default = 0.1. Extraction parameter used in 'wfrac' mode.
        :param npix: int. Default = 0. Extraction parameter used in several modes. stddev of the Gaussian kernel to smooth
                     the white image. If npix = 0, no smooth is done.
        :param empirical_std: float, Default = False. If True, the stddev of the spectra will be empirically estimated.
        :param redmonster_format: float. Default = True. If True, the spectra will be saved in a rdeable format for Redmonster software.
        :param ids: string or iterable. Default = 'all'. If ids = 'all', all the spectra in the MUSELET table will be extracted.
                    if ids is iterable, it must contain the ids in the MUSELET table of the sources to extract (e.g. ids = [1,15,23] will
                    extract only the sources with the ids 1, 15 and 23)
        :return:
        """

        fits_table = Table.read(fits.open(filename)[1])
        ID = fits_table['ID'].data.data
        RA = fits_table['RA'].data.data
        DEC = fits_table['DEC'].data.data
        if ids == 'all':
            ids = fits_table['ID'].data.data
        n = len(ids)
        for i in xrange(n):
            j = np.where(ids[i] == ID)[0][0]
            x_world = RA[j]
            y_world = DEC[j]
            coord = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
            str_id = str(ids[i]).zfill(3)
            spec_fits_name = str_id + '_' + name_from_coord(coord)
            x, y = self.w2p(x_world, y_world)
            spec = self.get_spec_from_ellipse_params(x, y, params, mode=mode, npix=npix, frac=frac,
                                                     empirical_std=empirical_std, save=False)
            if redmonster_format:
                mcu.spec_to_redmonster_format(spec=spec, fitsname=spec_fits_name + '_RMF.fits', n_id=ids[i])
            else:
                spec.write_to_fits(spec_fits_name + '.fits')
            print('ID = ' + str_id + ' Ready!!')

    def save_ds9regfile_specs(self, regfile, mode='wwm', frac=0.1, npix=0, empirical_std=False, redmonster_format=True,
                              id_start=1, coord_name=False, debug=False):
        """
        Function used to save a set of spectra given by a DS9 regionfile "regfile"
        :param regfile: str. Name of the DS9 region file
        :param mode: str.  Default = 'wwm'. see more modes and details in self.spec_from_minicube_mask()
        :param frac. FLoat, default = 0.1
                     Parameter needed for wfrac mode
        :param npix: int. Default = 0
            Standard deviation of the gaussian filter to smooth (Only in wwm methods)
        :param empirical_std: boolean. Default = False.
            If True, the errors of the spectrum will be determined empirically
        :param redmonster_format: If True, the specta will be saved in a redeable format for redmonster software
        :param coord_name: Boolean. Default = False.
            If True, The name of each spectrum will be computed from the coordinates of the first (X,Y) pair in the region
            string. Otherwhise, the spectra will be named with and ID and the name of the region file.
        :param id_start: int. Default = 1
                Initial id assigned to diferent spectra
        """
        r = pyregion.open(regfile)
        n = len(r)
        self.reload_canvas()
        for i in xrange(n):
            id_ = id_start + i
            r_i = pyregion.ShapeList([r[i]])
            self.draw_region(r_i)
            mask3d = self.region_3dmask(r_i)
            ##Get spec
            spec = self.spec_from_minicube_mask(mask3d, mode=mode, npix=npix, frac=frac)
            if empirical_std:
                spec = mcu.calculate_empirical_rms(spec)
            spec = self.spec_to_vacuum(spec)
            str_id = str(id_).zfill(3)
            spec_fits_name = str_id + '_' + regfile[:-4]
            if coord_name:
                r_aux = r[i]
                x = r_aux.coord_list[0]
                y = r_aux.coord_list[1]
                x_world, y_world = self.p2w(x, y)
                coord = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
                spec_fits_name = str_id + '_' + name_from_coord(coord)
            if redmonster_format:
                if debug:
                    mag_tuple = ['mag_r', '-']
                else:
                    mag_tuple = None
                mcu.spec_to_redmonster_format(spec=spec, fitsname=spec_fits_name + '_RMF.fits', n_id=id_, mag=mag_tuple)
            else:
                spec.write_to_fits(spec_fits_name + '.fits')
            print('ID = ' + str_id + ' Ready!!')

    def get_spec_from_ds9regfile(self, regfile, mode='wwm', i=0, frac=0.1, npix=0, empirical_std=False, n_figure=2,
                                 save=False):
        """
        Function to get the spec of a region defined in a ds9 .reg file
        The .reg file MUST be in physical coordiantes
        :param regfile: str. Name of the DS9 region file
        :param mode: str
            Mode for combining spaxels:
              * `ivar` - Inverse variance weighting, variance is taken only spatially, from a "white variance image"
              * `sum` - Sum of total flux
              * `wwm` - 'White Weighted Mean'. Weigted mean, weights are obtained from the white image, smoothed using a gaussian filter of sigma = npix. If npix=0, no smooth is done
              * `ivarwv` - Weighted mean, the weight of every pixel is given by the inverse of it's variance
              * `mean`  -  Mean of the total flux
              * `median` - Median of the total flux
              * `wwm_ivarwv' - Weights given by both, `ivarwv` and `wwm`
              * `wwm_ivar` - Weghts given by both, `wwm` and `ivar`
              * `wfrac` - It only takes the fraction `frac` of brightest spaxels (white) in the region
                         (e.g. frac=0.1 means 10% brightest) with equal weight.
        :param i: int, default = 0
                  Index of the region in the region file. i = 0 corresponds to the first region listed.
        :param frac: Float, default = 0.1
                     Parameter needed for wfrac mode
        :param npix: int. Default = 0
            Standard deviation of the gaussian filter to smooth (Only in wwm methods)
        :param n_figure: int. Default = 2. Figure to display the spectrum
        :param empirical_std: boolean. Default = False.
            If True, the errors of the spectrum will be determined empirically
        :return: spec: XSpectrum1D object
        """
        r = pyregion.open(regfile)
        r = pyregion.ShapeList([r[i]])

        self.draw_region(r)

        mask3d = self.region_3dmask(r)

        spec = self.spec_from_minicube_mask(mask3d, mode=mode, npix=npix, frac=frac)
        if empirical_std:
            spec = mcu.calculate_empirical_rms(spec)
        spec = self.spec_to_vacuum(spec)
        if save:
            spec.write_to_fits(regfile[:-4] + '.fits')

        plt.figure(n_figure)
        plt.plot(spec.wavelength, spec.flux)
        plt.title('Spectrum from ' + regfile)
        plt.xlabel('Angstroms')
        plt.ylabel('Flux (' + str(self.flux_units) + ')')
        return spec

    @property
    def wavelength(self):
        """
        Creates the wavelength array for the spectrum. The values of dw, and limits will depend
        of the data and should be revised.
        :return: w: array[]
                 array which contain an evenly sampled wavelength range
        """
        dw = self.header_1['CD3_3']
        w_ini = self.header_1['CRVAL3']
        N = self.header_1['NAXIS3']
        w_fin = w_ini + (N - 1) * dw
        # w_aux = w_ini + dw*np.arange(0, N) #todo: check whether w_aux and w are the same
        w = np.linspace(w_ini, w_fin, N)
        # print('wavelength in range ' + str(w[0]) + ' to ' + str(w[len(w) - 1]) + ' and dw = ' + str(dw))
        return w

    def __edit_header(self, hdulist, values_list,
                      keywords_list=['CRPIX1', 'CRPIX2', 'CD1_1', 'CD2_2', 'CRVAL1', 'CRVAL2'], hdu=1):
        hdu_element = hdulist[hdu]
        if len(keywords_list) != len(values_list):
            raise ValueError('Dimensions of keywords_list and values-list does not match')
        n = len(values_list)
        for i in xrange(n):
            keyword = keywords_list[i]
            value = values_list[i]
            hdu_element.header[keyword] = value
        # CSYER1=hdu_element.header['CSYER1']
        # hdu_element.header['CSYER1']=1000.0860135214331
        hdulist_edited = hdulist
        hdulist_edited[hdu] = hdu_element
        return hdulist_edited

    def __save2fits(self, fitsname, data_to_save, stat=False, type='cube', n_figure=2, edit_header=[]):
        if type == 'white':
            hdulist = fits.HDUList.fromfile(self.filename_white)
            hdulist[1].data = data_to_save
            if len(edit_header) == 0:
                hdulist.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, figure=plt.figure(n_figure))
                im.show_grayscale()
            elif len(edit_header) == 1:
                values_list = edit_header[0]
                hdulist_edited = self.__edit_header(hdulist, values_list=values_list)
                hdulist_edited.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, figure=plt.figure(n_figure))
                im.show_grayscale()
            elif len(edit_header) == 2:
                values_list = edit_header[0]
                keywords_list = edit_header[1]
                hdulist_edited = self.__edit_header(hdulist, values_list=values_list, keywords_list=keywords_list)
                hdulist_edited.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, figure=plt.figure(n_figure))
                im.show_grayscale()
            elif len(edit_header) == 3:
                values_list = edit_header[0]
                keywords_list = edit_header[1]
                hdu = edit_header[2]
                hdulist_edited = self.__edit_header(hdulist, values_list=values_list, keywords_list=keywords_list,
                                                    hdu=hdu)
                hdulist_edited.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, figure=plt.figure(n_figure))
                im.show_grayscale()

        if type == 'cube':
            hdulist = fits.HDUList.fromfile(self.filename)
            if stat == False:
                hdulist[1].data = data_to_save
            if stat == True:
                hdulist[2].data = data_to_save
            if len(edit_header) == 0:
                hdulist.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, slices=[1], figure=plt.figure(n_figure))
                im.show_grayscale()
            elif len(edit_header) == 1:
                values_list = edit_header[0]
                hdulist_edited = self.__edit_header(hdulist, values_list=values_list)
                hdulist_edited.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, slices=[1], figure=plt.figure(n_figure))
                im.show_grayscale()
            elif len(edit_header) == 2:
                values_list = edit_header[0]
                keywords_list = edit_header[1]
                hdulist_edited = self.__edit_header(hdulist, values_list=values_list, keywords_list=keywords_list)
                hdulist_edited.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, slices=[1], figure=plt.figure(n_figure))
                im.show_grayscale()
            elif len(edit_header) == 3:
                values_list = edit_header[0]
                keywords_list = edit_header[1]
                hdu = edit_header[2]
                hdulist_edited = self.__edit_header(hdulist, values_list=values_list, keywords_list=keywords_list,
                                                    hdu=hdu)
                hdulist_edited.writeto(fitsname, clobber=True)
                im = aplpy.FITSFigure(fitsname, slices=[1], figure=plt.figure(n_figure))
                im.show_grayscale()

    def ellipse_params_to_pixel(self, xc, yc, params):
        """
        Function to transform the parameters of an ellipse from degrees to pixels
        :param xc:
        :param yc:
        :param radius:
        :return:
        """
        a = params[0]
        b = params[1]
        xaux, yaux, a2 = self.xyr_to_pixel(xc, yc, a)
        xc2, yc2, b2 = self.xyr_to_pixel(xc, yc, b)
        params2 = [a2, b2, params[2]]
        return xc2, yc2, params2

    def get_mini_cube_mask_from_region_string(self, region_string):
        """
        Creates a 3D mask where all original masked voxels are masked out,
        plus all voxels associated to spaxels outside the elliptical region
        defined by the given parameters.
        :param region_string: Region defined by ds9 format
        :return: complete_mask_new: a new mask for the cube
        """
        complete_mask_new = self.get_new_3dmask(region_string)
        return complete_mask_new

    def get_mini_cube_mask_from_ellipse_params(self, x_c, y_c, params, coord_system='pix',color='green'):
        """
        Creates a 3D mask where all original masked voxels are masked out,
        plus all voxels associated to spaxels outside the elliptical region
        defined by the given parameters.

        :param x_c: center of the elliptical aperture
        :param y_c: center of the elliptical aperture
        :param params: can be a single radius (float) of an circular aperture, or a (a,b,theta) tuple
        :param coord_system: default: pix, possible values: pix, wcs
        :return: complete_mask_new: a new mask for the cube
        """

        if not isinstance(params, (int, float, tuple, list, np.array)):
            raise ValueError('Not ready for this `radius` type.')

        if isinstance(params, (int, float)):
            a = params
            b = params
            theta = 0
        elif isiterable(params) and (len(params) == 3):
            a = max(params[:2])
            b = min(params[:2])
            theta = params[2]
        else:
            raise ValueError('If iterable, the length of radius must be == 3; otherwise try float.')

        region_string = self.ellipse_param_to_ds9reg_string(x_c, y_c, a, b, theta, coord_system=coord_system,color=color)
        complete_mask_new = self.get_new_3dmask(region_string)
        return complete_mask_new

    def ellipse_param_to_ds9reg_string(self, xc, yc, a, b, theta, color='green', coord_system='pix'):
        """Creates a string that defines an elliptical region given by the
        parameters using the DS9 convention.
        """
        if coord_system == 'wcs':
            x_center, y_center, radius = self.ellipse_params_to_pixel(xc, yc, params=[a, b, theta])
        else:  # already in pixels
            x_center, y_center, radius = xc, yc, [a, b, theta]
        region_string = 'physical;ellipse({},{},{},{},{}) # color = {}'.format(x_center, y_center, radius[0],
                                                                               radius[1],
                                                                               radius[2], color)
        return region_string

    def wfrac_show_spaxels(self, frac, mask2d, smoothed_white):
        y, x = np.where(~mask2d)
        n = len(x)
        im_white = smoothed_white[~mask2d]
        fl_limit = np.percentile(im_white, (1. - frac) * 100.)
        for i in xrange(n):
            if smoothed_white[y[i]][x[i]] >= fl_limit:
                plt.figure(self.n)
                plt.plot(x[i] + 1, y[i] + 1, 'o', color='Blue')

    def _test_3dmask(self, region_string, alpha=0.8, slice=0):
        complete_mask = self.get_new_3dmask(region_string)
        mask_slice = complete_mask[int(slice)]
        plt.figure(self.n)
        plt.imshow(mask_slice, alpha=alpha)
        self.draw_pyregion(region_string)

    def get_new_2dmask(self, region_string):
        """Creates a 2D mask for the white image that mask out spaxel that are outside
        the region defined by region_string"""

        from pyregion.region_to_filter import as_region_filter
        im_aux = np.ones_like(self.white_data)
        hdu_aux = fits.open(self.filename_white)[1]
        hdu_aux.data = im_aux
        hdulist = self.hdulist_white
        r = pyregion.parse(region_string).as_imagecoord(hdulist[1].header)
        shape = hdu_aux.data.shape
        region_filter = as_region_filter(r, origin=0)
        mask_new = region_filter.mask(shape)
        mask_new_inverse = np.where(~mask_new, True, False)
        return mask_new_inverse

    def get_new_3dmask(self, region_string):
        """Creates a 3D mask for the cube that also mask out
        spaxels that are outside the gemoetrical redion defined by
        region_string.

        Parameters
        ----------
        region_string : str
            A string that defines a geometrical region using the
            DS9 format (e.g. see http://ds9.si.edu/doc/ref/region.html)

        Returns
        -------
        A 3D mask that includes already masked voxels from the original cube,
        plus all spaxels outside the region defined by region_string.

        Notes: It uses pyregion package.

        """
        mask2d = self.get_new_2dmask(region_string)
        complete_mask_new = mask2d + self.mask_init
        complete_mask_new = np.where(complete_mask_new != 0, True, False)
        self.draw_pyregion(region_string)
        return complete_mask_new

    def plot_sextractor_regions(self, sextractor_filename, a_min=3.5, flag_threshold=32, wcs_coords=False, n_id=None, border_thresh=1):
        self.reload_canvas()
        x_pix = np.array(self.get_from_table(sextractor_filename, 'X_IMAGE'))
        y_pix = np.array(self.get_from_table(sextractor_filename, 'Y_IMAGE'))
        a = np.array(self.get_from_table(sextractor_filename, 'A_IMAGE'))
        a_new = np.where(a < a_min, a_min, a)
        b = np.array(self.get_from_table(sextractor_filename, 'B_IMAGE'))
        ratios = a / b
        b_new = a_new / ratios
        b_new = np.where(b_new < 1, 1, b_new)
        a = a_new
        b = b_new
        theta = np.array(self.get_from_table(sextractor_filename, 'THETA_IMAGE'))
        flags = self.get_from_table(sextractor_filename, 'FLAGS').data
        id = self.get_from_table(sextractor_filename, 'NUMBER').data
        mag = self.get_from_table(sextractor_filename, 'MAG_AUTO').data
        n = len(x_pix)
        if wcs_coords:
            x_world = np.array(self.get_from_table(sextractor_filename, 'X_WORLD'))
            y_world = np.array(self.get_from_table(sextractor_filename, 'Y_WORLD'))
            a_world = np.array(self.get_from_table(sextractor_filename, 'A_WORLD'))
            b_world = np.array(self.get_from_table(sextractor_filename, 'B_WORLD'))
            a_min_wcs = a_min * self.pixelsize
            a_min_wcs = a_min_wcs.to(u.deg).value
            a_world_new = np.where(a_world < a_min_wcs, a_min_wcs, a_world)
            ratios_wcs = a_world / b_world
            b_world_new = a_world_new / ratios_wcs
            b_world_new = np.where(b_world_new < self.pixelsize.to(u.deg).value, self.pixelsize.to(u.deg).value,
                                   b_world_new)
            a_world = a_world_new
            b_world = b_world_new
            for i in xrange(n):
                params_wcs = [a_world[i], b_world[i], theta[i]]
                x_pix[i], y_pix[i], params = self.ellipse_params_to_pixel(x_world[i], y_world[i], params=params_wcs)
                a[i] = params[0]
                b[i] = params[1]
            x2=[]
            y2=[]
            a2=[]
            b2=[]
            theta2=[]
            flags2=[]
            id2=[]
            mag2=[]
            ly,lx=self.white_data.shape
            for i in xrange(n):
                if x_pix[i]>=border_thresh and y_pix[i]>=border_thresh and x_pix[i]<=lx-border_thresh and y_pix[i]<=ly-border_thresh:
                    x2.append(x_pix[i])
                    y2.append(y_pix[i])
                    a2.append(a[i])
                    b2.append(b[i])
                    theta2.append(theta[i])
                    flags2.append(flags[i])
                    id2.append(id[i])
                    mag2.append(mag[i])
            x_pix=np.array(x2)
            y_pix=np.array(y2)
            a=np.array(a2)
            b=np.array(b2)
            theta=np.array(theta2)
            flags=np.array(flags2)
            id=np.array(id2)
            mag=np.array(mag2)
            n=len(x_pix)



        if n_id != None:
            j = np.where(id == n_id)[0][0]
            region_string = self.ellipse_param_to_ds9reg_string(x_pix[j], y_pix[j], a[j], b[j], theta[j], color='Green')
            self.draw_pyregion(region_string)
            plt.text(x_pix[j], y_pix[j], id[j], color='Red')
            return


        for i in xrange(n):
            color = 'Green'
            if flags[i] > flag_threshold:
                color = 'Red'
            region_string = self.ellipse_param_to_ds9reg_string(x_pix[i], y_pix[i], a[i], b[i], theta[i], color=color)
            self.draw_pyregion(region_string)
            plt.text(x_pix[i], y_pix[i], id[i], color='Red')
        return x_pix, y_pix, a, b, theta, flags, id, mag

    def save_sextractor_specs(self, sextractor_filename, flag_threshold=32, redmonster_format=True, a_min=3.5,
                              n_figure=2, wcs_coords=False,
                              mode='wwm', mag_kwrd='mag_r', npix=0, frac=0.1, border_thresh=1):
        x_pix, y_pix, a, b, theta, flags, id, mag = self.plot_sextractor_regions(
            sextractor_filename=sextractor_filename, a_min=a_min,
            flag_threshold=flag_threshold, wcs_coords=wcs_coords, border_thresh=border_thresh)
        self.reload_canvas()
        n = len(x_pix)
        for i in xrange(n):
            if flags[i] <= flag_threshold:
                x_world, y_world = self.p2w(x_pix[i], y_pix[i])
                coord = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
                spec_fits_name = name_from_coord(coord)
                spec = self.get_spec_from_ellipse_params(x_c=x_pix[i], y_c=y_pix[i], params=[a[i], b[i], theta[i]],
                                                         mode=mode, npix=npix, frac=frac, save=False, n_figure=n_figure)

                str_id = str(id[i]).zfill(3)
                spec_fits_name = str_id + '_' + spec_fits_name
                if redmonster_format:
                    mcu.spec_to_redmonster_format(spec=spec, fitsname=spec_fits_name + '_RMF.fits', n_id=id[i],
                                                  mag=[mag_kwrd, mag[i]])
                else:
                    spec.write_to_fits(spec_fits_name + '.fits')
                    hdulist = fits.open(spec_fits_name + '.fits')
                    hdulist[0].header[mag_kwrd] = mag[i]
                    hdulist.writeto(spec_fits_name + '.fits', clobber=True)
                print('ID = ' + str_id + ' Ready!!')

    def __read_files(self, input):
        path = input
        files = glob.glob(path)
        return files

    def create_movie_wavelength_range(self, initial_wavelength, final_wavelength, width=5., outvid='wave_video.avi',
                                      erase=True):
        """
        Function to create a film over a wavelength range of the cube
        :param initial_wavelength: initial wavelength of the film
        :param final_wavelength:  final wavelength of the film
        :param width: width of the wavelength range in each frame
        :param outvid: name of the final video
        :param erase: if True, the individual frames will be erased after producing the video
        :return:
        """
        wave = self.wavelength
        n = len(wave)
        w_max = wave[n - 1] - width - 1
        if initial_wavelength < wave[0]:
            print(str(
                initial_wavelength) + ' es menor al limite inferior minimo permitido, se usara en su lugar ' + str(
                wave[0]))
            initial_wavelength = wave[0]
        if final_wavelength > wave[n - 1]:
            print(str(final_wavelength) + ' es mayor al limite superior maximo permitido, se usara en su lugar ' + str(
                w_max))
            final_wavelength = w_max
        if final_wavelength <= wave[0] or initial_wavelength >= wave[n - 1]:
            raise ValueError('Input wavelength is not in valid range')

        images_names = []
        fitsnames = []
        for i in xrange(initial_wavelength, final_wavelength):
            wavelength_range = (i, i + width)
            filename = 'colapsed_image_' + str(i) + '_'
            im = self.get_image(wv_input=[wavelength_range], fitsname=filename + '.fits', type='sum', save='True')
            plt.close(15)
            image = aplpy.FITSFigure(filename + '.fits', figure=plt.figure(15))
            image.show_grayscale()
            image.save(filename=filename + '.png')
            fitsnames.append(filename + '.fits')
            images_names.append(filename + '.png')
            plt.close(15)
        video = self.make_video(images=images_names, outvid=outvid)
        n_im = len(fitsnames)
        if erase:
            for i in xrange(n_im):
                fits_im = fitsnames[i]
                png_im = images_names[i]
                command_fits = 'rm ' + fits_im
                command_png = 'rm ' + png_im
                os.system(command_fits)
                os.system(command_png)
        return video

    def find_wv_inds(self, wv_array):
        """

        :param wv_array
        :return: Returns the indices in the cube, that are closest to wv_array
        """
        inds = [np.argmin(np.fabs(wv_ii - self.wavelength)) for wv_ii in wv_array]
        inds = np.unique(inds)
        return inds

    def sub_cube(self, wv_input, stat=False):
        """
        Returns a cube-like object with fewer wavelength elements

        :param wv_input: tuple or np.array
        :return: XXXX
        """
        if isinstance(wv_input[0], (tuple, list, np.ndarray)):
            if len(wv_input[0]) != 2:
                raise ValueError(
                    "If wv_input is given as tuple, it must be of lenght = 2, interpreted as (wv_min, wv_max)")
            wv_inds = self.find_wv_inds(wv_input[0])
            ind_min = np.min(wv_inds)
            ind_max = np.max(wv_inds)
            if stat:
                sub_cube = self.stat[ind_min:ind_max + 1, :, :]
            else:
                sub_cube = self.cube[ind_min:ind_max + 1, :, :]
        else:  # assuming array-like for wv_input
            wv_inds = self.find_wv_inds(wv_input)
            if stat:
                sub_cube = self.stat[wv_inds, :, :]
            else:
                sub_cube = self.cube[wv_inds, :, :]
        return sub_cube

    def get_filtered_image(self, _filter='r', save=True, n_figure=5, custom_filter=None):
        """
        Function used to produce a filtered image from the cube
        :param _filter: string, default = r
                        possible values: u,g,r,i,z , sdss filter or Johnson V,r to get the new image
        :param save: Boolean, default = True
                     If True, the image will be saved
        :param custom_filter: Default = None.
                              If not, can be a customized filter created by the user formated as [wc,fc],
                              where the first element is the wavelength array of the filter and the second is the
                              corresponding transmission curve.
        :return:
        """

        w = self.wavelength
        if not custom_filter:
            filter_curve = self.get_filter(wavelength_spec=w, _filter=_filter)
        else:
            wave_filter = custom_filter[0]
            flux_filter = custom_filter[1]
            filter_curve = self.filter_to_MUSE_wavelength(wave_filter, flux_filter, wavelength_spec=w)

        condition = np.where(filter_curve > 0)[0]
        fitsname = 'new_image_' + _filter + '_filter.fits'
        sub_cube = self.cube[condition]
        filter_curve_final = filter_curve[condition]
        extra_dims = sub_cube.ndim - filter_curve_final.ndim
        new_shape = filter_curve_final.shape + (1,) * extra_dims
        new_filter_curve = filter_curve_final.reshape(new_shape)
        new_filtered_cube = sub_cube * new_filter_curve
        new_filtered_image = np.sum(new_filtered_cube, axis=0)
        if save:
            self.__save2fits(fitsname, new_filtered_image.data, type='white', n_figure=n_figure)
        return new_filtered_image

    def get_image(self, wv_input, fitsname='new_collapsed_cube.fits', type='sum', n_figure=2, save=False, stat=False,
                  maskfile=None, inverse_mask=True):
        """
        Function used to colapse a determined wavelength range in a sum or a median type
        :param wv_input: tuple or list
                         can be a list of wavelengths or a tuple that will represent a  range
        :param fitsname: str
                         The name of the fits that will contain the new image
        :param type: str, possible values: 'sum' or 'median'
                     The type of combination that will be done.
        :param n_figure: int
                         Figure to display the new image if it is saved
        :return:
        """
        if maskfile:
            r = pyregion.open(maskfile)
            n = len(r)
            masks = []
            for i in xrange(n):
                masks.append(self.region_2dmask(pyregion.ShapeList([r[i]])))

            mask_final = masks[0]
            for i in xrange(n):
                mask_final = np.logical_and(mask_final, masks[i])
            if inverse_mask:
                mask_final = np.where(~mask_final, True, False)

        sub_cube = self.sub_cube(wv_input, stat=stat)
        if type == 'sum':
            matrix_flat = np.sum(sub_cube, axis=0)
        elif type == 'median':
            matrix_flat = np.median(sub_cube, axis=0)
        else:
            raise ValueError('Unknown type, please chose sum or median')
        if maskfile:
            matrix_flat = np.where(mask_final == 1, matrix_flat, np.nan)
            if save:
                self.__save2fits(fitsname, matrix_flat, type='white', n_figure=n_figure)

        else:
            if save:
                self.__save2fits(fitsname, matrix_flat.data, type='white', n_figure=n_figure)

        return matrix_flat

    def get_continuum_range(self, range):
        """

        :param range: tuple
                      contain the range of a emission line. Continuum will be computed around this range
        :return: cont_range_inf: The continuum range at the left of the Emission line, same length than input range
                 cont_range_sup: The continuum range at the right of the Emission line, same length than input range
                 n             : The number of element in the wavelength space inside the ranges
        """
        wv_inds = self.find_wv_inds(range)
        n = wv_inds[1] - wv_inds[0]
        wv_inds_sup = wv_inds + n
        wv_inds_inf = wv_inds - n
        cont_range_inf = self.wavelength[wv_inds_inf]
        cont_range_sup = self.wavelength[wv_inds_sup]
        return cont_range_inf, cont_range_sup, n

    def get_image_wv_ranges(self, wv_ranges, substract_cont=True, fitsname='new_collapsed_cube.fits', save=False,
                            n_figure=3):
        image_stacker = np.zeros_like(self.white_data)
        for r in wv_ranges:
            image = self.get_image([r])
            cont_range_inf, cont_range_sup, n = self.get_continuum_range(r)
            cont_inf_image = self.get_image([cont_range_inf], type='median')
            cont_sup_image = self.get_image([cont_range_sup], type='median')
            cont_image = (n + 1) * (cont_inf_image + cont_sup_image) / 2.
            if substract_cont:
                image = image - cont_image
            image_stacker = image_stacker + image.data
        image_stacker = np.where(image_stacker < 0, 0, image_stacker)
        if save:
            self.__save2fits(fitsname, image_stacker, type='white', n_figure=n_figure)
        return image_stacker

    def create_white(self, new_white_fitsname='white_from_colapse.fits', stat=False, save=True):
        """
        Function that collapses all wavelengths available to produce a new white image
        :param new_white_fitsname: Name of the new image
        :return:
        """
        wave = self.wavelength
        n = len(wave)
        wv_input = [[wave[0], wave[n - 1]]]
        white_image = self.get_image(wv_input, fitsname=new_white_fitsname, stat=stat, save=save)
        return white_image

    def calculate_mag(self, wavelength, flux, _filter, zeropoint_flux=9.275222661263278e-07):
        dw = np.diff(wavelength)
        new_flux = flux * _filter
        f_mean = (new_flux[:-1] + new_flux[1:]) * 0.5
        total_flux = np.sum(f_mean * dw) * self.flux_units.value
        mag = -2.5 * np.log10(total_flux / zeropoint_flux)
        return mag

    def get_filter(self, wavelength_spec, _filter='r'):

        wave_u = np.arange(2980, 4155, 25)

        wave_g = np.arange(3630, 5855, 25)

        wave_r = np.arange(5380, 7255, 25)

        wave_i = np.arange(6430, 8655, 25)

        wave_z = np.arange(7730, 11255, 25)

        wave_R = np.array([5445., 5450., 5455., 5460., 5465., 5470., 5475., 5480.,
                           5485., 5490., 5495., 5500., 5505., 5510., 5515., 5520.,
                           5525., 5530., 5535., 5540., 5545., 5550., 5555., 5560.,
                           5565., 5570., 5575., 5580., 5585., 5590., 5595., 5600.,
                           5605., 5610., 5615., 5620., 5625., 5630., 5635., 5640.,
                           5645., 5650., 5655., 5660., 5665., 5670., 5675., 5680.,
                           5685., 5690., 5695., 5700., 5705., 5710., 5715., 5720.,
                           5725., 5730., 5735., 5740., 5745., 5750., 5755., 5760.,
                           5765., 5770., 5775., 5780., 5785., 5790., 5795., 5800.,
                           5805., 5810., 5815., 5820., 5825., 5830., 5835., 5840.,
                           5845., 5850., 5855., 5860., 5865., 5870., 5875., 5880.,
                           5885., 5890., 5895., 5900., 5905., 5910., 5915., 5920.,
                           5925., 5930., 5935., 5940., 5945., 5950., 5955., 5960.,
                           5965., 5970., 5975., 5980., 5985., 5990., 5995., 6000.,
                           6005., 6010., 6015., 6020., 6025., 6030., 6035., 6040.,
                           6045., 6050., 6055., 6060., 6065., 6070., 6075., 6080.,
                           6085., 6090., 6095., 6100., 6105., 6110., 6115., 6120.,
                           6125., 6130., 6135., 6140., 6145., 6150., 6155., 6160.,
                           6165., 6170., 6175., 6180., 6185., 6190., 6195., 6200.,
                           6205., 6210., 6215., 6220., 6225., 6230., 6235., 6240.,
                           6245., 6250., 6255., 6260., 6265., 6270., 6275., 6280.,
                           6285., 6290., 6295., 6300., 6305., 6310., 6315., 6320.,
                           6325., 6330., 6335., 6340., 6345., 6350., 6355., 6360.,
                           6365., 6370., 6375., 6380., 6385., 6390., 6395., 6400.,
                           6405., 6410., 6415., 6420., 6425., 6430., 6435., 6440.,
                           6445., 6450., 6455., 6460., 6465., 6470., 6475., 6480.,
                           6485., 6490., 6495., 6500., 6505., 6510., 6515., 6520.,
                           6525., 6530., 6535., 6540., 6545., 6550., 6555., 6560.,
                           6565., 6570., 6575., 6580., 6585., 6590., 6595., 6600.,
                           6605., 6610., 6615., 6620., 6625., 6630., 6635., 6640.,
                           6645., 6650., 6655., 6660., 6665., 6670., 6675., 6680.,
                           6685., 6690., 6695., 6700., 6705., 6710., 6715., 6720.,
                           6725., 6730., 6735., 6740., 6745., 6750., 6755., 6760.,
                           6765., 6770., 6775., 6780., 6785., 6790., 6795., 6800.,
                           6805., 6810., 6815., 6820., 6825., 6830., 6835., 6840.,
                           6845., 6850., 6855., 6860., 6865., 6870., 6875., 6880.,
                           6885., 6890., 6895., 6900., 6905., 6910., 6915., 6920.,
                           6925., 6930., 6935., 6940., 6945., 6950., 6955., 6960.,
                           6965., 6970., 6975., 6980., 6985., 6990., 6995., 7000.,
                           7005., 7010., 7015., 7020., 7025., 7030., 7035., 7040.,
                           7045., 7050., 7055., 7060., 7065., 7070., 7075., 7080.,
                           7085., 7090., 7095., 7100., 7105., 7110., 7115., 7120.,
                           7125., 7130., 7135., 7140., 7145., 7150., 7155., 7160.,
                           7165., 7170., 7175., 7180., 7185., 7190., 7195., 7200.,
                           7205., 7210., 7215., 7220., 7225., 7230., 7235., 7240.,
                           7245., 7250., 7255., 7260., 7265., 7270., 7275., 7280.,
                           7285., 7290., 7295., 7300., 7305., 7310., 7315., 7320.,
                           7325., 7330., 7335., 7340., 7345., 7350., 7355., 7360.,
                           7365., 7370., 7375., 7380., 7385., 7390., 7395., 7400.,
                           7405., 7410., 7415., 7420., 7425., 7430., 7435., 7440.,
                           7445., 7450., 7455., 7460., 7465., 7470., 7475., 7480.,
                           7485., 7490., 7495., 7500., 7505., 7510., 7515., 7520.,
                           7525., 7530., 7535., 7540., 7545., 7550., 7555., 7560.,
                           7565., 7570., 7575., 7580., 7585., 7590., 7595., 7600.,
                           7605., 7610., 7615., 7620., 7625., 7630., 7635., 7640.,
                           7645., 7650., 7655., 7660., 7665., 7670., 7675., 7680.,
                           7685., 7690., 7695., 7700., 7705., 7710., 7715., 7720.,
                           7725., 7730., 7735., 7740., 7745., 7750., 7755., 7760.,
                           7765., 7770., 7775., 7780., 7785., 7790., 7795., 7800.,
                           7805., 7810., 7815., 7820., 7825., 7830., 7835., 7840.,
                           7845., 7850., 7855., 7860., 7865., 7870., 7875., 7880.,
                           7885., 7890., 7895., 7900., 7905., 7910., 7915., 7920.,
                           7925., 7930., 7935., 7940., 7945., 7950., 7955., 7960.,
                           7965., 7970., 7975., 7980., 7985., 7990., 7995., 8000.,
                           8005., 8010., 8015., 8020., 8025., 8030., 8035., 8040.,
                           8045., 8050., 8055., 8060., 8065., 8070., 8075., 8080.,
                           8085., 8090., 8095., 8100., 8105., 8110., 8115., 8120.,
                           8125., 8130., 8135., 8140., 8145., 8150., 8155., 8160.,
                           8165., 8170., 8175., 8180., 8185., 8190., 8195., 8200.,
                           8205., 8210., 8215., 8220., 8225., 8230., 8235., 8240.,
                           8245., 8250., 8255., 8260., 8265., 8270., 8275., 8280.,
                           8285., 8290., 8295., 8300., 8305., 8310., 8315., 8320.,
                           8325., 8330., 8335., 8340., 8345., 8350., 8355., 8360.,
                           8365., 8370., 8375., 8380., 8385., 8390., 8395., 8400.,
                           8405., 8410., 8415., 8420., 8425., 8430., 8435., 8440.,
                           8445., 8450., 8455., 8460., 8465., 8470., 8475., 8480.,
                           8485., 8490., 8495., 8500., 8505., 8510., 8515., 8520.,
                           8525., 8530., 8535., 8540., 8545., 8550., 8555., 8560.,
                           8565., 8570., 8575., 8580., 8585., 8590., 8595., 8600.,
                           8605., 8610., 8615., 8620., 8625., 8630., 8635., 8640.,
                           8645., 8650., 8655., 8660., 8665., 8670., 8675., 8680.,
                           8685., 8690., 8695., 8700., 8705., 8710., 8715., 8720.,
                           8725., 8730., 8735., 8740., 8745., 8750., 8755., 8760.,
                           8765., 8770., 8775., 8780., 8785., 8790., 8795., 8800.,
                           8805., 8810., 8815., 8820., 8825., 8830., 8835., 8840.,
                           8845., 8850., 8855., 8860., 8865., 8870., 8875., 8880.,
                           8885., 8890., 8895., 8900., 8905., 8910., 8915., 8920.,
                           8925., 8930., 8935., 8940., 8945., 8950., 8955., 8960.,
                           8965., 8970., 8975., 8980., 8985., 8990., 8995., 9000.,
                           9005., 9010., 9015., 9020., 9025., 9030., 9035., 9040.,
                           9045., 9050., 9055., 9060., 9065., 9070., 9075., 9080.,
                           9085., 9090., 9095., 9100., 9105., 9110., 9115., 9120.,
                           9125., 9130., 9135., 9140., 9145., 9150., 9155., 9160.,
                           9165., 9170., 9175., 9180., 9185., 9190., 9195., 9200.,
                           9205., 9210., 9215., 9220., 9225., 9230., 9235., 9240.,
                           9245., 9250., 9255., 9260., 9265., 9270., 9275., 9280.,
                           9285., 9290., 9295., 9300., 9305., 9310., 9315., 9320.,
                           9325., 9330., 9335., 9340.])

        wave_V = np.array([4760., 4765., 4770., 4775., 4780., 4785., 4790., 4795.,
                           4800., 4805., 4810., 4815., 4820., 4825., 4830., 4835.,
                           4840., 4845., 4850., 4855., 4860., 4865., 4870., 4875.,
                           4880., 4885., 4890., 4895., 4900., 4905., 4910., 4915.,
                           4920., 4925., 4930., 4935., 4940., 4945., 4950., 4955.,
                           4960., 4965., 4970., 4975., 4980., 4985., 4990., 4995.,
                           5000., 5005., 5010., 5015., 5020., 5025., 5030., 5035.,
                           5040., 5045., 5050., 5055., 5060., 5065., 5070., 5075.,
                           5080., 5085., 5090., 5095., 5100., 5105., 5110., 5115.,
                           5120., 5125., 5130., 5135., 5140., 5145., 5150., 5155.,
                           5160., 5165., 5170., 5175., 5180., 5185., 5190., 5195.,
                           5200., 5205., 5210., 5215., 5220., 5225., 5230., 5235.,
                           5240., 5245., 5250., 5255., 5260., 5265., 5270., 5275.,
                           5280., 5285., 5290., 5295., 5300., 5305., 5310., 5315.,
                           5320., 5325., 5330., 5335., 5340., 5345., 5350., 5355.,
                           5360., 5365., 5370., 5375., 5380., 5385., 5390., 5395.,
                           5400., 5405., 5410., 5415., 5420., 5425., 5430., 5435.,
                           5440., 5445., 5450., 5455., 5460., 5465., 5470., 5475.,
                           5480., 5485., 5490., 5495., 5500., 5505., 5510., 5515.,
                           5520., 5525., 5530., 5535., 5540., 5545., 5550., 5555.,
                           5560., 5565., 5570., 5575., 5580., 5585., 5590., 5595.,
                           5600., 5605., 5610., 5615., 5620., 5625., 5630., 5635.,
                           5640., 5645., 5650., 5655., 5660., 5665., 5670., 5675.,
                           5680., 5685., 5690., 5695., 5700., 5705., 5710., 5715.,
                           5720., 5725., 5730., 5735., 5740., 5745., 5750., 5755.,
                           5760., 5765., 5770., 5775., 5780., 5785., 5790., 5795.,
                           5800., 5805., 5810., 5815., 5820., 5825., 5830., 5835.,
                           5840., 5845., 5850., 5855., 5860., 5865., 5870., 5875.,
                           5880., 5885., 5890., 5895., 5900., 5905., 5910., 5915.,
                           5920., 5925., 5930., 5935., 5940., 5945., 5950., 5955.,
                           5960., 5965., 5970., 5975., 5980., 5985., 5990., 5995.,
                           6000., 6005., 6010., 6015., 6020., 6025., 6030., 6035.,
                           6040., 6045., 6050., 6055., 6060., 6065., 6070., 6075.,
                           6080., 6085., 6090., 6095., 6100., 6105., 6110., 6115.,
                           6120., 6125., 6130., 6135., 6140., 6145., 6150., 6155.,
                           6160., 6165., 6170., 6175., 6180., 6185., 6190., 6195.,
                           6200., 6205., 6210., 6215., 6220., 6225., 6230., 6235.,
                           6240., 6245., 6250., 6255., 6260., 6265., 6270., 6275.,
                           6280., 6285., 6290., 6295., 6300., 6305., 6310., 6315.,
                           6320., 6325., 6330., 6335., 6340., 6345., 6350., 6355.,
                           6360., 6365., 6370., 6375., 6380., 6385., 6390., 6395.,
                           6400., 6405., 6410., 6415., 6420., 6425., 6430., 6435.,
                           6440., 6445., 6450., 6455., 6460., 6465., 6470., 6475.,
                           6480., 6485., 6490., 6495., 6500., 6505., 6510., 6515.,
                           6520., 6525., 6530., 6535., 6540., 6545., 6550., 6555.,
                           6560., 6565., 6570., 6575., 6580., 6585., 6590., 6595.,
                           6600., 6605., 6610., 6615., 6620., 6625., 6630., 6635.,
                           6640., 6645., 6650., 6655., 6660., 6665., 6670., 6675.,
                           6680., 6685., 6690., 6695., 6700., 6705., 6710., 6715.,
                           6720., 6725., 6730., 6735., 6740., 6745., 6750., 6755.,
                           6760., 6765., 6770., 6775., 6780., 6785., 6790., 6795.,
                           6800., 6805., 6810., 6815., 6820., 6825., 6830., 6835.,
                           6840., 6845., 6850., 6855., 6860., 6865., 6870., 6875.,
                           6880., 6885., 6890., 6895., 6900., 6905., 6910., 6915.,
                           6920., 6925., 6930., 6935., 6940., 6945., 6950., 6955.,
                           6960., 6965., 6970., 6975., 6980., 6985., 6990., 6995.,
                           7000., 7005., 7010., 7015., 7020., 7025., 7030., 7035.,
                           7040., 7045., 7050., 7055., 7060., 7065., 7070., 7075.,
                           7080., 7085., 7090., 7095., 7100., 7105., 7110., 7115.,
                           7120., 7125., 7130., 7135., 7140., 7145., 7150., 7155.,
                           7160., 7165., 7170., 7175., 7180., 7185., 7190., 7195.,
                           7200., 7205., 7210., 7215., 7220., 7225., 7230., 7235.,
                           7240., 7245., 7250., 7255., 7260., 7265., 7270., 7280.])

        flux_V = np.array([9.64320839e-03, 1.17108273e-02, 1.43528032e-02,
                           1.75631618e-02, 2.11335897e-02, 2.55253673e-02,
                           3.07395792e-02, 3.66303658e-02, 4.38177156e-02,
                           5.15626001e-02, 6.09055328e-02, 7.15601015e-02,
                           8.32171154e-02, 9.64917278e-02, 1.11321487e-01,
                           1.27047434e-01, 1.45095301e-01, 1.63879433e-01,
                           1.84025288e-01, 2.05674400e-01, 2.27541790e-01,
                           2.51783009e-01, 2.76728153e-01, 3.02018051e-01,
                           3.28636360e-01, 3.54072228e-01, 3.81254387e-01,
                           4.08208084e-01, 4.34315758e-01, 4.61384430e-01,
                           4.87483635e-01, 5.12711716e-01, 5.38157120e-01,
                           5.61274338e-01, 5.85662842e-01, 6.07098885e-01,
                           6.29042625e-01, 6.51120758e-01, 6.71111679e-01,
                           6.87856445e-01, 7.05869598e-01, 7.21706085e-01,
                           7.38656692e-01, 7.51982346e-01, 7.66451569e-01,
                           7.79320374e-01, 7.91537857e-01, 8.01387253e-01,
                           8.12644043e-01, 8.21886444e-01, 8.30849152e-01,
                           8.39123459e-01, 8.45743408e-01, 8.53470001e-01,
                           8.60292893e-01, 8.66531220e-01, 8.72752762e-01,
                           8.77110748e-01, 8.82006912e-01, 8.87016678e-01,
                           8.91045380e-01, 8.94107590e-01, 8.97235336e-01,
                           9.00786133e-01, 9.03548050e-01, 9.06549301e-01,
                           9.08831177e-01, 9.11690445e-01, 9.12861023e-01,
                           9.15185928e-01, 9.17089386e-01, 9.17668686e-01,
                           9.20558548e-01, 9.21113205e-01, 9.22701874e-01,
                           9.23237000e-01, 9.24772034e-01, 9.25894012e-01,
                           9.26325073e-01, 9.27905960e-01, 9.27411652e-01,
                           9.28828430e-01, 9.28686295e-01, 9.30086288e-01,
                           9.29822846e-01, 9.30881195e-01, 9.30577240e-01,
                           9.31094971e-01, 9.30789261e-01, 9.30882034e-01,
                           9.31607895e-01, 9.31012649e-01, 9.30543594e-01,
                           9.30507584e-01, 9.30894165e-01, 9.30728226e-01,
                           9.30551834e-01, 9.30233002e-01, 9.30283814e-01,
                           9.30285187e-01, 9.29119644e-01, 9.28713150e-01,
                           9.28867035e-01, 9.28172684e-01, 9.28012314e-01,
                           9.27614441e-01, 9.26771698e-01, 9.26360092e-01,
                           9.25508957e-01, 9.24991302e-01, 9.24198074e-01,
                           9.22970123e-01, 9.22512283e-01, 9.21908951e-01,
                           9.20856094e-01, 9.20415039e-01, 9.19665604e-01,
                           9.18579636e-01, 9.17498093e-01, 9.16515350e-01,
                           9.15503616e-01, 9.14212112e-01, 9.13366013e-01,
                           9.12551498e-01, 9.11715393e-01, 9.10380325e-01,
                           9.09479599e-01, 9.07609863e-01, 9.06777115e-01,
                           9.05421143e-01, 9.04353409e-01, 9.02455139e-01,
                           9.00539398e-01, 9.00131378e-01, 8.98344574e-01,
                           8.96168747e-01, 8.94843826e-01, 8.92673111e-01,
                           8.91329804e-01, 8.90147629e-01, 8.88428879e-01,
                           8.87021027e-01, 8.85309372e-01, 8.83131332e-01,
                           8.81392059e-01, 8.78589477e-01, 8.76842956e-01,
                           8.75344315e-01, 8.73290176e-01, 8.71898727e-01,
                           8.69045715e-01, 8.67195282e-01, 8.64461823e-01,
                           8.62905884e-01, 8.60242310e-01, 8.57690887e-01,
                           8.55785751e-01, 8.53161774e-01, 8.51369553e-01,
                           8.48543091e-01, 8.46092071e-01, 8.43811874e-01,
                           8.40855102e-01, 8.38205032e-01, 8.35638428e-01,
                           8.33058090e-01, 8.29829483e-01, 8.26507950e-01,
                           8.24152756e-01, 8.21133499e-01, 8.17982101e-01,
                           8.14945984e-01, 8.11371536e-01, 8.08797302e-01,
                           8.05465164e-01, 8.02152329e-01, 7.99375458e-01,
                           7.95579987e-01, 7.91873245e-01, 7.88838119e-01,
                           7.84947052e-01, 7.82865982e-01, 7.77375183e-01,
                           7.74711151e-01, 7.71566467e-01, 7.67292709e-01,
                           7.63668289e-01, 7.60665512e-01, 7.55569534e-01,
                           7.52378006e-01, 7.48392868e-01, 7.44523621e-01,
                           7.40757904e-01, 7.36248322e-01, 7.32364731e-01,
                           7.28448029e-01, 7.23732147e-01, 7.19756775e-01,
                           7.15782394e-01, 7.11536713e-01, 7.07296219e-01,
                           7.02669830e-01, 6.98336868e-01, 6.93820877e-01,
                           6.89229584e-01, 6.85463638e-01, 6.80321579e-01,
                           6.75755997e-01, 6.71247406e-01, 6.66305160e-01,
                           6.61537552e-01, 6.56552429e-01, 6.51618576e-01,
                           6.46831970e-01, 6.42130890e-01, 6.37422791e-01,
                           6.32663307e-01, 6.26985092e-01, 6.22300797e-01,
                           6.17429542e-01, 6.11961975e-01, 6.07117996e-01,
                           6.01615372e-01, 5.96683311e-01, 5.91556473e-01,
                           5.85764580e-01, 5.81412506e-01, 5.75745583e-01,
                           5.70708580e-01, 5.65521469e-01, 5.60354004e-01,
                           5.55104981e-01, 5.49598465e-01, 5.44442787e-01,
                           5.39409828e-01, 5.34089699e-01, 5.28689613e-01,
                           5.23753700e-01, 5.18192368e-01, 5.12720947e-01,
                           5.07284508e-01, 5.01651344e-01, 4.96233330e-01,
                           4.90987473e-01, 4.85806465e-01, 4.80457954e-01,
                           4.74516029e-01, 4.69459343e-01, 4.63997955e-01,
                           4.58108025e-01, 4.52913590e-01, 4.47898445e-01,
                           4.41578674e-01, 4.36835709e-01, 4.31392746e-01,
                           4.25792809e-01, 4.20569115e-01, 4.14983521e-01,
                           4.09441910e-01, 4.04065590e-01, 3.98449898e-01,
                           3.93368378e-01, 3.88108597e-01, 3.82731361e-01,
                           3.77610168e-01, 3.72011795e-01, 3.66899109e-01,
                           3.61709938e-01, 3.56277771e-01, 3.51459427e-01,
                           3.46341896e-01, 3.41169662e-01, 3.36199074e-01,
                           3.31208305e-01, 3.26275864e-01, 3.21232452e-01,
                           3.15962257e-01, 3.11138630e-01, 3.06086445e-01,
                           3.01351910e-01, 2.96466599e-01, 2.91627788e-01,
                           2.86797676e-01, 2.81993294e-01, 2.77036629e-01,
                           2.72600326e-01, 2.67752075e-01, 2.63035870e-01,
                           2.58718491e-01, 2.53945446e-01, 2.49440594e-01,
                           2.44970150e-01, 2.40328617e-01, 2.36014404e-01,
                           2.31458073e-01, 2.27129078e-01, 2.22980728e-01,
                           2.18599091e-01, 2.14399776e-01, 2.10105076e-01,
                           2.05955944e-01, 2.01979485e-01, 1.97873592e-01,
                           1.93701324e-01, 1.89863262e-01, 1.85919723e-01,
                           1.82102280e-01, 1.78372879e-01, 1.74555264e-01,
                           1.70942688e-01, 1.67413940e-01, 1.63823414e-01,
                           1.60374756e-01, 1.56812820e-01, 1.53197708e-01,
                           1.49876614e-01, 1.46493282e-01, 1.43237667e-01,
                           1.40090466e-01, 1.36744709e-01, 1.33655767e-01,
                           1.30583868e-01, 1.27497015e-01, 1.24574251e-01,
                           1.21548195e-01, 1.18785553e-01, 1.15858727e-01,
                           1.12972259e-01, 1.10239296e-01, 1.07432098e-01,
                           1.04911184e-01, 1.02240067e-01, 9.96163654e-02,
                           9.71846867e-02, 9.46867275e-02, 9.21891499e-02,
                           8.98626804e-02, 8.74147129e-02, 8.50797844e-02,
                           8.28987694e-02, 8.06197929e-02, 7.84934664e-02,
                           7.63682270e-02, 7.41679907e-02, 7.21602154e-02,
                           7.01406241e-02, 6.82159948e-02, 6.62652016e-02,
                           6.43459272e-02, 6.24867964e-02, 6.07102966e-02,
                           5.90227270e-02, 5.73293352e-02, 5.56865645e-02,
                           5.40774345e-02, 5.24679184e-02, 5.08922577e-02,
                           4.93965530e-02, 4.79321527e-02, 4.64570713e-02,
                           4.50907946e-02, 4.36638164e-02, 4.23424053e-02,
                           4.10112333e-02, 3.97419786e-02, 3.85188985e-02,
                           3.72569108e-02, 3.61442852e-02, 3.49567914e-02,
                           3.37763834e-02, 3.27081037e-02, 3.15532732e-02,
                           3.05547738e-02, 2.96382666e-02, 2.86316228e-02,
                           2.76253200e-02, 2.67284703e-02, 2.57629275e-02,
                           2.48762655e-02, 2.40548301e-02, 2.32087660e-02,
                           2.23887801e-02, 2.16649318e-02, 2.08810973e-02,
                           2.01191974e-02, 1.93965495e-02, 1.86923802e-02,
                           1.80622673e-02, 1.73420966e-02, 1.67779624e-02,
                           1.61432099e-02, 1.55458522e-02, 1.49808991e-02,
                           1.44260824e-02, 1.38898337e-02, 1.33757555e-02,
                           1.28895402e-02, 1.24336338e-02, 1.19317114e-02,
                           1.14778078e-02, 1.10224903e-02, 1.05936778e-02,
                           1.01979625e-02, 9.80331957e-03, 9.42119420e-03,
                           9.06843662e-03, 8.70236576e-03, 8.36401224e-03,
                           8.02174568e-03, 7.69513190e-03, 7.42049038e-03,
                           7.12957501e-03, 6.81147277e-03, 6.56225324e-03,
                           6.28752470e-03, 6.03279233e-03, 5.78228355e-03,
                           5.52640975e-03, 5.31245232e-03, 5.07642031e-03,
                           4.86187398e-03, 4.66857612e-03, 4.48455602e-03,
                           4.28951621e-03, 4.10438061e-03, 3.94181907e-03,
                           3.77903283e-03, 3.61310929e-03, 3.43858838e-03,
                           3.30562413e-03, 3.16893756e-03, 3.00862283e-03,
                           2.88184345e-03, 2.75286794e-03, 2.63536334e-03,
                           2.52844244e-03, 2.39721924e-03, 2.31343344e-03,
                           2.19719976e-03, 2.09656358e-03, 2.02219427e-03,
                           1.91874027e-03, 1.81754440e-03, 1.74118712e-03,
                           1.66113898e-03, 1.58724680e-03, 1.51313767e-03,
                           1.44662365e-03, 1.39100656e-03, 1.33283704e-03,
                           1.26319885e-03, 1.18512645e-03, 1.14880271e-03,
                           1.08921751e-03, 1.04411282e-03, 1.01634525e-03,
                           9.41211507e-04, 9.03511718e-04, 8.70077759e-04,
                           8.34191218e-04, 7.73599520e-04, 7.44963065e-04,
                           7.18376786e-04, 6.85756877e-04, 6.50605410e-04,
                           6.14275858e-04, 5.89862131e-04, 5.59216291e-04,
                           5.29026911e-04, 4.99960780e-04, 4.72659841e-04,
                           4.56626341e-04, 4.29005548e-04, 4.13897783e-04,
                           3.97251360e-04, 3.70411240e-04, 3.54581289e-04,
                           3.36891152e-04, 3.18884142e-04, 3.09158638e-04,
                           2.87089385e-04, 2.75648981e-04, 2.56309062e-04,
                           2.48264093e-04, 2.32592076e-04, 2.18097549e-04,
                           2.10234672e-04, 2.01618839e-04, 1.92721710e-04,
                           1.84358787e-04, 1.78293809e-04, 1.73047427e-04,
                           1.48465503e-04, 1.50579475e-04, 1.37227150e-04,
                           1.30995326e-04, 1.18210996e-04, 1.10485023e-04,
                           1.12393992e-04, 1.07742772e-04, 1.06566232e-04,
                           8.77865311e-05, 9.66540072e-05, 8.63869675e-05])

        flux_R = np.array([1.12660611e-04, 1.33478958e-04, 1.80384908e-04,
                           2.26182416e-04, 2.96486858e-04, 3.83854918e-04,
                           4.94274013e-04, 6.20536394e-04, 8.19598287e-04,
                           1.01240180e-03, 1.29484743e-03, 1.64972723e-03,
                           2.04623789e-03, 2.60429144e-03, 3.19142252e-03,
                           3.95557463e-03, 4.87352252e-03, 5.92993259e-03,
                           7.22202599e-03, 8.75534654e-03, 1.05062985e-02,
                           1.26144767e-02, 1.49658072e-02, 1.76800156e-02,
                           2.09657979e-02, 2.44697619e-02, 2.87300396e-02,
                           3.34529758e-02, 3.85330200e-02, 4.46062708e-02,
                           5.08374691e-02, 5.79355812e-02, 6.60423279e-02,
                           7.43976021e-02, 8.39634419e-02, 9.44021988e-02,
                           1.04971266e-01, 1.16864176e-01, 1.29295054e-01,
                           1.42394171e-01, 1.56620798e-01, 1.70939655e-01,
                           1.86083679e-01, 2.02246418e-01, 2.18151264e-01,
                           2.35699348e-01, 2.52898312e-01, 2.70299339e-01,
                           2.88551636e-01, 3.06377716e-01, 3.25947761e-01,
                           3.45086975e-01, 3.63418694e-01, 3.82655678e-01,
                           4.01391029e-01, 4.19963226e-01, 4.39177132e-01,
                           4.56956482e-01, 4.75537567e-01, 4.93223953e-01,
                           5.10155792e-01, 5.27090416e-01, 5.43785629e-01,
                           5.59207916e-01, 5.75155678e-01, 5.89269867e-01,
                           6.03433266e-01, 6.18236656e-01, 6.30981636e-01,
                           6.43544693e-01, 6.55758591e-01, 6.67161560e-01,
                           6.78610764e-01, 6.89398499e-01, 6.99007721e-01,
                           7.09150238e-01, 7.17486267e-01, 7.26359787e-01,
                           7.34181595e-01, 7.41922607e-01, 7.49040909e-01,
                           7.55139770e-01, 7.61801071e-01, 7.67739029e-01,
                           7.72209625e-01, 7.77520752e-01, 7.82076034e-01,
                           7.86005707e-01, 7.90121536e-01, 7.94920044e-01,
                           7.97914963e-01, 8.01576385e-01, 8.04085770e-01,
                           8.06881256e-01, 8.09733276e-01, 8.12508926e-01,
                           8.14496231e-01, 8.16916046e-01, 8.18313217e-01,
                           8.20173111e-01, 8.21818848e-01, 8.23354797e-01,
                           8.24062653e-01, 8.25225525e-01, 8.26539078e-01,
                           8.27467270e-01, 8.28310471e-01, 8.29260254e-01,
                           8.29644699e-01, 8.29694901e-01, 8.30798569e-01,
                           8.31418304e-01, 8.31113281e-01, 8.31175461e-01,
                           8.31436615e-01, 8.31268921e-01, 8.31743851e-01,
                           8.31236649e-01, 8.31876831e-01, 8.31575623e-01,
                           8.31600800e-01, 8.31209564e-01, 8.30701218e-01,
                           8.30457306e-01, 8.29995575e-01, 8.29173889e-01,
                           8.28681335e-01, 8.28388367e-01, 8.27705078e-01,
                           8.26961517e-01, 8.26470642e-01, 8.25616913e-01,
                           8.25088272e-01, 8.24414825e-01, 8.23818588e-01,
                           8.22574463e-01, 8.21790543e-01, 8.20854645e-01,
                           8.20430603e-01, 8.19333649e-01, 8.18388138e-01,
                           8.17239914e-01, 8.16441727e-01, 8.15142059e-01,
                           8.14114456e-01, 8.13138275e-01, 8.12385178e-01,
                           8.11399994e-01, 8.10151062e-01, 8.09062042e-01,
                           8.07826004e-01, 8.06391449e-01, 8.05179291e-01,
                           8.04337387e-01, 8.02874298e-01, 8.01418991e-01,
                           8.00320816e-01, 7.99105682e-01, 7.97680512e-01,
                           7.96293411e-01, 7.94735107e-01, 7.93599701e-01,
                           7.92142716e-01, 7.90940323e-01, 7.89540253e-01,
                           7.87977982e-01, 7.86476135e-01, 7.85149383e-01,
                           7.83683319e-01, 7.82463837e-01, 7.80975647e-01,
                           7.79384079e-01, 7.77804413e-01, 7.76397171e-01,
                           7.74585876e-01, 7.73283157e-01, 7.71683350e-01,
                           7.70116653e-01, 7.68394089e-01, 7.66989212e-01,
                           7.65374298e-01, 7.63670044e-01, 7.61980438e-01,
                           7.60181885e-01, 7.58677445e-01, 7.57341537e-01,
                           7.55792389e-01, 7.54106216e-01, 7.52319260e-01,
                           7.50747833e-01, 7.48828659e-01, 7.47205200e-01,
                           7.45405502e-01, 7.43702850e-01, 7.42157440e-01,
                           7.40391464e-01, 7.38478088e-01, 7.36322479e-01,
                           7.34597397e-01, 7.32816925e-01, 7.31027298e-01,
                           7.29303818e-01, 7.27694702e-01, 7.25626068e-01,
                           7.24098816e-01, 7.22092285e-01, 7.20166626e-01,
                           7.18592148e-01, 7.16398239e-01, 7.14680633e-01,
                           7.12456436e-01, 7.10820770e-01, 7.09065247e-01,
                           7.06785812e-01, 7.05026474e-01, 7.03354034e-01,
                           7.01381912e-01, 6.99503784e-01, 6.97199249e-01,
                           6.95120850e-01, 6.93079453e-01, 6.91699600e-01,
                           6.89639130e-01, 6.88427200e-01, 6.85872650e-01,
                           6.84145126e-01, 6.81911545e-01, 6.80322800e-01,
                           6.78288803e-01, 6.76393280e-01, 6.74223022e-01,
                           6.72408447e-01, 6.70496292e-01, 6.68415146e-01,
                           6.66331940e-01, 6.64745712e-01, 6.62663345e-01,
                           6.60627213e-01, 6.58656998e-01, 6.56490936e-01,
                           6.54593048e-01, 6.52417145e-01, 6.50451279e-01,
                           6.48244934e-01, 6.46139450e-01, 6.44154511e-01,
                           6.41925736e-01, 6.39975548e-01, 6.37752533e-01,
                           6.35898399e-01, 6.33897591e-01, 6.31938820e-01,
                           6.29536552e-01, 6.27312431e-01, 6.25279121e-01,
                           6.23031921e-01, 6.20859680e-01, 6.18729477e-01,
                           6.16721458e-01, 6.14748001e-01, 6.12250404e-01,
                           6.09872932e-01, 6.07715263e-01, 6.05285225e-01,
                           6.03101807e-01, 6.01018982e-01, 5.99403038e-01,
                           5.96835365e-01, 5.94723625e-01, 5.92363167e-01,
                           5.89933815e-01, 5.86952133e-01, 5.84768906e-01,
                           5.82397041e-01, 5.80457268e-01, 5.77794266e-01,
                           5.75973740e-01, 5.73014793e-01, 5.70719414e-01,
                           5.68651657e-01, 5.66127243e-01, 5.63723564e-01,
                           5.61353035e-01, 5.58687668e-01, 5.56360054e-01,
                           5.53829727e-01, 5.51511993e-01, 5.49103394e-01,
                           5.46937523e-01, 5.44495354e-01, 5.42087212e-01,
                           5.39432335e-01, 5.37001495e-01, 5.34510727e-01,
                           5.31703186e-01, 5.29667206e-01, 5.27464333e-01,
                           5.24670296e-01, 5.22587357e-01, 5.19773483e-01,
                           5.17762489e-01, 5.14889717e-01, 5.12675095e-01,
                           5.10391426e-01, 5.07693596e-01, 5.05560875e-01,
                           5.02788238e-01, 5.00663567e-01, 4.98405113e-01,
                           4.95754623e-01, 4.93308716e-01, 4.90971375e-01,
                           4.88512230e-01, 4.85908470e-01, 4.84007683e-01,
                           4.81591797e-01, 4.79094429e-01, 4.76312561e-01,
                           4.73944168e-01, 4.71328812e-01, 4.69270897e-01,
                           4.66906967e-01, 4.64348908e-01, 4.61959457e-01,
                           4.59419556e-01, 4.57119751e-01, 4.54282990e-01,
                           4.52030411e-01, 4.49744415e-01, 4.47503815e-01,
                           4.44987106e-01, 4.42915993e-01, 4.40122299e-01,
                           4.38269691e-01, 4.35202255e-01, 4.33002968e-01,
                           4.30703163e-01, 4.28281441e-01, 4.25861244e-01,
                           4.23408241e-01, 4.21262741e-01, 4.19147110e-01,
                           4.16939697e-01, 4.14542465e-01, 4.11997719e-01,
                           4.09688759e-01, 4.07355232e-01, 4.04657173e-01,
                           4.02887306e-01, 4.00700073e-01, 3.98309898e-01,
                           3.95669937e-01, 3.93478394e-01, 3.91111298e-01,
                           3.88895645e-01, 3.86983261e-01, 3.84384155e-01,
                           3.81797638e-01, 3.79871559e-01, 3.77870216e-01,
                           3.75476189e-01, 3.73131638e-01, 3.70839462e-01,
                           3.69031487e-01, 3.66161499e-01, 3.63859253e-01,
                           3.61430778e-01, 3.59496612e-01, 3.57683106e-01,
                           3.55424080e-01, 3.52959938e-01, 3.50599556e-01,
                           3.48366928e-01, 3.46199951e-01, 3.43800392e-01,
                           3.41833038e-01, 3.39689293e-01, 3.37388229e-01,
                           3.35983315e-01, 3.33557548e-01, 3.31361923e-01,
                           3.29263535e-01, 3.27118683e-01, 3.24498863e-01,
                           3.22609215e-01, 3.20428238e-01, 3.18339233e-01,
                           3.16222420e-01, 3.14079876e-01, 3.12005463e-01,
                           3.09681053e-01, 3.07576656e-01, 3.05554867e-01,
                           3.03675804e-01, 3.01599236e-01, 2.99350357e-01,
                           2.97287026e-01, 2.95042343e-01, 2.93254433e-01,
                           2.91312427e-01, 2.89098625e-01, 2.86699619e-01,
                           2.84973373e-01, 2.82804375e-01, 2.81043167e-01,
                           2.79479942e-01, 2.76905003e-01, 2.74912872e-01,
                           2.72875061e-01, 2.71315537e-01, 2.68872356e-01,
                           2.67071037e-01, 2.64945831e-01, 2.62771225e-01,
                           2.60814991e-01, 2.59156818e-01, 2.56677303e-01,
                           2.54789314e-01, 2.53038921e-01, 2.51051693e-01,
                           2.49118004e-01, 2.46885796e-01, 2.45392628e-01,
                           2.43349152e-01, 2.41043224e-01, 2.39375744e-01,
                           2.37449379e-01, 2.35649910e-01, 2.33648262e-01,
                           2.32286263e-01, 2.30330391e-01, 2.28001060e-01,
                           2.26452904e-01, 2.24508724e-01, 2.22819996e-01,
                           2.20511837e-01, 2.19196682e-01, 2.17359448e-01,
                           2.15409527e-01, 2.13571644e-01, 2.11919060e-01,
                           2.10245914e-01, 2.08496246e-01, 2.06775856e-01,
                           2.05235577e-01, 2.03262482e-01, 2.01522713e-01,
                           1.99663773e-01, 1.97996788e-01, 1.96391239e-01,
                           1.94632092e-01, 1.92989120e-01, 1.91479111e-01,
                           1.89962959e-01, 1.87962627e-01, 1.86370125e-01,
                           1.84920654e-01, 1.83073902e-01, 1.81668034e-01,
                           1.80077705e-01, 1.78313961e-01, 1.76784782e-01,
                           1.75110645e-01, 1.73803921e-01, 1.72050915e-01,
                           1.70811748e-01, 1.68707829e-01, 1.67500534e-01,
                           1.65955715e-01, 1.64152584e-01, 1.62616043e-01,
                           1.61383820e-01, 1.59913750e-01, 1.58476162e-01,
                           1.57111960e-01, 1.55604382e-01, 1.54195471e-01,
                           1.52868767e-01, 1.51168289e-01, 1.50135088e-01,
                           1.48432417e-01, 1.46854248e-01, 1.45500660e-01,
                           1.44040155e-01, 1.43029194e-01, 1.41359615e-01,
                           1.40144958e-01, 1.38888855e-01, 1.37300205e-01,
                           1.36141462e-01, 1.34810266e-01, 1.33652449e-01,
                           1.32385340e-01, 1.30962801e-01, 1.29514580e-01,
                           1.28492441e-01, 1.26976881e-01, 1.26109915e-01,
                           1.24681196e-01, 1.23733912e-01, 1.22387972e-01,
                           1.21014032e-01, 1.19707127e-01, 1.18950415e-01,
                           1.17601652e-01, 1.16029644e-01, 1.15246582e-01,
                           1.13969402e-01, 1.12859097e-01, 1.11570110e-01,
                           1.10585833e-01, 1.09544601e-01, 1.08406753e-01,
                           1.07325516e-01, 1.05842676e-01, 1.04812813e-01,
                           1.03711939e-01, 1.02703686e-01, 1.01885681e-01,
                           1.00853710e-01, 9.96105671e-02, 9.87637615e-02,
                           9.77460957e-02, 9.68516922e-02, 9.56964302e-02,
                           9.48740578e-02, 9.36437607e-02, 9.26385784e-02,
                           9.13605881e-02, 9.08198070e-02, 8.97638321e-02,
                           8.86697960e-02, 8.77115726e-02, 8.71175385e-02,
                           8.63109493e-02, 8.48536015e-02, 8.42036724e-02,
                           8.32233620e-02, 8.23537445e-02, 8.15705395e-02,
                           8.05418396e-02, 7.98623276e-02, 7.91370583e-02,
                           7.78403139e-02, 7.73310661e-02, 7.62543249e-02,
                           7.54598522e-02, 7.44599009e-02, 7.38250256e-02,
                           7.31048202e-02, 7.23627281e-02, 7.15131903e-02,
                           7.05549860e-02, 6.98634911e-02, 6.91224623e-02,
                           6.86638069e-02, 6.76796818e-02, 6.68600273e-02,
                           6.60720110e-02, 6.53426409e-02, 6.48589230e-02,
                           6.40281153e-02, 6.31698275e-02, 6.24832773e-02,
                           6.17807865e-02, 6.11954021e-02, 6.05794573e-02,
                           5.96689224e-02, 5.90339708e-02, 5.84838772e-02,
                           5.78847265e-02, 5.68160105e-02, 5.64464664e-02,
                           5.57960987e-02, 5.50762606e-02, 5.47479629e-02,
                           5.40395975e-02, 5.31866121e-02, 5.24796009e-02,
                           5.18524837e-02, 5.13265848e-02, 5.05894184e-02,
                           5.04498529e-02, 4.95917797e-02, 4.92178106e-02,
                           4.86410618e-02, 4.78479099e-02, 4.73841429e-02,
                           4.68996859e-02, 4.65036964e-02, 4.57519102e-02,
                           4.53436470e-02, 4.48195744e-02, 4.40284443e-02,
                           4.36079264e-02, 4.33500671e-02, 4.26576328e-02,
                           4.20515776e-02, 4.15753365e-02, 4.11065292e-02,
                           4.07284117e-02, 4.01105547e-02, 3.95491576e-02,
                           3.92478895e-02, 3.86123323e-02, 3.83627343e-02,
                           3.81744385e-02, 3.72538948e-02, 3.67257714e-02,
                           3.64651537e-02, 3.61046267e-02, 3.56324434e-02,
                           3.50495958e-02, 3.47760701e-02, 3.45552087e-02,
                           3.38934398e-02, 3.36678410e-02, 3.31091881e-02,
                           3.26658273e-02, 3.23304272e-02, 3.17972445e-02,
                           3.14868403e-02, 3.11922049e-02, 3.07040787e-02,
                           3.03110600e-02, 2.99594235e-02, 2.98183370e-02,
                           2.92352104e-02, 2.89947557e-02, 2.86772442e-02,
                           2.83287978e-02, 2.79210877e-02, 2.72823572e-02,
                           2.73149657e-02, 2.69718742e-02, 2.67807961e-02,
                           2.61144757e-02, 2.57569838e-02, 2.57412481e-02,
                           2.51048923e-02, 2.50279760e-02, 2.49131537e-02,
                           2.45391846e-02, 2.42700195e-02, 2.38901758e-02,
                           2.35897589e-02, 2.28670168e-02, 2.28611231e-02,
                           2.27534866e-02, 2.24620295e-02, 2.19526005e-02,
                           2.16079593e-02, 2.14886975e-02, 2.11848760e-02,
                           2.12790751e-02, 2.06619120e-02, 2.07371426e-02,
                           2.00993228e-02, 1.95814931e-02, 1.95096111e-02,
                           1.88129783e-02, 1.91138482e-02, 1.89894068e-02,
                           1.82900357e-02, 1.82558620e-02, 1.84180438e-02,
                           1.78343022e-02, 1.79508388e-02, 1.98078752e-02,
                           2.35607266e-02, 1.64428818e-02, 1.63446629e-02,
                           1.61414671e-02, 1.59015155e-02, 1.57553589e-02,
                           1.55644822e-02, 1.53442860e-02, 1.52152765e-02,
                           1.49248958e-02, 1.47469020e-02, 1.46128261e-02,
                           1.45537209e-02, 1.43860090e-02, 1.40903854e-02,
                           1.39411104e-02, 1.37448251e-02, 1.35096633e-02,
                           1.34330940e-02, 1.32138276e-02, 1.30654049e-02,
                           1.28928685e-02, 1.27844548e-02, 1.25968790e-02,
                           1.24387026e-02, 1.23236620e-02, 1.21577203e-02,
                           1.19817626e-02, 1.18997812e-02, 1.17299104e-02,
                           1.16228032e-02, 1.13986945e-02, 1.13025677e-02,
                           1.11602139e-02, 1.10250735e-02, 1.09074187e-02,
                           1.07202637e-02, 1.06087947e-02, 1.05153501e-02,
                           1.03730762e-02, 1.02454245e-02, 1.00866878e-02,
                           9.99053955e-03, 9.78911459e-03, 9.76708233e-03,
                           9.62086201e-03, 9.47241306e-03, 9.33747649e-03,
                           9.41326499e-03, 9.13064659e-03, 9.12852585e-03,
                           9.06752527e-03, 8.93405914e-03, 8.67768466e-03,
                           8.64216387e-03, 8.60476136e-03, 8.40433478e-03,
                           8.29408765e-03, 8.28387678e-03, 8.08252513e-03,
                           8.08622956e-03, 7.89401472e-03, 7.83714354e-03,
                           7.71972716e-03, 7.65594542e-03, 7.46691644e-03,
                           7.51844585e-03, 7.36561239e-03, 7.31347740e-03,
                           7.21074879e-03, 7.17079341e-03, 7.00386226e-03,
                           7.00467884e-03, 6.87995970e-03, 6.80604935e-03,
                           6.66877091e-03, 6.58461690e-03, 6.56225383e-03,
                           6.54657483e-03, 6.29706144e-03, 6.29498184e-03,
                           6.20202959e-03, 6.14432633e-03, 6.14413202e-03,
                           6.01232946e-03, 5.90509057e-03, 5.87786853e-03,
                           5.79836965e-03, 5.70700347e-03, 5.57661533e-03,
                           5.59826493e-03, 5.52282333e-03, 5.46855211e-03,
                           5.39687157e-03, 5.30140877e-03, 5.28882802e-03,
                           5.22834003e-03, 5.12682915e-03, 5.03452301e-03,
                           4.97473180e-03, 5.00698507e-03, 4.91672516e-03,
                           4.86153126e-03, 4.76140350e-03, 4.73320752e-03,
                           4.78468746e-03, 4.58373725e-03, 4.58816707e-03,
                           4.48710144e-03, 4.41632897e-03, 4.37773258e-03])

        flux_u = np.array(
            [0.00000000e+00, 1.00000000e-04, 5.00000000e-04, 1.30000000e-03, 2.60000000e-03, 5.20000000e-03,
             9.30000000e-03, 1.61000000e-02, 2.40000000e-02, 3.23000000e-02, 4.05000000e-02, 4.85000000e-02,
             5.61000000e-02, 6.34000000e-02, 7.00000000e-02, 7.56000000e-02, 8.03000000e-02, 8.48000000e-02,
             8.83000000e-02, 9.17000000e-02, 9.59000000e-02, 1.00100000e-01, 1.02900000e-01, 1.04400000e-01,
             1.05300000e-01, 1.06300000e-01, 1.07500000e-01, 1.08500000e-01, 1.08400000e-01, 1.06400000e-01,
             1.02400000e-01, 9.66000000e-02, 8.87000000e-02, 7.87000000e-02, 6.72000000e-02, 5.49000000e-02,
             4.13000000e-02, 2.68000000e-02, 1.45000000e-02, 7.50000000e-03, 4.20000000e-03, 2.20000000e-03,
             1.00000000e-03, 6.00000000e-04, 4.00000000e-04, 2.00000000e-04, 0.00000000e+00])

        flux_g = np.array(
            [0.00000000e+00, 3.00000000e-04, 8.00000000e-04,
             1.30000000e-03, 1.90000000e-03, 2.40000000e-03,
             3.40000000e-03, 5.50000000e-03, 1.03000000e-02,
             1.94000000e-02, 3.26000000e-02, 4.92000000e-02,
             6.86000000e-02, 9.00000000e-02, 1.12300000e-01,
             1.34200000e-01, 1.54500000e-01, 1.72200000e-01,
             1.87300000e-01, 2.00300000e-01, 2.11600000e-01,
             2.21400000e-01, 2.30100000e-01, 2.37800000e-01,
             2.44800000e-01, 2.51300000e-01, 2.57400000e-01,
             2.63300000e-01, 2.69100000e-01, 2.74700000e-01,
             2.80100000e-01, 2.85200000e-01, 2.89900000e-01,
             2.94000000e-01, 2.97900000e-01, 3.01600000e-01,
             3.05500000e-01, 3.09700000e-01, 3.14100000e-01,
             3.18400000e-01, 3.22400000e-01, 3.25700000e-01,
             3.28400000e-01, 3.30700000e-01, 3.32700000e-01,
             3.34600000e-01, 3.36400000e-01, 3.38300000e-01,
             3.40300000e-01, 3.42500000e-01, 3.44800000e-01,
             3.47200000e-01, 3.49500000e-01, 3.51900000e-01,
             3.54100000e-01, 3.56200000e-01, 3.58100000e-01,
             3.59700000e-01, 3.60900000e-01, 3.61300000e-01,
             3.60900000e-01, 3.59500000e-01, 3.58100000e-01,
             3.55800000e-01, 3.45200000e-01, 3.19400000e-01,
             2.80700000e-01, 2.33900000e-01, 1.83900000e-01,
             1.35200000e-01, 9.11000000e-02, 5.48000000e-02,
             2.95000000e-02, 1.66000000e-02, 1.12000000e-02,
             7.70000000e-03, 5.00000000e-03, 3.20000000e-03,
             2.10000000e-03, 1.50000000e-03, 1.20000000e-03,
             1.00000000e-03, 9.00000000e-04, 8.00000000e-04,
             6.00000000e-04, 5.00000000e-04, 3.00000000e-04,
             1.00000000e-04, 0.00000000e+00])

        flux_r = np.array(
            [0.00000000e+00, 1.40000000e-03, 9.90000000e-03,
             2.60000000e-02, 4.98000000e-02, 8.09000000e-02,
             1.19000000e-01, 1.63000000e-01, 2.10000000e-01,
             2.56400000e-01, 2.98600000e-01, 3.33900000e-01,
             3.62300000e-01, 3.84900000e-01, 4.02700000e-01,
             4.16500000e-01, 4.27100000e-01, 4.35300000e-01,
             4.41600000e-01, 4.46700000e-01, 4.51100000e-01,
             4.55000000e-01, 4.58700000e-01, 4.62400000e-01,
             4.66000000e-01, 4.69200000e-01, 4.71600000e-01,
             4.73100000e-01, 4.74000000e-01, 4.74700000e-01,
             4.75800000e-01, 4.77600000e-01, 4.80000000e-01,
             4.82700000e-01, 4.85400000e-01, 4.88100000e-01,
             4.90500000e-01, 4.92600000e-01, 4.94200000e-01,
             4.95100000e-01, 4.95500000e-01, 4.95600000e-01,
             4.95800000e-01, 4.96100000e-01, 4.96400000e-01,
             4.96200000e-01, 4.95300000e-01, 4.93100000e-01,
             4.90600000e-01, 4.87300000e-01, 4.75200000e-01,
             4.47400000e-01, 4.05900000e-01, 3.54400000e-01,
             2.96300000e-01, 2.35000000e-01, 1.73900000e-01,
             1.16800000e-01, 6.97000000e-02, 3.86000000e-02,
             2.15000000e-02, 1.36000000e-02, 1.01000000e-02,
             7.70000000e-03, 5.60000000e-03, 3.90000000e-03,
             2.80000000e-03, 2.00000000e-03, 1.60000000e-03,
             1.30000000e-03, 1.00000000e-03, 7.00000000e-04,
             4.00000000e-04, 2.00000000e-04, 0.00000000e+00])

        flux_i = np.array(
            [0.00000000e+00, 1.00000000e-04, 3.00000000e-04,
             4.00000000e-04, 4.00000000e-04, 4.00000000e-04,
             3.00000000e-04, 4.00000000e-04, 9.00000000e-04,
             1.90000000e-03, 3.40000000e-03, 5.60000000e-03,
             1.04000000e-02, 1.97000000e-02, 3.49000000e-02,
             5.69000000e-02, 8.51000000e-02, 1.18100000e-01,
             1.55200000e-01, 1.98000000e-01, 2.44800000e-01,
             2.90600000e-01, 3.29000000e-01, 3.56600000e-01,
             3.82900000e-01, 4.06700000e-01, 4.24500000e-01,
             4.32000000e-01, 4.25200000e-01, 4.02800000e-01,
             3.84400000e-01, 3.91100000e-01, 4.01100000e-01,
             3.98800000e-01, 3.92400000e-01, 3.91900000e-01,
             3.98800000e-01, 3.97900000e-01, 3.93000000e-01,
             3.89800000e-01, 3.87200000e-01, 3.84200000e-01,
             3.79900000e-01, 3.73700000e-01, 3.68500000e-01,
             3.67800000e-01, 3.60300000e-01, 1.52700000e-01,
             2.17600000e-01, 2.75200000e-01, 3.43400000e-01,
             3.39200000e-01, 3.36100000e-01, 3.31900000e-01,
             3.27200000e-01, 3.22100000e-01, 3.17300000e-01,
             3.12900000e-01, 3.09500000e-01, 3.07700000e-01,
             3.07500000e-01, 3.08600000e-01, 3.09800000e-01,
             3.09800000e-01, 3.07600000e-01, 3.02100000e-01,
             2.93900000e-01, 2.82100000e-01, 2.59700000e-01,
             2.24200000e-01, 1.81500000e-01, 1.37400000e-01,
             9.73000000e-02, 6.52000000e-02, 4.10000000e-02,
             2.37000000e-02, 1.28000000e-02, 7.40000000e-03,
             5.30000000e-03, 3.60000000e-03, 2.20000000e-03,
             1.40000000e-03, 1.10000000e-03, 1.00000000e-03,
             1.00000000e-03, 9.00000000e-04, 6.00000000e-04,
             3.00000000e-04, 0.00000000e+00])

        flux_z = np.array(
            [0., 0., 0.0001, 0.0001, 0.0001, 0.0002, 0.0002,
             0.0003, 0.0005, 0.0007, 0.0011, 0.0017, 0.0027, 0.004,
             0.0057, 0.0079, 0.0106, 0.0139, 0.0178, 0.0222, 0.0271,
             0.0324, 0.0382, 0.0446, 0.0511, 0.0564, 0.0603, 0.0637,
             0.0667, 0.0694, 0.0717, 0.0736, 0.0752, 0.0765, 0.0775,
             0.0782, 0.0786, 0.0787, 0.0785, 0.078, 0.0772, 0.0763,
             0.0751, 0.0738, 0.0723, 0.0708, 0.0693, 0.0674, 0.0632,
             0.0581, 0.0543, 0.0526, 0.0523, 0.0522, 0.0512, 0.0496,
             0.0481, 0.0473, 0.0476, 0.0482, 0.0476, 0.0447, 0.0391,
             0.0329, 0.0283, 0.0264, 0.0271, 0.0283, 0.0275, 0.0254,
             0.0252, 0.0256, 0.0246, 0.0244, 0.0252, 0.0258, 0.0265,
             0.0274, 0.0279, 0.0271, 0.0252, 0.0236, 0.0227, 0.0222,
             0.0216, 0.0208, 0.0196, 0.0183, 0.0171, 0.016, 0.0149,
             0.0138, 0.0128, 0.0118, 0.0108, 0.0099, 0.0091, 0.0083,
             0.0075, 0.0068, 0.0061, 0.0055, 0.005, 0.0045, 0.0041,
             0.0037, 0.0033, 0.003, 0.0027, 0.0025, 0.0023, 0.0021,
             0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0014, 0.0013,
             0.0012, 0.0011, 0.001, 0.0009, 0.0008, 0.0008, 0.0007,
             0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0004, 0.0004,
             0.0003, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0., 0.])
        if _filter == 'R':
            wave_filter = wave_R
            flux_filter = flux_R
        elif _filter == 'V':
            wave_filter = wave_V
            flux_filter = flux_V
        elif _filter == 'u':
            wave_filter = wave_u
            flux_filter = flux_u
        elif _filter == 'g':
            wave_filter = wave_g
            flux_filter = flux_g
        elif _filter == 'r':
            wave_filter = wave_r
            flux_filter = flux_r
        elif _filter == 'i':
            wave_filter = wave_i
            flux_filter = flux_i
        elif _filter == 'z':
            wave_filter = wave_z
            flux_filter = flux_z
        else:
            raise ValueError('not implemented transmission curve')
            # filter es una built-in the python, creo que es mejor cambiarlo a ese nombre para evitar confusiones.

        final_flux_filter = self.filter_to_MUSE_wavelength(wave_filter, flux_filter, wavelength_spec)
        return final_flux_filter

    def filter_to_MUSE_wavelength(self, wave_filter, flux_filter, wavelength_spec):
        new_filter_wavelength = self.overlap_filter(wave_filter, wavelength_spec)
        interpolator = interpolate.interp1d(wave_filter, flux_filter)
        new_filter_flux = interpolator(new_filter_wavelength)
        final_flux_filter = []

        for j, w in enumerate(wavelength_spec):
            k = mcu.indexOf(new_filter_wavelength, w)
            if k >= 0:
                final_flux_filter.append(new_filter_flux[k])
            else:
                final_flux_filter.append(0.)
        return np.array(final_flux_filter)

    def overlap_filter(self, wave_filter, wavelength_spec):
        n = len(wave_filter)
        w_min = wave_filter[0]
        w_max = wave_filter[n - 1]
        w_spec_overlap = []
        if wave_filter[1] < wavelength_spec[0] or wave_filter[n - 2] > wavelength_spec[len(wavelength_spec) - 1]:
            raise ValueError('Filter wavelength range is wider that spectrum range and convolution is not valid')

        for w in wavelength_spec:
            if w >= w_min and w <= w_max:
                w_spec_overlap.append(w)
        return np.array(w_spec_overlap)

    def reload_canvas(self, vmin=None, vmax=None):
        """
        Clean everything from the canvas with the white image
        :param self:
        :return:
        """

        plt.figure(self.n)
        plt.clf()
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax
        self.gc2 = aplpy.FITSFigure(self.filename_white, figure=plt.figure(self.n))
        if self.color:
            self.gc2.show_colorscale(cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        else:
            self.gc2.show_grayscale(vmin=self.vmin, vmax=self.vmax)
        plt.show()

    def get_from_table(self, input_file, keyword):
        """
        Get a columns that correspond to a given keyword from a SExtractor outputfile
        :param input_file: string
                         name of the SExtractor output file

        :param keyword: string
                        keyword in the SExtractor output file
        :return: data
                 the column associated to the keyword
        """
        sex = SExtractor()
        table = sex.read(input_file)
        data = table[keyword]
        return data

    def get_gaussian_profile_weighted_spec(self, x_c=None, y_c=None, params=None, region_string_=None,
                                           coord_system='pix'):
        """
        Function that extract the spectrum from an aperture defined either by elliptical parameters or  by an elliptical region defined by region_string in ds9 format
        :param x_c: x_coordinate of the center of the aperture
        :param y_c: y_coordinate of the center of the aperture
        :param params: Either a single radius or a set of [a,b,theta] params
        :param region_string: region defined by ds9 format (ellipse)
        :param coord_system: in the case of defining and aperture using x_c,y_c,params, must indicate the type of this coordiantes. Possible values: 'pix' and 'wcs'
        :return: XSpectrum1D object
        """
        if max(x_c, y_c, params, region_string_) == None:
            raise ValueError('Not valid input')
        if region_string_ != None:
            x_c, y_c, params = self.params_from_ellipse_region_string(region_string_)
        if not isinstance(params, (int, float, tuple, list, np.array)):
            raise ValueError('Not ready for this `radius` type.')
        if isinstance(params, (int, float)):
            a = params
            b = params
            theta = 0
        elif isiterable(params) and (len(params) == 3):
            a = max(params[:2])
            b = min(params[:2])
            theta = params[2]
        else:
            raise ValueError('If iterable, the length of radius must be == 3; otherwise try float.')
        if coord_system == 'wcs':
            x_center, y_center, params = self.ellipse_params_to_pixel(x_c, y_c, params=[a, b, theta])
        else:  # already in pixels
            x_center, y_center, params = x_c, y_c, [a, b, theta]
        xc = x_center
        yc = y_center

        new_mask = self.get_mini_cube_mask_from_ellipse_params(x_center, y_center, params)
        spec_sum = self.spec_from_minicube_mask(new_mask, mode='sum')

        halfsize = [a, b]
        if region_string_ == None:
            region_string = self.ellipse_param_to_ds9reg_string(xc, yc, a, b, theta)
        else:
            region_string = region_string_
        new_2dmask = self.get_new_2dmask(region_string)
        masked_white = ma.MaskedArray(self.white_data)
        masked_white.mask = new_2dmask
        ###### Define domain matrix:
        matrix_x = np.zeros_like(self.white_data)
        matrix_y = np.zeros_like(self.white_data)
        n = self.white_data.shape[0]
        m = self.white_data.shape[1]
        for i in xrange(m):
            matrix_x[:, i] = i
        for j in xrange(n):
            matrix_y[j, :] = j
        ###########

        amp_init = masked_white.max()
        stdev_init_x = 0.33 * halfsize[0]
        stdev_init_y = 0.33 * halfsize[1]
        g_init = models.Gaussian2D(x_mean=xc, y_mean=yc, x_stddev=stdev_init_x,
                                   y_stddev=stdev_init_y, amplitude=amp_init, theta=theta)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, matrix_x, matrix_y, masked_white)
        weights = ma.MaskedArray(g(matrix_x, matrix_y))
        if (g.y_stddev < 0) or (g.x_stddev < 0):
            raise ValueError('Cannot trust the model, please try other input parameters.')
        w = self.wavelength
        n = len(w)
        fl = np.zeros(n)
        sig = np.zeros(n)
        new_3dmask = self.get_new_3dmask(region_string)
        self.cube.mask = new_3dmask
        for wv_ii in range(n):
            mask = new_3dmask[wv_ii]
            weights.mask = mask
            # n_spaxels = np.sum(mask)
            weights = weights / np.sum(weights)
            fl[wv_ii] = np.sum(self.cube[wv_ii] * weights)  # * n_spaxels
            sig[wv_ii] = np.sqrt(np.sum(self.stat[wv_ii] * (weights ** 2)))  # * n_spaxels
        # reset mask
        self.cube.mask = self.mask_init

        # renormalize
        fl_sum = spec_sum.flux.value
        norm = np.sum(fl_sum) / np.sum(fl)
        fl = fl * norm
        sig = sig * norm
        return XSpectrum1D.from_tuple((w, fl, sig))

    def determinate_seeing_from_white(self, xc, yc, halfsize):
        """
        Function used to estimate the observation seeing of an exposure, fitting a gaussian to  a brigth source of the  image
        :param xc: x coordinate in pixels of a bright source
        :param yc: y coordinate  in pixels of a bright source
        :param halfsize: the radius of the area to fit the gaussian
        :return: seeing: float
                         the observational seeing of the image defined as the FWHM of the gaussian
        """
        hdulist = self.hdulist_white
        data = hdulist[1].data
        matrix_data = np.array(self.get_mini_image([xc, yc], halfsize=halfsize))
        x = np.arange(0, matrix_data.shape[0], 1)
        y = np.arange(0, matrix_data.shape[1], 1)
        matrix_x, matrix_y = np.meshgrid(x, y)
        amp_init = np.matrix(matrix_data).max()
        stdev_init = 0.33 * halfsize

        def tie_stddev(model):  # we need this for tying x_std and y_std
            xstddev = model.x_stddev
            return xstddev

        g_init = models.Gaussian2D(x_mean=halfsize + 0.5, y_mean=halfsize + 0.5, x_stddev=stdev_init,
                                   y_stddev=stdev_init, amplitude=amp_init, tied={'y_stddev': tie_stddev})

        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, matrix_x, matrix_y, matrix_data)
        if (g.y_stddev < 0) or (g.y_stddev > halfsize):
            raise ValueError('Cannot trust the model, please try other imput parameters.')

        seeing = 2.355 * g.y_stddev * self.pixelsize.to('arcsec')  # in arcsecs
        print('FWHM={:.2f}'.format(seeing))
        print('stddev from the 2D gaussian = {:.3f}'.format(g.y_stddev * self.pixelsize.to('arcsec')))
        return seeing

    def w2p(self, xw, yw):
        """
        Transform from wcs coordinates system to pixel coordinates

        :param self:
        :param xw: float
                   x coordinate in wcs
        :param yw: float
                   y coordinate in wcs
        :return: xpix: float
                       x coordinate in pixels
                 ypix: float
                       y coordinate in pixels
        """
        xpix, ypix = self.gc2.world2pixel(xw, yw)
        if xpix < 0:
            xpix = 0
        if ypix < 0:
            ypix = 0
        return int(round(xpix)), int(round(ypix))

    def p2w(self, xp, yp):
        """
        Transform from pixel coordinate system to wcs coordinates

        :param self:
        :param xp: float
                   x coordinate in pixels
        :param yp: float
                   y coordinate in pixels
        :return: xw: float
                     x coordinate in wcs
                 yw: float
                     y coordinate in wcs
        """
        xw, yw = self.gc2.pixel2world(xp, yp)
        return xw, yw

    def xyr_to_pixel(self, x_center, y_center, radius):
        """
        Transform the (x,y) center and radius that define a circular region from wcs system coordinate to pixels

        :param self:
        :param x_center: float
                         x coordinate in wcs
        :param y_center: float
                         y coordinate in wcs
        :param radius: float
                       radius of the circular region
        :return: x_center_pix: float
                               x coordinate in pixels
                 y_center_pix: float
                               y coordinate in pixels
                 radius_pix: float
                             radius of the circular region in pixels
        """
        x_r = x_center + radius
        x_r_pix, y_center_pix = self.w2p(x_r, y_center)
        x_center, y_center = self.w2p(x_center, y_center)
        radius = abs(x_r_pix - x_center)
        x_center = int(round(x_center))
        y_center = int(round(y_center))
        radius = int(round(radius + 1))
        x_center_pix = x_center
        y_center_pix = y_center
        radius_pix = radius
        return x_center_pix, y_center_pix, radius_pix

    @property
    def shape(self):
        """
        :param self:
        :return:
        """
        return self.cube.data.shape

    def create_movie_redshift_range(self, z_ini=0., z_fin=1., dz=0.001, width=30, outvid='emission_lines_video.avi',
                                    erase=True):
        """
        Function to create a film, colapsing diferent wavelength ranges in which some strong emission lines would fall at certain redshifts
        :param z_ini: initial redshift
        :param z_fin: final redshift
        :param dz: delta redshift
        :param outvid: name of the final video
        :param width: width of the lines that will be collapsed, in Angstroms
        :param erase: If true, the individual frames to make the video will be erased after the video is produced
        :return:
        """
        OII = 3728.483
        wave = self.wavelength
        n = len(wave)
        w_max = wave[n - 1 - 20]
        max_z_allowed = (w_max / OII) - 1.
        if z_fin > max_z_allowed:
            print('maximum redshift allowed is ' + str(max_z_allowed) + ', this value will be used  instead of ' + str(
                z_fin))
            z_fin = max_z_allowed
        z_array = np.arange(z_ini, z_fin, dz)
        images_names = []
        fitsnames = []
        for z in z_array:
            print('z = ' + str(z))
            ranges = self.create_ranges(z, width=width)
            filename = 'emission_line_image_redshif_' + str(z) + '_'
            image = self.get_image_wv_ranges(wv_ranges=ranges, fitsname=filename + '.fits', save=True)
            plt.close(15)
            image = aplpy.FITSFigure(filename + '.fits', figure=plt.figure(15))
            image.show_grayscale()
            plt.title('Emission lines image at z = ' + str(z))
            image.save(filename=filename + '.png')
            images_names.append(filename + '.png')
            fitsnames.append(filename + '.fits')
            plt.close(15)
        video = self.make_video(images=images_names, outvid=outvid)
        n_im = len(fitsnames)
        if erase:
            for i in xrange(n_im):
                fits_im = fitsnames[i]
                png_im = images_names[i]
                command_fits = 'rm ' + fits_im
                command_png = 'rm ' + png_im
                os.system(command_fits)
                os.system(command_png)
        return video

    def collapse_highSN(self, sn_min=5, fitsname='collapsed_emission_image.fits', save=True):
        """
        Function used to sum only voxels in which the signal to noise is greater that sn_min value. This will create a new image
        :param sn_min: float
                       threshold to signal to noise
        :param fitsname: string
                         name of the new image
        :param save: Boolean
                     If True, the new image is saved to the hard disk.
        :return:
        """
        count_voxel_cube = np.where(self.cube > (self.stat ** 0.5) * sn_min, 1., 0.)
        count_voxel_im = np.sum(count_voxel_cube, axis=0) + 1
        del count_voxel_cube
        valid_voxel_cube = np.where(self.cube > (self.stat ** 0.5) * sn_min, self.cube, 0.)
        valid_voxel_im = np.sum(valid_voxel_cube, axis=0)
        del valid_voxel_cube
        normalized_im = valid_voxel_im / count_voxel_im
        normalized_im = np.where(np.isnan(normalized_im), 0, normalized_im)
        if save:
            hdulist = self.hdulist_white
            hdulist[1].data = normalized_im
            hdulist.writeto(fitsname, clobber=True)
        return normalized_im

    def create_ranges(self, z, width=30.):
        """
        Function used to create the wavelength ranges around strong emission lines at a given redshift
        :param z: redshift
        :param width: width  in Angstroms of the emission lines
        :return:
        """
        wave = self.wavelength
        n = len(wave)
        w_max = wave[n - 1]
        w_min = wave[0]
        half = width / 2.
        OII = 3728.483
        Hb = 4862.683
        Ha = 6564.613
        OIII_4959 = 4960.295
        OIII_5007 = 5008.239
        lines_wvs = {'OII': OII * (1. + z), 'Hb': Hb * (1. + z), 'OIII_4959': OIII_4959 * (1. + z),
                     'OIII_5007': OIII_5007 * (1. + z), 'Ha': Ha * (1. + z)}
        range_OII = np.array([lines_wvs['OII'] - half, lines_wvs['OII'] + half])
        range_Hb = np.array([lines_wvs['Hb'] - half, lines_wvs['Hb'] + half])
        range_Ha = np.array([lines_wvs['Ha'] - half, lines_wvs['Ha'] + half])
        range_OIII_4959 = np.array([lines_wvs['OIII_4959'] - half, lines_wvs['OIII_4959'] + half])
        range_OIII_5007 = np.array([lines_wvs['OIII_5007'] - half, lines_wvs['OIII_5007'] + half])
        ranges = [range_Ha, range_Hb, range_OII, range_OIII_4959, range_OIII_5007]
        output_ranges = []
        for range in ranges:
            if range[0] - width >= w_min and range[1] + width <= w_max:
                output_ranges.append(range)
        return output_ranges

    def make_video(self, images, outimg=None, fps=2, size=None, is_color=True, format="XVID", outvid='image_video.avi'):
        from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
        fourcc = VideoWriter_fourcc(*format)
        vid = None
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            img = imread(image)
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        return vid

        # Un radio de 4 pixeles es equivalente a un radio de 0.0002 en wcs
