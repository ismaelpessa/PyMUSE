import numpy as np
import numpy.ma as ma
from scipy import interpolate
import math as m
import aplpy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from matplotlib import pyplot as plt
import glob
import os
from linetools.spectra.xspectrum1d import XSpectrum1D

#spec = XSpectrum1D.from


class MuseCube:
    """
    Class made to manage datacubes from MUSE

    """

    def __init__(self, filename_cube, filename_white, pixelsize=0.2*u.arcsec):
        """
        :param filename_cube: string
                              name of the fits file containing the datacube
        :param filename_white: string
                               name of the fits file containing the colapsed datacube
        pixel_size : float or Quantity
            Pixel size of the datacube, if float we assume arcsecs.
        """
        plt.close(1)
        self.cube = filename_cube
        hdulist = fits.open(self.cube)
        self.data = hdulist[1].data
        self.stat = hdulist[2].data
        self.white = filename_white
        self.gc2 = aplpy.FITSFigure(self.white, figure=plt.figure(1))
        self.gc2.show_grayscale()
        self.gc = aplpy.FITSFigure(self.cube, slices=[1], figure=plt.figure(2))
        self.pixelsize = pixelsize
        plt.close(2)

    def min_max_ra_dec(self, exposure_name):
        """
        Funcion to compute the coordinates range in a datacube

        :param exposure_name: string
                              name of the fits file datacube of the exposure
        :return: ra_range: array[]
                           array that contain the minimum and the maximum value for right ascension
                            in the exposure
                 dec_range: array[]
                            array that contain the minimum and the maximum value for declination
                            in the exposure
        """
        exp_hdulist = fits.open(exposure_name)
        exp_data = exp_hdulist[1].data
        nx = len(exp_data[0])
        ny = len(exp_data[0][0])
        corners = [[0, 0], [0, ny - 1], [nx - 1, 0], [nx - 1, ny - 1]]
        x_corners = []
        y_corners = []
        for point in corners:
            x_corners.append(point[0])
            y_corners.append(point[1])
        ra, dec = self.p2w(y_corners, x_corners)
        ra_min = min(ra)
        ra_max = max(ra)
        dec_min = min(dec)
        dec_max = max(dec)
        ra_range = [ra_min, ra_max]
        dec_range = [dec_min, dec_max]
        return ra_range, dec_range

    def create_wavelength_array(self):
        """
        Creates the wavelength array for the spectrum. The values of dw, and limits will depend
        of the data and should be revised.
        :return: w: array[]
                 array which contain an evenly sampled wavelength range
        """
        dw = 1.25
        w = np.arange(4750, 9351.25, dw)
        # print 'wavelength in range ' + str(w[0]) + ' to ' + str(w[len(w) - 1]) + ' and dw = ' + str(dw)
        return w

    def indexOf(self, array, element):
        """
        Function to search a given element in a given array

        :param array: array[]
                      array that will be explored
        :param element: any type
                        element that will be searched in array
        :return: the index of the first location of element in array. -1 if the element is not found
        """
        n = len(array)
        for i in xrange(0, n):
            if float(array[i]) == float(element):
                return i
        return -1

    def closest_element(self, array, element):
        """
        Find in array the closest value to the value of element
        :param array: array[]
                      array of numbers to search for the closest value possible to element
        :param element: float
                        element to match with array
        :return: k: int
                    index of the closest element
        """
        min_dif = 100
        index = -1
        n = len(array)
        for i in xrange(0, n):
            dif = abs(element - array[i])
            if dif < min_dif:
                index = i
                min_dif = dif
        return index

    def __cut_bright_pixel(self, flux, n):
        '''
        cuts the brightest n elements in a sorted flux array

        :param flux: array[]
                     array containing the sorted flux
        :param n: int
                  number of elements to cut
        :return: cuted_flux: array[]
                             array that contains the remaining elements of the flux
        '''
        N = len(flux)
        cuted_flux = []
        for i in xrange(0, N - n):
            cuted_flux.append(flux[i])
        return cuted_flux

    def __rms_determinate(self, k, plotname='rms_test.png'):
        '''
        Function used to check the right number of pixels to cut before measure rms
        :param k: wavelength index
        :return:
        '''
        # import time
        masked_flux = self.__matrix2array(k)
        sorted_flux = np.sort(masked_flux)
        N = len(sorted_flux)
        std_array = []
        n_cuted_elements_array = []
        j_ini = 0  # Indice desde donde empezar a cortar elementos, todos los anteriores son cortados
        sorted_flux = self.__cut_bright_pixel(sorted_flux, j_ini)
        N_new = len(sorted_flux)
        j_fin = N_new - 10  # Indice en donde acabar de cortar elementos

        for j in xrange(0, j_fin):
            if j % 1000 == 0:
                print 'iteracion ' + str(j) + ' de ' + str(j_fin)
            sorted_flux = self.__cut_bright_pixel(sorted_flux, 1)
            std = np.std(sorted_flux)
            n_cuted_elements = j + 1
            std_array.append(std)
            n_cuted_elements_array.append(n_cuted_elements)
        plt.plot(n_cuted_elements_array, std_array)
        plt.savefig(plotname)

    def __xyz_nan_erase(self, x, y, z):
        n = len(z)
        x_out = []
        y_out = []
        z_out = []
        for i in xrange(n):
            if (z[i] != z[i]) == False:
                x_out.append(x[i])
                y_out.append(y[i])
                z_out.append(z[i])
        return x_out, y_out, z_out

    def __matrix2array(self, k, stat=False):
        matrix = self.data[k]
        if stat == True:
            matrix = self.stat[k]
        n1 = len(matrix)
        n2 = len(matrix[0])
        array_flux = []
        for i in xrange(n1):
            for j in xrange(n2):
                array_flux.append(matrix[i][j])
        array_flux_aux = np.where(np.isnan(np.array(array_flux)), -1, np.array(array_flux))
        masked_flux = ma.masked_equal(array_flux_aux, -1)
        return masked_flux

    def __rms_measure(self, k, n=50000):
        '''

        :param k: int
                  index in wavelength array, where the rms will be measured
        :param n: int
                   number of bright pixel to cut before measure rms, default = 40000
        :return: rms: flot
                      value found for rms in the given wavelength
        '''
        flux = self.__matrix2array(k)
        sorted_flux = np.sort(flux)
        cuted_sorted_flux = self.__cut_bright_pixel(sorted_flux, n)
        rms = np.std(cuted_sorted_flux)
        return rms

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
        hdulist_edited = hdulist
        hdulist_edited[hdu] = hdu_element
        return hdulist_edited

    def __save2fitsimage(self, fitsname, data_to_save, stat=False, type='cube', n_figure=2, edit_header=[]):
        if type == 'white':
            hdulist = fits.HDUList.fromfile(self.white)
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
            hdulist = fits.HDUList.fromfile(self.cube)
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

    def __rms_measure2(self, k, threshold=0.5):
        flux = self.__matrix2array(k, stat=False)
        f = flux.data
        fmin = min(f)
        fmax = max(f)
        nbins = 2000
        bin = np.linspace(0, fmax, nbins)
        plt.close(10)
        plt.figure(10)
        n, bines, patch = plt.hist(f, bins=bin)
        bin_new = np.linspace(min(bin), max(bin), nbins - 1)
        plt.close(10)
        n_norm = self.__normalize2max(n)
        plt.plot(bin_new, n_norm)
        limit_hist_index = self.closest_element(n_norm, threshold)
        limit_flux = bin_new[limit_hist_index]
        lower_fluxes = self.__cut_over_limit(flux, limit_flux)
        std = np.std(np.array(lower_fluxes))
        return std

    def __cut_over_limit(self, flux_array, upper_limit):
        cuted_flux = []
        for f in flux_array:
            if f <= upper_limit:
                cuted_flux.append(f)
        return cuted_flux

    def rms_normalize_stat_2(self, new_cube_name='new_cube_stat_normalized.fits'):
        '''
        Function that creates a new cube with the stat dimension normalized.
        :param new_cube_name: string
                              name of the fits cube that will be created

        :return:
        '''
        n_wave = len(self.data)
        stat_normalized = []
        print n_wave
        for k in xrange(n_wave):
            print 'iteration ' + str(k) + ' of ' + str(n_wave)
            rms_obs = self.__rms_measure2(k)
            stat = self.__matrix2array(k, stat=True)
            rms_stat = np.median(stat)
            normalization_factor = rms_obs / rms_stat
            stat_normalized.append(self.stat[k] * normalization_factor)
        self.__save2fitsimage(new_cube_name, stat_normalized, stat=True, type='cube')
        print 'New cube saved in ' + new_cube_name
        return stat_normalized

    def __normalize2max(self, array):
        m = max(array)
        normalized_array = []
        for element in array:
            normalized_array.append(element / m)
        return normalized_array

    def define_elipse_region(self,x_center,y_center,a,b,theta,coord_system):
        Xc=x_center
        Yc=y_center
        if coord_system == 'wcs':
            X_aux, Y_aux, a = self.xyr_to_pixel(Xc, Yc, a)
            Xc,Yc,b=self.xyr_to_pixel(Xc, Yc, b)
        if b > a:
            aux = a
            a = b
            b = aux
            theta = theta + np.pi / 2.
        x = np.linspace(-a, a, 5000)

        B = x * np.sin(2. * theta) * (a ** 2 - b ** 2)
        A = (b * np.sin(theta)) ** 2 + (a * np.cos(theta)) ** 2
        C = (x ** 2) * ((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2) - (a ** 2) * (b ** 2)

        y_positive = (-B + np.sqrt(B ** 2 - 4. * A * C)) / (2. * A)
        y_negative = (-B - np.sqrt(B ** 2 - 4. * A * C)) / (2. * A)

        y_positive_2 = y_positive + Yc
        y_negative_2 = y_negative + Yc
        x_2 = x + Xc
        reg = []
        for i in xrange(Xc-a-1,Xc+a+1):
            for j in xrange(Yc-a-1,Yc+a+1):
                k=self.closest_element(x_2,i)
                if j<=y_positive_2[k] and j>=y_negative_2[k]:
                    reg.append([i,j])


        return reg


    def rms_normalize_stat(self, new_cube_name='new_cube_stat_normalized.fits', n=50000):
        '''
        Function that creates a new cube with the stat dimension normalized
        :param new_cube_name: string
                              name of the fits cube that will be created
               n: float
                  number of pixels that will be ignored to calculate the observed rms
        :return:
        '''
        n_wave = len(self.data)
        stat_normalized = []
        print n_wave
        for k in xrange(n_wave):
            print k
            rms_obs = self.__rms_measure(k, n=n)
            stat = self.__matrix2array(k, stat=True)
            rms_stat = np.median(stat)
            normalization_factor = rms_obs / rms_stat
            stat_normalized.append(self.stat[k] * normalization_factor)

        self.__save2fitsimage(new_cube_name, stat_normalized, stat=True, type='cube')
        print 'New cube saved in ' + new_cube_name
        return stat_normalized

    def __find_wavelength_index(self, wavelength):
        wave = self.create_wavelength_array()
        if wavelength < min(wave) or wavelength > max(wave):
            raise ValueError('Longitud de onda dada no esta dentro del rango valido')
        elif wavelength >= min(wave) and wavelength <= max(wave) and self.indexOf(wave, wavelength) == -1:
            print 'Longitud de onda en rango valido, pero el valor asignado no esta definido'
            k = int(self.closest_element(wave, wavelength))
            print 'Se usara wavelength = ' + str(wave[k])
        elif wavelength >= min(wave) and wavelength <= max(wave) and self.indexOf(wave, wavelength) >= 0:
            k = self.indexOf(wave, wavelength)
        return k

    def __matrix2line(self, matrix):
        n1 = len(matrix)
        n2 = len(matrix[0])
        out = ''
        for i in xrange(n1):
            for j in xrange(n2):
                out = out + str(matrix[i][j])
                if j < n2 - 1:
                    out = out + ' '
            if i < n1 - 1:
                out = out + ';'
        return out

    def __line2file(self, line, filename):
        f = open(filename, 'a')
        f.write(line + '\n')
        f.close()
        return

    def __filelines2cube(self, filename):
        cube = []
        f = open(filename, 'r')
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            matrix = np.matrix(line)
            cube.append(matrix)
        return cube

    def create_combined_cube(self, exposure_names, kind='sum', fitsname='new_combined_cube.fits'):
        wave = self.create_wavelength_array()
        new_cube = []
        for w in wave:
            print 'wavelength ' + str(w) + ' of ' + str(max(wave))
            combined_matrix, interpolated_fluxes, values_list = self.combine_not_aligned(exposure_names=exposure_names,
                                                                                         wavelength=w, kind=kind)
            matrix_line = self.__matrix2line(combined_matrix)
            self.__line2file(matrix_line, 'matrix.dat')
            # new_cube.append(combined_matrix)
        new_cube = self.__filelines2cube('matrix.dat')
        self.__save2fitsimage(fitsname, new_cube, type='cube', stat=False, edit_header=[values_list])
        print 'New cube saved in ' + fitsname

    def combine_not_aligned(self, exposure_names, wavelength, n_resolution=50000, kind='std'):
        k = self.__find_wavelength_index(wavelength)
        ra_min_exposures = []
        ra_max_exposures = []
        dec_min_exposures = []
        dec_max_exposures = []
        for exposure in exposure_names:
            ra_range, dec_range = self.min_max_ra_dec(exposure)
            ra_min_exposures.append(ra_range[0])
            ra_max_exposures.append(ra_range[1])
            dec_min_exposures.append(dec_range[0])
            dec_max_exposures.append(dec_range[1])
        ra_min_all = min(ra_min_exposures)
        ra_max_all = max(ra_max_exposures)
        dec_min_all = min(dec_min_exposures)
        dec_max_all = max(dec_max_exposures)
        n_ra = int(round((ra_max_all - ra_min_all) * n_resolution))
        n_dec = int(round((dec_max_all - dec_min_all) * n_resolution))
        print 'n_ra = ' + str(n_ra)
        print 'n_dec = ' + str(n_dec)
        ra2 = list(np.linspace(ra_min_all, ra_max_all, n_ra))
        ra2.reverse()
        ra_array = np.array(ra2)
        dec_array = np.linspace(dec_min_all, dec_max_all, n_dec)

        interpolated_fluxes = []

        for exposure in exposure_names:
            hdulist = fits.open(exposure)
            data = hdulist[1].data
            n1 = len(data[0])
            n2 = len(data[0][0])
            im = aplpy.FITSFigure(exposure, figure=plt.figure(10), slices=[1])
            plt.close(10)
            ra = []
            dec = []
            for i in xrange(n1):
                for j in xrange(n2):
                    x_wcs, y_wcs = im.pixel2world(j, i)
                    if i == n1 - 1:
                        ra.append(x_wcs)
                    if j == n2 - 1:
                        dec.append(y_wcs)
                        # flux.append(data[k][i][j])
            data_aux = np.where(np.isnan(data[k]), -1, data[k])
            data_aux_masked = ma.masked_equal(data_aux, -1)
            interpolator = interpolate.interp2d(ra, dec, data_aux_masked, bounds_error=False, fill_value=np.nan)
            flux_new = interpolator(ra_array, dec_array)
            flux_new2 = np.zeros_like(flux_new)
            m1 = len(flux_new)
            m2 = len(flux_new[0])
            for i in xrange(m1):
                for j in xrange(m2):
                    flux_new2[i][j] = flux_new[i][m2 - 1 - j]

            interpolated_fluxes.append(flux_new2)

        matrix_combined = self.__calculate_combined_matrix(interpolated_fluxes, kind=kind)
        delta_ra = ra_array[1] - ra_array[0]
        delta_dec = dec_array[1] - dec_array[0]
        central_ra = ra_array[n_ra / 2]
        central_dec = dec_array[n_dec / 2]
        central_j = n_ra / 2
        central_i = n_dec / 2

        data_to_header = [central_j, central_i, delta_ra, delta_dec, central_ra, central_dec]
        return matrix_combined, interpolated_fluxes, data_to_header

    def __calculate_combined_matrix(self, interpolated_fluxes, kind='std'):
        '''

        :param interpolated_fluxes: array[]
                                    array of matrix, each matrix contain the flux of an exposure at a given wavelength
               kind: string
                     default = 'std', possible values = 'std, 'ave','sum. Kind of combination of the input matrix in each point
        :return:
        '''
        matrix_combined = np.zeros_like(interpolated_fluxes[0])
        n1 = len(matrix_combined)
        n2 = len(matrix_combined[0])
        for i in xrange(n1):
            for j in xrange(n2):
                aux_array = []
                for matrix in interpolated_fluxes:
                    aux_array.append(matrix[i][j])
                if kind == 'std':
                    matrix_combined[i][j] = np.nanstd(aux_array)
                if kind == 'ave':
                    matrix_combined[i][j] = np.nanmean(aux_array)
                if kind == 'sum':
                    matrix_combined[i][j] = np.nansum(aux_array)
        return matrix_combined

    def __read_files(self, input):
        path = input
        files = glob.glob(path)
        return files

    def calculate_error(self, exposures_names, wavelength, fitsname='errors.fits', n_figure=2):
        """
        From differents exposures of a field, this function computes the error, by calculating the
        std of all exposures, in all the field, for a given wavelength. The exposures must be aligned.
        :param exposures_names: array[]
                                array containing the names of the differents exposures in each position
        :param wavelength: float
                           the wavelength in which the error will be calculated
        :param fitsname: string, default = 'errors.fits'
                         name of the fits file where the error image will be saved
        :param n_figure: int, d efault = 2
                         number of the figure where the image will be displayed
        :return:
        """
        k = self.__find_wavelength_index(wavelength)

        data_matrix_array = []
        for exposure in exposures_names:
            hdulist = fits.open(exposure)
            data_exposure = hdulist[1].data
            print data_exposure.shape
            Nw = len(data_exposure)
            Nx = len(data_exposure[0])
            Ny = len(data_exposure[0][0])
            matrix = np.array([[0. for y in range(Ny)] for x in range(Nx)])
            for i in xrange(0, Nx):
                for j in xrange(0, Ny):
                    matrix[i][j] = data_exposure[k][i][j]
            data_matrix_array.append(matrix)
        matrix_errors = np.array([[0. for y in range(Ny)] for x in range(Nx)])
        for i in xrange(0, Nx):
            for j in xrange(0, Ny):
                matrix_elements = []
                for matrix in data_matrix_array:
                    matrix_elements.append(matrix[i][j])
                error = np.std(matrix_elements)
                matrix_errors[i][j] = error
        hdulist = fits.HDUList.fromfile(self.white)
        hdulist[1].data = matrix_errors
        hdulist.writeto(fitsname, clobber=True)
        errors = aplpy.FITSFigure(fitsname, figure=plt.figure(n_figure))
        errors.show_grayscale()

    def create_movie_wavelength_range(self, initial_wavelength, final_wavelength, width=5., outvid='image_video.avi',
                                      erase=True):
        wave = self.create_wavelength_array()
        n = len(wave)
        index_ini = int(self.closest_element(wave, initial_wavelength))
        if initial_wavelength < wave[0]:
            print str(
                initial_wavelength) + ' es menor al limite inferior minimo permitido, se usara en su lugar ' + str(
                wave[0])
            initial_wavelength = wave[0]
        if final_wavelength > wave[n - 1]:
            print str(final_wavelength) + ' es mayor al limite superior maximo permitido, se usara en su lugar ' + str(
                wave[n - 1])
            final_wavelength = wave[n - 1]

        images_names = []
        fitsnames = []
        for i in xrange(initial_wavelength, final_wavelength):
            wavelength_range = [i, i + width]
            filename = 'colapsed_image_' + str(i) + '_'
            self.colapse_cube([wavelength_range], fitsname=filename + '.fits', n_figure=15)
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
        return video

    def colapse_cube(self, wavelength, fitsname='new_colapsed_cube.fits', n_figure=2):
        """
        Function that creates a new image, by colapsing some wavelengths of the cube

        :param wavelength: array[]
                           wavelength contain the information of the wavelength that will be colapsed
                           This array can contain eithar individual values, which represents the wavelengths
                           that will be colapsed, or arrays of length = 2, which represents wavelengths ranges that will be
                           colapsed
        :param fitsname: string, default = 'new_colapsed_cube.fits'
                         name of the fits image that will be created
        :param n_figure: int, default = 2
                         number of the figure that will display the new image created

        :return:
        """
        if type(wavelength[0]) == int or type(wavelength[0]) == float:
            interval = 0
        if type(wavelength[0]) == list or type(wavelength[0]) == numpy.ndarray:
            interval = 1
        if type(wavelength[0]) != int and type(wavelength[0]) != float and type(wavelength[0]) != list and type(
                wavelength[0]) != numpy.ndarray:
            interval = -1
            raise ValueError(
                'Unkown format for wavelength, please use only int and float, or 1-D arrays with 2 elements')

        wave = self.create_wavelength_array()
        wave = list(wave)
        wave_index = []
        n = len(wavelength)
        dw = wave[1] - wave[0]
        if interval == 0:
            for i in xrange(0, n):
                if wavelength[i] < wave[0] or wavelength[i] > wave[len(wave) - 1]:
                    print str(wavelength[i]) + ' is not a valid wavelength value'
                else:
                    index = int(self.closest_element(wave, wavelength[i]))
                    wave_index.append(index)
        if interval == 1:
            for i in xrange(0, n):
                w_up = wavelength[i][1]
                w_low = wavelength[i][0]
                w_aux = w_low
                if w_low < wave[0]:
                    print str(w_low) + ' es menor al valor minimo posible, se usara como limite inferior ' + str(
                        wave[0])
                    w_low = wave[0]
                if w_up > wave[len(wave) - 1]:
                    print str(w_up) + ' es mayor al valor maximo posible, se usara como limite superior ' + str(
                        wave[len(wave) - 1])
                    w_up = wave[len(wave) - 1]

                w_aux = w_low
                while w_aux < w_up:
                    index = int(self.closest_element(wave, w_aux))
                    wave_index.append(index)
                    w_aux += dw
        Nw = len(self.data)
        Nx = len(self.data[0])
        Ny = len(self.data[0][0])
        Matrix = np.array([[0. for y in range(Ny)] for x in range(Nx)])
        image_stacker = Matrix
        for k in wave_index:
            for i in xrange(0, Nx):
                for j in xrange(0, Ny):
                    Matrix[i][j] = self.data[k][i][j]
        image_stacker = image_stacker + Matrix
        self.__save2fitsimage(fitsname, image_stacker, type='white', n_figure=n_figure)
        print 'Imaged writed in ' + fitsname

    def normalize_sky(self, flux_sky, normalization_factor):
        """
        Function used to normalize the flux of the sky, by the number of pixels used in it
        :param flux_sky: array[]
                         array containing the flux to normalize
        :param normalization_factor: float
                                     factor to normalize the flux. It should be the cuocient between
                                     the number of pixel in the region of the soruce and the number
                                     of pixels in the sky ring-region
        :return: flux_normalized: array[]
                                  array containing the normalized flux of the sky
        """
        n = len(flux_sky)
        flux_normalized = []
        for f in flux_sky:
            flux_normalized.append(f * normalization_factor)
        return flux_normalized

    def substract_spec(self, spec, sky_spec):
        """

        :param spec: array[]
                     flux array of the region
        :param sky_spec: array[]
                         flux array of the sky
        :return: substracted_spec: array[]
                                   flux array of the region with the substraction of the sky
        """
        substracted_spec = []
        n = len(spec)
        for i in xrange(0, n):
            substracted_spec.append(spec[i] - sky_spec[i])
        return substracted_spec

    def plot_region_spectrum_sky_substraction(self, x_center, y_center, radius, sky_radius_1, sky_radius_2,
                                              coord_system, n_figure=2,errors = False):
        """
        Function to obtain and display the spectrum of a source in circular region of R = radius,
        substracting the spectrum of the sky, obtained in a ring region around x_center and y_center,
        with internal radius = sky_radius_1 and external radius = sky_radius_2


        :param x_center: float
                         x coordinate of the center of the circular region
        :param y_center: float
                         y coordinate of the center of the circular region
        :param radius: float
                       radius of the circular region. In the case of an eliptical region, radius must be an 1-D array
                       of length = 3, containing [a,b,theta] parameters of the elipse
        :param sky_radius_1: float
                             internal radius of the ring where the sky will be calculated
        :param sky_radius_2: float
                             external radius of the ring where the sky will be calculated
        :param coord_system: string
                             possible values: 'wcs', 'pix', indicates the coordinante system used.
        :param n_figure: int, default = 2
                         figure number to display the spectrum
        :return: w: array[]
                    array with the wavelength of the spectrum
                 substracted_sky_spec: array[]
                                       array with the flux of the sky-substracted spectrum
        """

        w, spec = self.spectrum_region(x_center, y_center, radius, coord_system, debug=False)
        if errors:
            w,err=self.spectrum_region(x_center, y_center, radius, coord_system, debug=False,stat=True)
        w_sky, spec_sky = self.spectrum_ring_region(x_center, y_center, sky_radius_1, sky_radius_2, coord_system)
        self.draw_circle(x_center, y_center, sky_radius_1, 'Blue', coord_system)
        self.draw_circle(x_center, y_center, sky_radius_2, 'Blue', coord_system)
        if type(radius)==int or type(radius)==float:
            self.draw_circle(x_center, y_center, radius, 'Green', coord_system)
            reg = self.define_region(x_center, y_center, radius, coord_system)
        else:
            a=radius[0]
            b=radius[1]
            theta=radius[2]
            self.draw_elipse(x_center,y_center,a,b,theta,'Green',coord_system)
            reg = self.define_elipse_region(x_center,y_center,a,b,theta,coord_system)

        ring = self.define_ring_region(x_center, y_center, sky_radius_1, sky_radius_2, coord_system)
        # print ring
        # print reg
        normalization_factor = float(len(reg)) / len(ring)
        print normalization_factor
        spec_sky_normalized = self.normalize_sky(spec_sky, normalization_factor)
        substracted_sky_spec = np.array(self.substract_spec(spec, spec_sky_normalized))
        plt.figure(n_figure)
        plt.plot(w, substracted_sky_spec)
        plt.plot(w, spec_sky_normalized)
        if errors:
            spec_tuple=(w,substracted_sky_spec,err)
        else:
            spec_tuple=(w,substracted_sky_spec)

        spectrum = XSpectrum1D.from_tuple(spec_tuple)
        return spectrum



        # plt.plot(w,w_sky)#print spec_sky

    def clean_canvas(self):
        """
        Clean everything from the canvas with the colapsed cube image
        :param self:
        :return:
        """
        plt.close(1)
        self.gc2 = aplpy.FITSFigure(self.white, figure=plt.figure(1))


    def create_table(self,input_file):
        from astropy.io.ascii.sextractor import SExtractor
        sex = SExtractor()
        table=sex.read(input_file)
        return table

    def get_from_table(self,input_file,keyword):
        table=self.create_table(input_file)
        data = table[keyword]
        return data

    def determinate_seeing_from_white(self, xc, yc, halfsize):
        from astropy.modeling import models, fitting
        hdulist = fits.open(self.white)
        data = hdulist[1].data
        w,h = 2*halfsize+1,2*halfsize+1
        matrix_data = [[0 for x in range(w)] for y in range(h)]
        matrix_x= [[0 for x in range(w)] for y in range(h)]
        matrix_y= [[0 for x in range(w)] for y in range(h)]
        for i in xrange(xc-halfsize,xc+halfsize+1):
            for j in xrange(yc-halfsize,yc+halfsize+1):
                i2=i-(xc-halfsize)
                j2=j-(yc-halfsize)
                matrix_data[j2][i2]=data[j][i]
                matrix_x[j2][i2]=i2
                matrix_y[j2][i2]=j2
        amp_init = np.matrix(matrix_data).max()
        stdev_init = 0.33*halfsize
        def tie_stddev(model):  # we need this for tying x_std and y_std
            xstddev = model.x_stddev
            return xstddev
        g_init = models.Gaussian2D(x_mean=halfsize+0.5,y_mean=halfsize + 0.5,x_stddev=stdev_init,
                                   y_stddev=stdev_init,amplitude=amp_init, tied={'y_stddev': tie_stddev})

        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, matrix_x, matrix_y, matrix_data)
        if (g.y_stddev < 0) or (g.y_stddev > halfsize):
            raise ValueError('Cannot trust the model, please try other imput parameters.')

        seeing = 2.355 * g.y_stddev * self.pixelsize.to('arcsec')  # in arcsecs
        print 'FWHM={:.2f}'.format(seeing)
        return seeing











    def write_coords(self, filename):
        """

        :param self:
        :param filename: string
                         Name of the file that will contain the coordinates
        :param Nx:int
                  Number of pixels of the image in the x-axis
        :param Ny:int
                  Number of pixels of the image in the y-axis
        :return:
        """
        Nx = len(self.data[0])
        Ny = len(self.data[0][0])
        f = open(filename, 'w')
        for i in xrange(0, Nx):
            for j in xrange(0, Ny):
                x_world, y_world = self.gc.pixel2world(np.array([i]), np.array([j]))
                c = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
                f.write(str(i) + '\t' + str(j) + '\t' + str(x_world[0]) + '\t' + str(y_world[0]) + '\t' + str(
                    c.to_string('hmsdms')) + '\n')
        f.close()

    def is_in_ring(self, x_center, y_center, radius_1, radius_2, x, y):
        """
        Function to sheck if (x,y) is inside the ring region with inner radius = radius_1 and outer
        radius = radius_2.
        :param x_center: float
                         x-coordinate of the center of the region
        :param y_center: float
                         y-coordinate of the center of the region
        :param radius_1:  float
                          inner radius of the region
        :param radius_2: float
                         outer radius of the region
        :param x: float
                  x-coordinate of the point that will be checked
        :param y: float
                  y-coordinate of the point that will be checked
        :return: boolean:
                 True if (x,y) is inside the ring, False if not
        """
        if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius_2 ** 2 and (x - x_center) ** 2 + (
                    y - y_center) ** 2 >= radius_1 ** 2:
            return True
        return False

    def is_in_reg(self, x_center, y_center, radius, x, y):
        """
        Determines if the pair (x,y) is inside the circular region centered in (x_center,y_center) and r=radius
        returns True or False
        :param self:
        :param x_center: float
                         x coordinate of the center of the circular region
        :param y_center: float
                         y coordinate of the center of the circular region
        :param radius: float
                       radius of the circular region
        :param x: float
                  x coordinate of the pair
        :param y: float
                  y coordinate of the pair
        :return: boolean
                 True if (x,y) is inside the region, False if not
        """

        if (x - x_center) ** 2 + (y - y_center) ** 2 < radius ** 2:
            return True
        return False

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
        xpix, ypix = self.gc.world2pixel(xw, yw)
        if xpix < 0:
            xpix = 0
        if ypix < 0:
            ypix = 0
        return xpix, ypix

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
        xw, yw = self.gc.pixel2world(xp, yp)
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

    def define_ring_region(self, x_center, y_center, radius_1, radius_2, coord_system):
        if coord_system == 'wcs':
            x_center_temp, y_center_temp, radius_1 = self.xyr_to_pixel(x_center, y_center, radius_1)
            x_center, y_center, radius_2 = self.xyr_to_pixel(x_center, y_center, radius_2)

        reg = []
        # print y_center
        if x_center - radius_2 - 3 > 0 and y_center - radius_2 - 3 > 0:
            for i in xrange(x_center - radius_2 - 3, x_center + radius_2 + 4):
                for j in xrange(y_center - radius_2 - 3, y_center + radius_2 + 4):
                    if self.is_in_ring(x_center, y_center, radius_1, radius_2, i, j):
                        reg.append([i, j])
        if x_center - radius_2 - 3 < 0 and y_center - radius_2 - 3 > 0:
            for i in xrange(0, x_center + radius_2 + 4):
                for j in xrange(y_center - radius_2 - 3, y_center + radius_2 + 4):
                    if self.is_in_ring(x_center, y_center, radius_1, radius_2, i, j):
                        reg.append([i, j])
        if x_center - radius_2 - 3 > 0 and y_center - radius_2 - 3 < 0:
            for i in xrange(x_center - radius_2 - 3, x_center + radius_2 + 4):
                for j in xrange(0, y_center + radius_2 + 4):
                    if self.is_in_ring(x_center, y_center, radius_1, radius_2, i, j):
                        reg.append([i, j])
        if x_center - radius_2 - 3 < 0 and y_center - radius_2 - 3 < 0:
            for i in xrange(0, x_center + radius_2 + 4):
                for j in xrange(0, y_center + radius_2 + 4):
                    if self.is_in_ring(x_center, y_center, radius_1, radius_2, i, j):
                        reg.append([i, j])
        return reg

    def define_region(self, x_center, y_center, radius, coord_system):
        """
        Define a circular region by returning al the pixels which are inside the region

        :param self:
        :param x_center: float
                         x coordinate of the center of the region
        :param y_center: float
                         y coordinate of the center of the region
        :param radius: float
                       radius of the circular region
        :param coord_system: string
                             possible values: 'wcs', 'pix', indicates the coordinante system used.
        :return: reg: array[]
                      reg contains all the (x,y) pairs which are inside the region. The (x,y) are in pixels
                      coordinates
        """
        if coord_system == 'wcs':
            x_center, y_center, radius = self.xyr_to_pixel(x_center, y_center, radius)

        reg = []
        # print x_center,y_center,radius
        if x_center - radius - 3 > 0 and y_center - radius - 3 > 0:
            for i in xrange(x_center - radius - 3, x_center + radius + 4):
                for j in xrange(y_center - radius - 3, y_center + radius + 4):
                    if self.is_in_reg(x_center, y_center, radius, i, j):
                        reg.append([i, j])
        if x_center - radius - 3 < 0 and y_center - radius - 3 > 0:
            for i in xrange(0, x_center + radius + 4):
                for j in xrange(y_center - radius - 3, y_center + radius + 4):
                    if self.is_in_reg(x_center, y_center, radius, i, j):
                        reg.append([i, j])
        if x_center - radius - 3 > 0 and y_center - radius - 3 < 0:
            for i in xrange(x_center - radius - 3, x_center + radius + 4):
                for j in xrange(0, y_center + radius + 4):
                    if self.is_in_reg(x_center, y_center, radius, i, j):
                        reg.append([i, j])
        if x_center - radius - 3 < 0 and y_center - radius - 3 < 0:
            for i in xrange(0, x_center + radius + 4):
                for j in xrange(0, y_center + radius + 4):
                    if self.is_in_reg(x_center, y_center, radius, i, j):
                        reg.append([i, j])
        return reg

    def test_plot_reg(self, reg):
        """
        Plot all the pixels that are contained in a region. The input reg is an array, output from the function
        self.define_region, which has all the pixel contained in a region
        :param self:
        :param reg: array[]
                    array that in every space contains an (x,y) pair.
        :return:
        """
        for i in xrange(0, len(reg)):
            plt.figure(1)
            plt.plot(reg[i][0], reg[i][1], 'o')

    def draw_elipse(self,Xc,Yc,a,b,theta,color,coord_system):
        """
        Draw an elipse centered in (Xc,Yc) with semiaxis a and b, and a rotation angle theta
        :param Xc: float
                   x coordinate of the center of the elipse
        :param Yc: float
                   y coordinate of the center of the elipse
        :param a: float
                  semiaxis in the x axis of the elipse
        :param b: float
                  semiaxis in the y axis of the elipse
        :param theta: float
                      rotation angle between elipse and x axis
        :param color: string
                      Color of the elipse
        :param coord_system: string
                             possible values: 'wcs','pix', indicates the coordinate system used.
        :return:
        """
        if coord_system == 'wcs':
            X_aux, Y_aux, a = self.xyr_to_pixel(Xc, Yc, a)
            Xc,Yc,b=self.xyr_to_pixel(Xc, Yc, b)
        if b > a:
            aux = a
            a = b
            b = aux
            theta = theta + np.pi / 2.
        x = np.linspace(-a, a, 5000)

        B = x * np.sin(2. * theta) * (a ** 2 - b ** 2)
        A = (b * np.sin(theta)) ** 2 + (a * np.cos(theta)) ** 2
        C = (x ** 2) * ((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2) - (a ** 2) * (b ** 2)

        y_positive = (-B + np.sqrt(B ** 2 - 4. * A * C)) / (2. * A)
        y_negative = (-B - np.sqrt(B ** 2 - 4. * A * C)) / (2. * A)

        y_positive_2 = y_positive + Yc
        y_negative_2 = y_negative + Yc
        x_2 = x + Xc
        plt.figure(1)
        plt.plot(x_2, y_positive_2,color)
        plt.plot(x_2, y_negative_2,color)


    def draw_circle(self, Xc, Yc, R, color, coord_system):
        """
        Draw a circle, centered in (Xc,Yc) with radius R.
        :param self:
        :param Xc: float
                   x coordinate of the center of the circle
        :param Yc: float
                   y coorcinate of the center of the circle
        :param R: float
                  radius of the circle
        :param color: string
                      color of the circles that will be drawn
        :param coord_system: string
                             possible values: 'wcs', 'pix', indicates the coordinate system used.
        :return:
        """

        if coord_system == 'wcs':
            Xc, Yc, R = self.xyr_to_pixel(Xc, Yc, R)
        N = 10000
        X1 = np.linspace(Xc - R, Xc + R, N / 2)
        X2 = np.linspace(Xc - R, Xc + R, N / 2)
        Y1 = range(len(X1))
        Y2 = range(len(X2))
        for i in xrange(0, len(X1)):
            if ((X1[i] - Xc) ** 2) - R ** 2 >= 0:
                Y1[i] = m.sqrt(((X1[i] - Xc) ** 2) - R ** 2) + Yc
            if ((X1[i] - Xc) ** 2) - R ** 2 < 0:
                Y1[i] = m.sqrt(-(((X1[i] - Xc) ** 2) - R ** 2)) + Yc
        for i in xrange(0, len(X2)):
            if ((X2[i] - Xc) ** 2) - R ** 2 >= 0:
                Y2[i] = -m.sqrt(((X2[i] - Xc) ** 2) - R ** 2) + Yc
            if ((X2[i] - Xc) ** 2) - R ** 2 < 0:
                Y2[i] = -m.sqrt(-(((X2[i] - Xc) ** 2) - R ** 2)) + Yc
        plt.figure(1)
        plt.plot(X1, Y1, color)
        plt.plot(X2, Y2, color)

    def size(self):
        """
        Print the dimension size (x,y,lambda) of the datacube.
        :param self:
        :return:
        """
        input = self.cube
        hdulist = fits.open(input)
        hdulist.info()
        self.data.shape
        print 'X,Y,Lambda'

    def get_spectrum_point_aplpy(self, x, y, coord_system,stat=False):
        """
        Obtain the spectrum of a given point defined by (x,y) in the datacube
        :param self:
        :param x: float
                  x coordinate of the point
        :param y: float
                  y coordinate of the point
        :param coord_system: string
                             possible values: 'wcs', 'pix', indicates the coordinate system used.
        :param stat: boolean, default = False
                     if true, the spectra will be obtained from the stat image instead
        :return: wave: array[]
                       array with the wavelegth of the spectrum.
                 spec: array[]
                       array with the flux of the spectrum.
        """
        # gc = aplpy.FITSFigure(input,slices=[1])
        # gc.show_grayscale()
        if coord_system == 'wcs':
            x_pix, y_pix = self.w2p(x, y)
        if coord_system == 'pix':
            x_pix = x
            y_pix = y

        # DATA.shape ##Z,X,Y
        Nw = len(self.data)
        Nx = len(self.data[0])
        Ny = len(self.data[0][0])
        wave = self.create_wavelength_array()
        spec = []
        data = self.data
        if stat:
            data=self.stat
        for i in xrange(0, len(wave)):
            #print x_pix,y_pix
            #print len(self.data[0])
            spec.append(data[i][y_pix][x_pix])
        # figure(2)
        # plt.plot(wave,spec)
        # plt.show()
        return wave, spec

    def spectrum_ring_region(self, x_center, y_center, radius_1, radius_2, coord_system, debug=False):
        input = self.cube
        # print x_center
        Region = self.define_ring_region(x_center, y_center, radius_1, radius_2, coord_system)
        N = len(Region)
        S = []
        for i in xrange(0, N):
            S.append(self.get_spectrum_point_aplpy(Region[i][0], Region[i][1], 'pix'))
        wave = S[0][0]
        specs = []
        for i in xrange(0, N):
            specs.append(S[i][1])
        combined_spec = range(len(specs[0]))

        lambda_aux = []
        for j in xrange(0, len(specs[0])):
            for i in xrange(0, len(specs)):
                lambda_aux.append(specs[i][j])
            if debug:
                bins = np.linspace(np.min(lambda_aux), np.max(lambda_aux), 20)
                plt.hist(lambda_aux, bins)
                plt.show()

            combined_spec[j] = np.nansum(lambda_aux)

            lambda_aux = []
        return wave, combined_spec

    def spectrum_region(self, x_center, y_center, radius, coord_system, debug=False, stat = False):
        """
        Obtain the spectrum of a given region in the datacube, defined by a center (x_center,y_center), and
        radius. In the case of circular region, radius is a number, in the case of eliptical region, radius in an array that
        must contain a,b and angle.
        :param self:
        :param x_center: float
                         x coordinate of the center of the region
        :param y_center: float
                         y coordinate of the center of the region
        :param radius: float or 1-D array
                       radius of the circular region or the [a,b,theta] that defines an eliptical region
        :param coord_system: string
                             possible values: 'wcs, 'pix', indicates the coordinate system used.
        :return: wave: array[]
                       array with the wavelength of the spectrum
                 combined_spec: array[]
                                array with the flux of the spectrum. This flux is the sum of the fluxes of all
                                pixels in the region.

        """

        input = self.cube

        if type(radius) == int or type(radius)==float:

            Region = self.define_region(x_center, y_center, radius, coord_system)
        else:
            a=radius[0]
            b=radius[1]
            theta = radius[2]
            Region = self.define_elipse_region(x_center,y_center,a,b,theta,coord_system)

        N = len(Region)
        S = []
        for i in xrange(0, N):
            S.append(self.get_spectrum_point_aplpy(Region[i][0], Region[i][1], 'pix',stat=stat))
        wave = S[0][0]
        specs = []
        for i in xrange(0, N):
            specs.append(S[i][1])
        combined_spec = range(len(specs[0]))

        lambda_aux = []
        for j in xrange(0, len(specs[0])):
            for i in xrange(0, len(specs)):
                lambda_aux.append(specs[i][j])
            if debug:
                bins = np.linspace(np.min(lambda_aux), np.max(lambda_aux), 20)
                plt.hist(lambda_aux, bins)
                plt.show()

            if stat == False:
                combined_spec[j] = np.nansum(lambda_aux)
            else:
                combined_spec[j] = np.sqrt(np.nansum(np.array(lambda_aux)**2))


            lambda_aux = []
        return wave, combined_spec

    def plot_region_spectrum(self, x_center, y_center, radius, coord_system, color='Green', n_figure=2, debug=False):
        """
        Plot over the canvas a circle region, and aditionally, plots it's spectrum in other figure
        :param x_center: float
                         x coordinate of the circular region
        :param y_center: float
                         y coordinate of the circular region
        :param radius: float
                       radius of the circular region
        :param coord_system: string
                             possible vales: 'wcs', 'pix', indicates de coordinate system used
        :param color: string
                      default: 'Green'. Color of the circle to be drawn
        :param n_figure: int, default = 2
                         figure number to display the spectrum
        :return: w: array[]
                    array with the wavelength of the spectrum
                 spec: array[]
                       array with the flux of the spectrum


         """
        input = self.cube
        plt.figure(1)
        self.draw_circle(x_center, y_center, radius, color, coord_system)
        w, spec = self.spectrum_region(x_center, y_center, radius, coord_system, debug=debug)
        plt.figure(n_figure)
        plt.plot(w, spec)
        return w, spec

    def substring(self, string, i, j):
        """
        Obtain a the substring of string, from index i to index j, both included

        :param self:
        :param string: string
                       input string
        :param i: int
                  initial index
        :param j: int
                  final index
        :return: out: string
                      output substring
        """
        out = ''
        for k in xrange(i, j + 1):
            out = out + string[k]
        return out

    def read_circle_line(self, circle_line):
        a = circle_line.split('(')
        b = a[1]
        c = b.split(')')
        d = c[0]
        e = d.split(',')
        return float(e[0]), float(e[1]), float(e[2])

    def read_region_file(self, regionfile):
        """
        Read a ds9 region file. The region file must be in physical coordinates. The output is an array with the
        center (x,y) and radius that defines all circular regions
        :param self:
        :param regionfile: string
                           name of the region file
        :return: reg: array[]
                      array that in each space contains the (x,y,r) parameters that define a region
        """
        f = open(regionfile, 'r')
        lines = f.readlines()
        circles = []
        for line in lines:
            if len(line) > 5:
                if self.substring(line, 0, 5) == 'circle':
                    circles.append(line)

        reg = []
        for line in circles:
            x, y, r = self.read_circle_line(line)
            x = int(round(x))
            y = int(round(y))
            r = int(round(r))
            reg.append([x, y, r])
        return reg

    def plot_region_file(self, regionfile):
        """
        Plot and shows the spectrum of al regions in a ds9 region file. The region must be defined in phisical
        coordinates.
        :param self:
        :param regionfile: string
                           name of the region file
        :return:
        """
        regiones = self.read_region_file(regionfile)
        for i in xrange(0, len(regiones)):
            w, spec = self.plot_region_spectrum(regiones[i][0], regiones[i][1], regiones[i][2], 'pix',
                                                color='Green', n_figure=i + 2)
            plt.figure(1)
            plt.annotate('fig_' + str(i + 2), xy=(150, 150), xytext=(regiones[i][0], regiones[i][1]), color='red',
                         size=12)
            print 'Region: X=' + str(regiones[i][0]) + ' Y=' + str(regiones[i][1]) + ' R= ' + str(
                regiones[i][2]) + ' en Figure ' + str(i + 2)

    def create_movie_redshift_range(self, z_ini=0., z_fin=1., dz=0.001, outvid='emission_lines_video.avi', erase=True):
        OII = 3728.483
        wave = self.create_wavelength_array()
        n = len(wave)
        w_max = wave[n - 1]
        max_z_allowed = (w_max / OII) - 1.
        if z_fin > max_z_allowed:
            print 'maximum redshift allowed is ' + str(max_z_allowed) + ', this value will be used  instead of ' + str(
                z_fin)
            z_fin = max_z_allowed
        z_array = np.arange(z_ini, z_fin, dz)
        images_names = []
        fitsnames = []
        for z in z_array:
            ranges = self.create_ranges(z)
            filename = 'emission_linea_image_redshif_' + str(z) + '_'
            self.colapse_cube(ranges, fitsname=filename + '.fits', n_figure=15)
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



    def colapse_emission_lines_image(self,nsigma=2,fitsname='colapsed_emission_image.fits'):
        data=self.data
        image=data[0]
        n1=len(image)
        n2=len(image[0])
        colapsed_image=np.zeros_like(image)
        for i in xrange(n1):
            for j in xrange(n2):
                if j==n2-1:
                    print 'iteracion '+str(i)+' de '+str(n1-1)
                spec_data=self.get_spectrum_point_aplpy(j,i,coord_system='pix')
                flux_data=np.array(spec_data[1])
                spec_stat=self.get_spectrum_point_aplpy(j,i,coord_system='pix',stat=True)
                flux_stat=np.array(spec_stat[1])
                s2n=flux_data/flux_stat
                condition=s2n>=nsigma
                colapsed_emission=np.nansum(flux_data[condition])
                colapsed_image[i][j]=colapsed_emission
        self.__save2fitsimage(fitsname=fitsname,data_to_save=colapsed_image,type='white')
        return colapsed_image






    def create_ranges(self, z, width=5.):
        wave = self.create_wavelength_array()
        n = len(wave)
        w_max = wave[n - 1]
        w_min = wave[0]
        half = width / 2.
        OII = 3728.483
        Hb = 4862.683
        Ha = 6564.613
        OIII_4959 = 4960.295
        OIII_5007 = 5008.239
        range_OII = np.array([OII - half, OII + half])
        range_Hb = np.array([Hb - half, Hb + half])
        range_Ha = np.array([Ha - half, Ha + half])
        range_OIII_4959 = np.array([OIII_4959 - half, OIII_4959 + half])
        range_OIII_5007 = np.array([OIII_5007 - half, OIII_5007 + half])
        ranges = [range_Ha, range_Hb, range_OII, range_OIII_4959, range_OIII_5007]
        z_ranges = []
        for range in ranges:
            z_ranges.append(range * (1 + z))
        output_ranges = []
        for range in z_ranges:
            if range[0] >= w_min and range[1] <= w_max:
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
