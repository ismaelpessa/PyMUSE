import math as m

import aplpy
import numpy
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt


class MuseCube:
    """
    Class made to manage datacubes from MUSE

    """

    def __init__(self, filename_cube, filename_white):
        """
        :param filename_cube: string
                              name of the fits file containing the datacube
        :param filename_white: string
                               name of the fits file containing the colapsed datacube
        """
        plt.close(1)
        self.cube = filename_cube
        self.white = filename_white
        self.gc2 = aplpy.FITSFigure(self.white, figure=plt.figure(1))
        self.gc2.show_grayscale()
        self.gc = aplpy.FITSFigure(self.cube, slices=[1], figure=plt.figure(2))
        plt.close(2)

    def create_wavelength_array(self):
        """
        Creates the wavelength array for the spectrum. The values of dw, and limits will depend
        of the data and should be revised.
        :return: w: array[]
                 array which contain an evenly sampled wavelength range
        """
        dw = 1.25
        w = np.arange(4750, 9351.25, dw)
        #print 'wavelength in range ' + str(w[0]) + ' to ' + str(w[len(w) - 1]) + ' and dw = ' + str(dw)
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

    def closest_element(self,array,element):
        min_dif=100
        index=-1
        n=len(array)
        for i in xrange(0,n):
            dif=abs(element-w[i])
            if dif<min_dif:
                index=i
                min_dif=dif
        return index
    def calculate_error(self,exposures_names,wavelength):

        wave=self.create_wavelength_array()
        n=len(wave)
        if wavelength<min(wave) or wavelength>max(wave):
            raise ValueError('Longitud de onda dada no esta dentro del rango valido')
        elif wavelength>min(wave) and wavelength<max(wave) and self.indexOf(wave,wavelength)==-1:
            print 'Longitud de onda en rango valido, pero el valor asignado no esta definido'
            k=int(self.closest_element(wave,wavelength))
            print 'Se usara wavelength = '+str(wave[k])
        elif wavelength>min(wave) and wavelength<max(wave) and self.indexOf(wave,wavelength)>=0:
            k=self.indexOf(wave,wavelength)

        data_matrix_array=[]
        for exposure in exposures_names:
            hdulist=fits.open(exposure)
            DATA=hdulist[1].data
            print DATA.shape
            Nw = len(DATA)
            Ny = len(DATA[0])
            Nx = len(DATA[0][0])
            matrix = np.array([[0. for x in range(Nx)] for y in range(Ny)])
            for i in xrange(0, Ny):
                for j in xrange(0, Nx):
                    matrix[i][j] = DATA[k][i][j]
            data_matrix_array.append(matrix)
        matrix_errors=np.array([[0. for x in range(Nx)] for y in range(Ny)])
        for i in xrange(0, Ny):
            for j in xrange(0, Nx):
                matrix_elements=[]
                for matrix in data_matrix_array:
                    matrix_elements.append(matrix[i][j])
                error=np.std(matrix_elements)
                matrix_errors[i][j]=error
        hdulist = fits.HDUList.fromfile(self.white)
        hdulist[1].data = matrix_errors
        fitsname='errors'
        n_figure=2
        hdulist.writeto(fitsname, clobber=True)
        errors = aplpy.FITSFigure(fitsname, figure=plt.figure(n_figure))
        errors.show_grayscale()






    def colapse_cube(self, wavelength, fitsname='new_colapsed_cube.fits', n_figure=2):
        """
        Function that creates a new image, by colapsing some wavelengths of the cube

        :param wavelength: array[]
                           wavelength contain the information of the wavelength that will be colapsed
                           This array can contain eithar individual values, which represents the wavelengths
                           that will be colapsed, or tuples, which represents wavelengths ranges that will be
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
            print 'Tipo de variable usada para definir el longitudes de onda a colapsar desconocido'
            wave = self.create_wavelength_array()
        wave = list(wave)
        wave_index = []
        n = len(wavelength)
        dw = wave[1] - wave[0]
        if interval == 0:
            for i in xrange(0, n):
                index = self.indexOf(wave, wavelength[i])
                if index < 0:
                    print str(wavelength[i]) + ' is not a valid wavelength value'
                if index >= 0:
                    wave_index.append(index)
        if interval == 1:
            for i in xrange(0, n):
                w_up = wavelength[i][1]
                w_low = wavelength[i][0]
                w_aux = w_low
                if self.indexOf(wave, w_low) < 0:
                    print str(w_low) + ' no es un valor valido como limite inferior, intervalo descartado'
                if w_up >= wave[len(wave) - 1]:
                    print str(w_up) + ' no es un valor valido como limite superior, intervalo descartado'
                if self.indexOf(wave, w_low) >= 0 and w_up < wave[len(wave) - 1]:
                    while w_aux < w_up:
                        index = self.indexOf(wave, w_aux)
                        wave_index.append(index)
                        w_aux += dw

        hdulist = fits.open(self.cube)
        DATA = hdulist[1].data
        Nw = len(DATA)
        Ny = len(DATA[0])
        Nx = len(DATA[0][0])
        Matrix = np.array([[0. for x in range(Nx)] for y in range(Ny)])
        image_stacker = Matrix
        for k in wave_index:
            for i in xrange(0, Ny):
                for j in xrange(0, Nx):
                    Matrix[i][j] = DATA[k][i][j]
        image_stacker = image_stacker + Matrix
        hdulist = fits.HDUList.fromfile(self.white)
        hdulist[1].data = image_stacker
        hdulist.writeto(fitsname, clobber=True)
        colapsed = aplpy.FITSFigure(fitsname, figure=plt.figure(n_figure))
        colapsed.show_grayscale()
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
                                              coord_system, n_figure=2):
        """
        Function to obtain and display the spectrum of a source in circular region of R = radius,
        substracting the spectrum of the sky, obtained in a ring region around x_center and y_center,
        with internal radius = sky_radius_1 and external radius = sky_radius_2


        :param x_center: float
                         x coordinate of the center of the circular region
        :param y_center: float
                         y coordinate of the center of the circular region
        :param radius: float
                       radius of the circular region
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
        w_sky, spec_sky = self.spectrum_ring_region(x_center, y_center, sky_radius_1, sky_radius_2, coord_system)
        self.draw_circle(x_center, y_center, sky_radius_1, 'Blue', coord_system)
        self.draw_circle(x_center, y_center, sky_radius_2, 'Blue', coord_system)
        self.draw_circle(x_center, y_center, radius, 'Green', coord_system)
        reg = self.define_region(x_center, y_center, radius, coord_system)
        ring = self.define_ring_region(x_center, y_center, sky_radius_1, sky_radius_2, coord_system)
        #print ring
        #print reg
        normalization_factor = float(len(reg)) / len(ring)
        print normalization_factor
        spec_sky_normalized = self.normalize_sky(spec_sky, normalization_factor)
        substracted_sky_spec = self.substract_spec(spec, spec_sky_normalized)
        plt.figure(n_figure)
        plt.plot(w, substracted_sky_spec)
        plt.plot(w,spec_sky_normalized)
        return w, substracted_sky_spec



        # plt.plot(w,w_sky)#print spec_sky

    def clean_canvas(self):
        """
        Clean everything from the canvas with the colapsed cube image
        :param self:
        :return:
        """
        plt.close(1)
        self.gc2 = aplpy.FITSFigure(self.white, figure=plt.figure(1))
        self.gc2.show_grayscale()

    def write_coords(self, filename, Nx=315, Ny=309):
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
        f = open(filename, 'w')
        for i in xrange(0, Nx):
            for j in xrange(0, Ny):
                x_world, y_world = self.gc.pixel2world(np.array([i]), np.array([j]))
                c = SkyCoord(ra=x_world, dec=y_world, frame='icrs', unit='deg')
                f.write(str(i) + '\t' + str(j) + '\t' + str(x_world[0]) + '\t' + str(y_world[0]) + '\t' + str(
                    c.to_string('hmsdms')) + '\n')
        f.close()

    def is_in_ring(self, x_center, y_center, radius_1, radius_2, x, y):
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
                             possible values: 'wcs', 'pix', indicates de coordinate system used.
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
        DATA = hdulist[1].data
        DATA.shape
        print 'X,Y,Lambda'

    def get_spectrum_point_aplpy(self, x, y, coord_system):
        """
        Obtain the spectrum of a given point defined by (x,y) in the datacube
        :param self:
        :param x: float
                  x coordinate of the point
        :param y: float
                  y coordinate of the point
        :param coord_system: string
                             possible values: 'wcs', 'pix', indicates the coordinate system used.
        :return: wave: array[]
                       array with the wavelegth of the spectrum.
                 spec: array[]
                       array with the flux of the spectrum.
        """
        # gc = aplpy.FITSFigure(input,slices=[1])
        # gc.show_grayscale()
        input = self.cube
        if coord_system == 'wcs':
            x_pix, y_pix = self.w2p(x, y)
        if coord_system == 'pix':
            x_pix = x
            y_pix = y

        # print x_pix,y_pix
        hdulist = fits.open(input)
        # hdulist.info()
        DATA = hdulist[1].data
        # DATA.shape ##Z,X,Y
        Nw = len(DATA)
        Ny = len(DATA[0])
        Nx = len(DATA[0][0])
        wave = self.create_wavelength_array()
        spec = []
        for i in xrange(0, len(wave)):
            spec.append(DATA[i][y_pix][x_pix])
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

    def spectrum_region(self, x_center, y_center, radius, coord_system, debug=False):
        """
        Obtain the spectrum of a given region in the datacube, defined by a center (x_center,y_center), and
        radius.
        :param self:
        :param x_center: float
                         x coordinate of the center of the circular region
        :param y_center: float
                         y coordinate of the center of the circular region
        :param radius: float
                       radius of the circular region
        :param coord_system: string
                             possible values: 'wcs, 'pix', indicates the coordinate system used.
        :return: wave: array[]
                       array with the wavelength of the spectrum
                 combined_spec: array[]
                                array with the flux of the spectrum. This flux is the sum of the fluxes of all
                                pixels in the region.

        """
        input = self.cube
        Region = self.define_region(x_center, y_center, radius, coord_system)
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



            # Un radio de 4 pixeles es equivalente a un radio de 0.0002 en wcs
