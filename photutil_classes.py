### WARNING: FOR NEAR-TERM WORK, WE'VE DIRECTLY COPIED
### PHOTUTILS CLASSES HERE TO BE OVERWRITTEN

import photutils
import warnings

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import overlap_slices, NoOverlapError
from astropy.stats import SigmaClip, gaussian_sigma_to_fwhm
from astropy.table import Column, QTable, hstack, vstack
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from photutils.psf.groupstars import DAOGroup
from photutils.psf.utils import (_extract_psf_fitting_names, get_grouped_psf_model,
                    subtract_psf)
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import MMMBackground
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
from photutils.utils._misc import _get_version_info

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the DAOStarFinder class.
"""
import inspect
import warnings

import astropy
from astropy.nddata import extract_array
from astropy.table import QTable
from astropy.utils import lazyproperty
import numpy as np

from photutils.detection.base import StarFinderBase
from photutils.detection._utils import _StarFinderKernel, _find_stars
from photutils.utils._convolution import _filter_data
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import NDData

class dao_GriddedPSFModel(photutils.psf.models.Fittable2DModel):
    """
    A fittable 2D model containing a grid PSF models defined at specific
    locations that are interpolated to evaluate a PSF at an arbitrary
    (x, y) position.

    Parameters
    ----------
    data : `~astropy.nddata.NDData`
        An `~astropy.nddata.NDData` object containing the grid of
        reference PSF arrays.  The data attribute must contain a 3D
        `~numpy.ndarray` containing a stack of the 2D PSFs (the data
        shape should be (N_psf, PSF_ny, PSF_nx)).  The meta
        attribute must be `dict` containing the following:

            * ``'grid_xypos'``:  A list of the (x, y) grid positions of
              each reference PSF.  The order of positions should match
              the first axis of the 3D `~numpy.ndarray` of PSFs.  In
              other words, ``grid_xypos[i]`` should be the (x, y)
              position of the reference PSF defined in ``data[i]``.
            * ``'oversampling'``:  The integer oversampling factor of the
               PSF.

        The meta attribute may contain other properties such as the
        telescope, instrument, detector, and filter of the PSF.
    """

    flux = Parameter(description='Intensity scaling factor for the PSF '
                     'model.', default=1.0)
    x_0 = Parameter(description='x position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)
    y_0 = Parameter(description='y position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)

    def __init__(self, data, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, fill_value=0.0):

        if not isinstance(data, NDData):
            raise TypeError('data must be an NDData instance.')

        if data.data.ndim != 3:
            raise ValueError('The NDData data attribute must be a 3D numpy '
                             'ndarray')

        if 'grid_xypos' not in data.meta:
            raise ValueError('"grid_xypos" must be in the nddata meta '
                             'dictionary.')
        if len(data.meta['grid_xypos']) != data.data.shape[0]:
            raise ValueError('The length of grid_xypos must match the number '
                             'of input PSFs.')

        if 'oversampling' not in data.meta:
            raise ValueError('"oversampling" must be in the nddata meta '
                             'dictionary.')
        if not np.isscalar(data.meta['oversampling']):
            raise ValueError('oversampling must be a scalar value')

        self.data = np.array(data.data, copy=True, dtype=float)
        self.meta = data.meta
        self.grid_xypos = data.meta['grid_xypos']
        self.oversampling = data.meta['oversampling']

        self._grid_xpos, self._grid_ypos = np.transpose(self.grid_xypos)
        self._xgrid = np.unique(self._grid_xpos)  # also sorts values
        self._ygrid = np.unique(self._grid_ypos)  # also sorts values

    
        self._xgrid_min = self._xgrid[0]
        self._xgrid_max = self._xgrid[-1]
        self._ygrid_min = self._ygrid[0]
        self._ygrid_max = self._ygrid[-1]

        super().__init__(flux, x_0, y_0)

    @staticmethod
    def _find_bounds_1d(data, x):
        """
        Find the index of the lower bound where ``x`` should be inserted
        into ``a`` to maintain order.

        The index of the upper bound is the index of the lower bound
        plus 2.  Both bound indices must be within the array.

        Parameters
        ----------
        data : 1D `~numpy.ndarray`
            The 1D array to search.

        x : float
            The value to insert.

        Returns
        -------
        index : int
            The index of the lower bound.
        """
        idx = np.searchsorted(data, x)
        if idx == 0:
            idx0 = 0
        elif idx == len(data):  # pragma: no cover
            idx0 = idx - 2
        else:
            idx0 = idx - 1

        return idx0

    def _find_bounding_points(self, x, y):
        """
        Find the indices of the grid points that bound the input
        ``(x, y)`` position.

        Parameters
        ----------
        x, y : float
            The ``(x, y)`` position where the PSF is to be evaluated.

        Returns
        -------
        indices : list of int
            A list of indices of the bounding grid points.
        """
        if not np.isscalar(x) or not np.isscalar(y):  # pragma: no cover
            raise TypeError('x and y must be scalars')

        if (x < self._xgrid_min or x > self._xgrid_max or
                y < self._ygrid_min or y > self._ygrid_max):  # pragma: no cover
            raise ValueError('(x, y) position is outside of the region '
                             'defined by grid of PSF positions')

        x0 = self._find_bounds_1d(self._xgrid, x)
        y0 = self._find_bounds_1d(self._ygrid, y)
        points = list(itertools.product(self._xgrid[x0:x0 + 2],
                                        self._ygrid[y0:y0 + 2]))

        indices = []
        for xx, yy in points:
            indices.append(np.argsort(np.hypot(self._grid_xpos - xx,
                                               self._grid_ypos - yy))[0])

        return indices

    @staticmethod
    def _bilinear_interp(xyref, data, xi, yi):
        """
        Perform bilinear interpolation of four 2D arrays located at
        points on a regular grid.

        Parameters
        ----------
        xyref : list of 4 (x, y) pairs
            A list of 4 ``(x, y)`` pairs that form a rectangle.

        zref : 3D `~numpy.ndarray`
            A 3D `~numpy.ndarray` of shape ``(4, nx, ny)``. The first
            axis corresponds to ``xyref``, i.e., ``refdata[0, :, :]`` is
            the 2D array located at ``xyref[0]``.

        xi, yi : float
            The ``(xi, yi)`` point at which to perform the
            interpolation.  The ``(xi, yi)`` point must lie within the
            rectangle defined by ``xyref``.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The 2D interpolated array.
        """


        xyref = np.array([np.array(i) for i in xyref])

        distances = np.array([np.linalg.norm(np.array(xi,yi)-xy) for xy in xyref])
        keep = np.argsort(distances)[:4]
        weights = 1./distances[keep]**2
        norm = np.sum(weights)

        return np.sum(data[keep,:,:] * weights[:, None, None], axis=0) / norm

    def _compute_local_model(self, x_0, y_0):
        """
        Return `FittableImageModel` for interpolated PSF at some (x_0, y_0).
        """
        # NOTE: this is needed because the PSF photometry routines input
        # length-1 values instead of scalars.  TODO: fix the photometry
        # routines.
        if not np.isscalar(x_0):
            x_0 = x_0[0]
        if not np.isscalar(y_0):
            y_0 = y_0[0]

        
        # find the four bounding reference PSFs and interpolate
        xyref = np.array(self.grid_xypos)
        

        self._psf_interp = self._bilinear_interp(xyref, self.data, x_0, y_0)

        # Construct the model using the interpolated supersampled data
        psfmodel = photutils.psf.models.FittableImageModel(self._psf_interp,
                                      oversampling=self.oversampling)
        return psfmodel

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Evaluate the `GriddedPSFModel` for the input parameters.
        """
        # Get the local PSF at the (x_0,y_0)
        psfmodel = self._compute_local_model(x_0, y_0)

        # now evaluate the PSF at the (x_0, y_0) subpixel position on
        # the input (x, y) values
        return psfmodel.evaluate(x, y, flux, x_0, y_0)


class dao_DAOStarFinder(StarFinderBase):
    """
    Detect stars in an image using the DAOFIND (`Stetson 1987
    <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_)
    algorithm.

    DAOFIND (`Stetson 1987; PASP 99, 191
    <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_)
    searches images for local density maxima that have a peak amplitude
    greater than ``threshold`` (approximately; ``threshold`` is applied
    to a convolved image) and have a size and shape similar to the
    defined 2D Gaussian kernel.  The Gaussian kernel is defined by the
    ``fwhm``, ``ratio``, ``theta``, and ``sigma_radius`` input
    parameters.

    ``DAOStarFinder`` finds the object centroid by fitting the marginal x
    and y 1D distributions of the Gaussian kernel to the marginal x and
    y distributions of the input (unconvolved) ``data`` image.

    ``DAOStarFinder`` calculates the object roundness using two methods. The
    ``roundlo`` and ``roundhi`` bounds are applied to both measures of
    roundness.  The first method (``roundness1``; called ``SROUND`` in
    `DAOFIND`_) is based on the source symmetry and is the ratio of a
    measure of the object's bilateral (2-fold) to four-fold symmetry.
    The second roundness statistic (``roundness2``; called ``GROUND`` in
    `DAOFIND`_) measures the ratio of the difference in the height of
    the best fitting Gaussian function in x minus the best fitting
    Gaussian function in y, divided by the average of the best fitting
    Gaussian functions in x and y.  A circular source will have a zero
    roundness.  A source extended in x or y will have a negative or
    positive roundness, respectively.

    The sharpness statistic measures the ratio of the difference between
    the height of the central pixel and the mean of the surrounding
    non-bad pixels in the convolved image, to the height of the best
    fitting Gaussian function at that point.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.

    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.

    ratio : float, optional
        The ratio of the minor to major axis standard deviations of the
        Gaussian kernel.  ``ratio`` must be strictly positive and less
        than or equal to 1.0.  The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counter-clockwise from the positive x
        axis.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        (2.0*sqrt(2.0*log(2.0)))``].

    sharplo : float, optional
        The lower bound on sharpness for object detection.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

    roundlo : float, optional
        The lower bound on roundness for object detection.

    roundhi : float, optional
        The upper bound on roundness for object detection.

    sky : float, optional
        The background sky level of the image.  Setting ``sky`` affects
        only the output values of the object ``peak``, ``flux``, and
        ``mag`` values.  The default is 0.0, which should be used to
        replicate the results from `DAOFIND`_.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `DAOFIND`_.

    brightest : int, None, optional
        Number of brightest objects to keep after sorting the full object list.
        If ``brightest`` is set to `None`, all objects will be selected.

    peakmax : float, None, optional
        Maximum peak pixel value in an object. Only objects whose peak pixel
        values are *strictly smaller* than ``peakmax`` will be selected.
        This may be used to exclude saturated sources. By default, when
        ``peakmax`` is set to `None`, all objects will be selected.

        .. warning::
            `DAOStarFinder` automatically excludes objects whose peak
            pixel values are negative. Therefore, setting ``peakmax`` to a
            non-positive value would result in exclusion of all objects.

    xycoords : `None` or Nx2 `~numpy.ndarray`
        The (x, y) pixel coordinates of the approximate centroid
        positions of identified sources. If ``xycoords`` are input, the
        algorithm will skip the source-finding step.

    See Also
    --------
    IRAFStarFinder

    Notes
    -----
    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.  The equivalent parameters in `DAOFIND`_ are
    ``boundary='constant'`` and ``constant=0.0``.

    The main differences between `~photutils.detection.DAOStarFinder`
    and `~photutils.detection.IRAFStarFinder` are:

    * `~photutils.detection.IRAFStarFinder` always uses a 2D
      circular Gaussian kernel, while
      `~photutils.detection.DAOStarFinder` can use an elliptical
      Gaussian kernel.

    * `~photutils.detection.IRAFStarFinder` calculates the objects'
      centroid, roundness, and sharpness using image moments.

    References
    ----------
    .. [1] Stetson, P. 1987; PASP 99, 191
           (https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract)
    .. [2] https://iraf.net/irafhelp.php?val=daofind

    .. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind
    """

    def __init__(self, threshold, fwhm, ratio=1.0, theta=0.0,
                 sigma_radius=1.5, sharplo=0.2, sharphi=1.0, roundlo=-1.0,
                 roundhi=1.0, sky=0.0, exclude_border=False,
                 brightest=None, peakmax=None, xycoords=None):

        if not np.isscalar(threshold):
            raise TypeError('threshold must be a scalar value.')

        if not np.isscalar(fwhm):
            raise TypeError('fwhm must be a scalar value.')

        self.threshold = threshold
        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta
        self.sigma_radius = sigma_radius
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.sky = sky
        self.exclude_border = exclude_border
        self.brightest = self._validate_brightest(brightest)
        self.peakmax = peakmax

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                raise ValueError('xycoords must be shaped as a Nx2 array')
        self.xycoords = xycoords

        self.kernel = _StarFinderKernel(self.fwhm, self.ratio, self.theta,
                                        self.sigma_radius)
        self.threshold_eff = self.threshold * self.kernel.relerr

    @staticmethod
    def _validate_brightest(brightest):
        if brightest is not None:
            if brightest <= 0:
                raise ValueError('brightest must be >= 0')
            bright_int = int(brightest)
            if bright_int != brightest:
                raise ValueError('brightest must be an integer')
            brightest = bright_int
        return brightest

    def find_stars(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked.  Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.QTable` or `None`
            A table of found stars with the following parameters:

            * ``id``: unique object identification number.
            * ``xcentroid, ycentroid``: object centroid.
            * ``sharpness``: object sharpness.
            * ``roundness1``: object roundness based on symmetry.
            * ``roundness2``: object roundness based on marginal Gaussian
              fits.
            * ``npix``: the total number of pixels in the Gaussian kernel
              array.
            * ``sky``: the input ``sky`` parameter.
            * ``peak``: the peak, sky-subtracted, pixel value of the object.
            * ``flux``: the object flux calculated as the peak density in
              the convolved image divided by the detection threshold.  This
              derivation matches that of `DAOFIND`_ if ``sky`` is 0.0.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.  The derivation matches that of
              `DAOFIND`_ if ``sky`` is 0.0.

            `None` is returned if no stars are found.
        """
        convolved_data = _filter_data(data, self.kernel.data, mode='constant',
                                      fill_value=0.0,
                                      check_normalization=False)

        if self.xycoords is None:
            xypos = _find_stars(convolved_data, self.kernel,
                                self.threshold_eff, mask=mask,
                                exclude_border=self.exclude_border)
        else:
            xypos = self.xycoords

        if xypos is None:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        cat = _DAOStarFinderCatalog(data, convolved_data, xypos, self.kernel,
                                    self.threshold, sky=self.sky,
                                    sharplo=self.sharplo, sharphi=self.sharphi,
                                    roundlo=self.roundlo, roundhi=self.roundhi,
                                    brightest=self.brightest,
                                    peakmax=self.peakmax)

        # filter the catalog
        #cat = cat.apply_filters()
        if cat is None:
            return None
        cat = cat.select_brightest()
        cat.reset_ids()

        # create the output table
        return cat.to_table()


class _DAOStarFinderCatalog:
    """
    Class to create a catalog of the properties of each detected star,
    as defined by `DAOFIND`_.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image.

    convolved_data : 2D `~numpy.ndarray`
        The convolved 2D image.

    xypos: Nx2 `numpy.ndarray`
        A Nx2 array of (x, y) pixel coordinates denoting the central
        positions of the stars.

    kernel : `_StarFinderKernel`
        The convolution kernel. This kernel must match the kernel used
        to create the ``convolved_data``.

    threshold : float
        The absolute image value above which sources were selected.

    sky : float, optional
        The local sky level around the source.  ``sky`` is used only to
        calculate the source peak value, flux, and magnitude.  The
        default is 0.

    .. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind
    """

    def __init__(self, data, convolved_data, xypos, kernel, threshold,
                 sky=0., sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                 brightest=None, peakmax=None):

        self.data = data
        self.convolved_data = convolved_data
        self.xypos = np.atleast_2d(xypos)
        self.kernel = kernel
        self.threshold = threshold
        self._sky = sky  # DAOFIND has no sky input -> same as sky=0.
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.brightest = brightest
        self.peakmax = peakmax

        self.id = np.arange(len(self)) + 1
        self.threshold_eff = threshold * kernel.relerr
        self.cutout_shape = kernel.shape
        self.cutout_center = tuple([(size - 1) // 2 for size in kernel.shape])
        self.default_columns = ('id', 'xcentroid', 'ycentroid', 'sharpness',
                                'roundness1', 'roundness2', 'npix', 'sky',
                                'peak', 'flux', 'mag')

    def __len__(self):
        return len(self.xypos)

    def __getitem__(self, index):
        newcls = object.__new__(self.__class__)
        init_attr = ('data', 'convolved_data', 'kernel', 'threshold', '_sky',
                     'sharplo', 'sharphi', 'roundlo', 'roundhi', 'brightest',
                     'peakmax', 'threshold_eff', 'cutout_shape',
                     'cutout_center', 'default_columns')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # xypos determines ordering and isscalar
        # NOTE: always keep as a 2D array, even for a single source
        attr = 'xypos'
        value = getattr(self, attr)[index]
        setattr(newcls, attr, np.atleast_2d(value))

        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        keys.add('id')
        for key in keys:
            value = self.__dict__[key]

            # do not insert lazy attributes that are always scalar (e.g.,
            # isscalar), i.e., not an array/list for each source
            if np.isscalar(value):
                continue

            # value is always at least a 1D array, even for a single source
            value = np.atleast_1d(value[index])

            newcls.__dict__[key] = value
        return newcls

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single source).
        """
        return self.xypos.shape == (1, 2)

    @property
    def _lazyproperties(self):
        """
        Return all lazyproperties (even in superclasses).
        """
        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)
        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    def reset_ids(self):
        """Reset the ID column to be consecutive integers."""
        self.id = np.arange(len(self)) + 1

    def make_cutouts(self, data):
        cutouts = []
        for xpos, ypos in self.xypos:
            cutouts.append(extract_array(data, self.cutout_shape, (ypos, xpos),
                                         fill_value=0.0))
        return np.array(cutouts)

    @lazyproperty
    def cutout_data(self):
        return self.make_cutouts(self.data)

    @lazyproperty
    def cutout_convdata(self):
        return self.make_cutouts(self.convolved_data)

    @lazyproperty
    def data_peak(self):
        return self.cutout_data[:, self.cutout_center[0],
                                self.cutout_center[1]]

    @lazyproperty
    def convdata_peak(self):
        return self.cutout_convdata[:, self.cutout_center[0],
                                    self.cutout_center[1]]

    @lazyproperty
    def roundness1(self):
        # set the central (peak) pixel to zero for the sum4 calculation
        cutout_conv = self.cutout_convdata.copy()
        cutout_conv[:, self.cutout_center[0], self.cutout_center[1]] = 0.0

        # calculate the four roundness quadrants.
        # the cutout size always matches the kernel size, which has odd
        # dimensions.
        # quad1 = bottom right
        # quad2 = bottom left
        # quad3 = top left
        # quad4 = top right
        # 3 3 4 4 4
        # 3 3 4 4 4
        # 3 3 x 1 1
        # 2 2 2 1 1
        # 2 2 2 1 1
        quad1 = cutout_conv[:, 0:self.cutout_center[0] + 1,
                            self.cutout_center[1] + 1:]
        quad2 = cutout_conv[:, 0:self.cutout_center[0],
                            0:self.cutout_center[1] + 1]
        quad3 = cutout_conv[:, self.cutout_center[0]:,
                            0:self.cutout_center[1]]
        quad4 = cutout_conv[:, self.cutout_center[0] + 1:,
                            self.cutout_center[1]:]

        axis = (1, 2)
        sum2 = (-quad1.sum(axis=axis) + quad2.sum(axis=axis)
                - quad3.sum(axis=axis) + quad4.sum(axis=axis))
        sum4 = np.abs(cutout_conv).sum(axis=axis)

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            roundness1 = 2.0 * sum2 / sum4

        return roundness1

    @lazyproperty
    def sharpness(self):
        # mean value of the unconvolved data (excluding the peak)
        cutout_data_masked = self.cutout_data * self.kernel.mask
        data_mean = ((np.sum(cutout_data_masked, axis=(1, 2)) - self.data_peak)
                     / (self.kernel.npixels - 1))

        return (self.data_peak - data_mean) / self.convdata_peak

    def daofind_marginal_fit(self, axis=0):
        """
        Fit 1D Gaussians, defined from the marginal x/y kernel
        distributions, to the marginal x/y distributions of the original
        (unconvolved) image.

        These fits are used calculate the star centroid and roundness2
        ("GROUND") properties.

        Parameters
        ----------
        axis : {0, 1}, optional
            The axis for which the marginal fit is performed:

            * 0: for the x axis
            * 1: for the y axis

        Returns
        -------
        dx : float
            The fractional shift in x or y (depending on ``axis`` value)
            of the image centroid relative to the maximum pixel.

        hx : float
            The height of the best-fitting Gaussian to the marginal x or
            y (depending on ``axis`` value) distribution of the
            unconvolved source data.
        """
        # define triangular weighting functions along each axis, peaked
        # in the middle and equal to one at the edge
        ycen, xcen = self.cutout_center
        xx = xcen - np.abs(np.arange(self.cutout_shape[1]) - xcen) + 1
        yy = ycen - np.abs(np.arange(self.cutout_shape[0]) - ycen) + 1
        xwt, ywt = np.meshgrid(xx, yy)

        if axis == 0:  # marginal distributions along x axis
            wt = xwt[0]  # 1D
            wts = ywt  # 2D
            size = self.cutout_shape[1]
            center = xcen
            sigma = self.kernel.xsigma
            dxx = center - np.arange(size)
        elif axis == 1:  # marginal distributions along y axis
            wt = np.transpose(ywt)[0]  # 1D
            wts = xwt  # 2D
            size = self.cutout_shape[0]
            center = ycen
            sigma = self.kernel.ysigma
            dxx = np.arange(size) - center

        # compute marginal sums for given axis
        wt_sum = np.sum(wt)
        dx = center - np.arange(size)

        # weighted marginal sums
        kern_sum_1d = np.sum(self.kernel.gaussian_kernel_unmasked * wts,
                             axis=axis)
        kern_sum = np.sum(kern_sum_1d * wt)
        kern2_sum = np.sum(kern_sum_1d**2 * wt)

        dkern_dx = kern_sum_1d * dx
        dkern_dx_sum = np.sum(dkern_dx * wt)
        dkern_dx2_sum = np.sum(dkern_dx**2 * wt)
        kern_dkern_dx_sum = np.sum(kern_sum_1d * dkern_dx * wt)

        data_sum_1d = np.sum(self.cutout_data * wts, axis=axis + 1)
        data_sum = np.sum(data_sum_1d * wt, axis=1)
        data_kern_sum = np.sum(data_sum_1d * kern_sum_1d * wt, axis=1)
        data_dkern_dx_sum = np.sum(data_sum_1d * dkern_dx * wt, axis=1)
        data_dx_sum = np.sum(data_sum_1d * dxx * wt, axis=1)

        # perform linear least-squares fit (where data = sky + hx*kernel)
        # to find the amplitude (hx)
        hx_numer = data_kern_sum - (data_sum * kern_sum) / wt_sum
        hx_denom = kern2_sum - (kern_sum**2 / wt_sum)

        # reject the star if the fit amplitude is not positive
        mask1 = (hx_numer <= 0.) | (hx_denom <= 0.)

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # compute fit amplitude
            hx = hx_numer / hx_denom

            # sky = (data_sum - (hx * kern_sum)) / wt_sum

            # compute centroid shift
            dx = ((kern_dkern_dx_sum
                   - (data_dkern_dx_sum - dkern_dx_sum * data_sum))
                  / (hx * dkern_dx2_sum / sigma**2))

            dx2 = data_dx_sum / data_sum

        hsize = size / 2.0
        mask2 = (np.abs(dx) > hsize)
        mask3 = (data_sum == 0.)
        mask4 = (mask2 & mask3)
        mask5 = (mask2 & ~mask3)

        dx[mask4] = 0.0
        dx[mask5] = dx2[mask5]
        mask6 = (np.abs(dx) > hsize)
        dx[mask6] = 0.0

        hx[mask1] = np.nan
        dx[mask1] = np.nan

        return np.transpose((dx, hx))

    @lazyproperty
    def dx_hx(self):
        return self.daofind_marginal_fit(axis=0)

    @lazyproperty
    def dy_hy(self):
        return self.daofind_marginal_fit(axis=1)

    @lazyproperty
    def dx(self):
        return np.transpose(self.dx_hx)[0]

    @lazyproperty
    def dy(self):
        return np.transpose(self.dy_hy)[0]

    @lazyproperty
    def hx(self):
        return np.transpose(self.dx_hx)[1]

    @lazyproperty
    def hy(self):
        return np.transpose(self.dy_hy)[1]

    @lazyproperty
    def xcentroid(self):
        return np.transpose(self.xypos)[0] + self.dx

    @lazyproperty
    def ycentroid(self):
        return np.transpose(self.xypos)[1] + self.dy

    @lazyproperty
    def roundness2(self):
        """
        The star roundness.

        This roundness parameter represents the ratio of the difference
        in the height of the best fitting Gaussian function in x minus
        the best fitting Gaussian function in y, divided by the average
        of the best fitting Gaussian functions in x and y.  A circular
        source will have a zero roundness.  A source extended in x or y
        will have a negative or positive roundness, respectively.
        """
        return 2.0 * (self.hx - self.hy) / (self.hx + self.hy)

    @lazyproperty
    def peak(self):
        return self.data_peak - self.sky

    @lazyproperty
    def flux(self):
        return ((self.convdata_peak / self.threshold_eff)
                - (self.sky * self.npix))

    @lazyproperty
    def mag(self):
        # ignore RunTimeWarning if flux is <= 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mag = -2.5 * np.log10(self.flux)
            mag[self.flux <= 0] = np.nan
        return mag

    @lazyproperty
    def sky(self):
        return np.full(len(self), fill_value=self._sky)

    @lazyproperty
    def npix(self):
        return np.full(len(self), fill_value=self.kernel.data.size)

    def apply_filters(self):
        """Filter the catalog."""
        print(len(self))
        mask = (~np.isnan(self.dx) & ~np.isnan(self.dy)
                & ~np.isnan(self.hx) & ~np.isnan(self.hy))
        print(len(self[mask]))
        mask &= ((self.sharpness > self.sharplo)
                 & (self.sharpness < self.sharphi)
                 & (self.roundness1 > self.roundlo)
                 & (self.roundness1 < self.roundhi)
                 & (self.roundness2 > self.roundlo)
                 & (self.roundness2 < self.roundhi))
        if self.peakmax is not None:
            mask &= (self.peak < self.peakmax)
        newcat = self[mask]
        print(len(newcat))
        if len(newcat) == 0:
            warnings.warn('Sources were found, but none pass the sharpness, '
                          'roundness, or peakmax criteria',
                          NoDetectionsWarning)
            return None

        return newcat

    def select_brightest(self):
        """
        Sort the catalog by the brightest fluxes and select the
        top brightest sources.
        """
        newcat = self
        if self.brightest is not None:
            idx = np.argsort(self.flux)[::-1][:self.brightest]
            newcat = self[idx]
        return newcat

    def to_table(self, columns=None):
        meta = {'version': _get_version_info()}
        table = QTable(meta=meta)
        if columns is None:
            columns = self.default_columns
        for column in columns:
            table[column] = getattr(self, column)
        return table


class dao_BasicPSFPhotometry:
    """
    This class implements a PSF photometry algorithm that can find
    sources in an image, group overlapping sources into a single model,
    fit the model to the sources, and subtracting the models from the
    image. This is roughly equivalent to the DAOPHOT routines FIND,
    GROUP, NSTAR, and SUBTRACT.  This implementation allows a flexible
    and customizable interface to perform photometry. For instance, one
    is able to use different implementations for grouping and finding
    sources by using ``group_maker`` and ``finder`` respectivelly. In
    addition, sky background estimation is performed by
    ``bkg_estimator``.

    Parameters
    ----------
    group_maker : callable or `~photutils.psf.GroupStarsBase`
        ``group_maker`` should be able to decide whether a given
        star overlaps with any other and label them as belonging
        to the same group. ``group_maker`` receives as input an
        `~astropy.table.Table` object with columns named as ``id``,
        ``x_0``, ``y_0``, in which ``x_0`` and ``y_0`` have the same
        meaning of ``xcentroid`` and ``ycentroid``. This callable must
        return an `~astropy.table.Table` with columns ``id``, ``x_0``,
        ``y_0``, and ``group_id``. The column ``group_id`` should
        contain integers starting from ``1`` that indicate which group a
        given source belongs to. See, e.g., `~photutils.psf.DAOGroup`.
    bkg_estimator : callable, instance of any \
            `~photutils.background.BackgroundBase` subclass, or None
        ``bkg_estimator`` should be able to compute either a scalar
        background or a 2D background of a given 2D image. See, e.g.,
        `~photutils.background.MedianBackground`.  If None, no
        background subtraction is performed.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.sandbox.DiscretePRF`,
        `~photutils.psf.IntegratedGaussianPRF`, or any other suitable 2D
        model.  This object needs to identify three parameters (position
        of center in x and y coordinates and the flux) in order to set
        them to suitable starting values for each fit. The names of
        these parameters should be given as ``x_0``, ``y_0`` and
        ``flux``.  `~photutils.psf.prepare_psf_model` can be used to
        prepare any 2D model to match this assumption.
    fitshape : int or length-2 array-like
        Rectangular shape around the center of a star which will be
        used to collect the data to do the fitting. Can be an integer
        to be the same along both axes. For example, 5 is the same as
        (5, 5), which means to fit only at the following relative pixel
        positions: [-2, -1, 0, 1, 2]. Each element of ``fitshape`` must
        be an odd number.
    finder : callable or instance of any \
            `~photutils.detection.StarFinderBase` subclasses or None
        ``finder`` should be able to identify stars, i.e., compute a
        rough estimate of the centroids, in a given 2D image.
        ``finder`` receives as input a 2D image and returns an
        `~astropy.table.Table` object which contains columns with names:
        ``id``, ``xcentroid``, ``ycentroid``, and ``flux``. In which
        ``id`` is an integer-valued column starting from ``1``,
        ``xcentroid`` and ``ycentroid`` are center position estimates of
        the sources and ``flux`` contains flux estimates of the sources.
        See, e.g., `~photutils.detection.DAOStarFinder`.  If ``finder``
        is ``None``, initial guesses for positions of objects must be
        provided.
    fitter : `~astropy.modeling.fitting.Fitter` instance
        Fitter object used to compute the optimized centroid positions
        and/or flux of the identified sources. See
        `~astropy.modeling.fitting` for more details on fitters.
    aperture_radius : `None` or float
        The radius (in units of pixels) used to compute initial
        estimates for the fluxes of sources.  ``aperture_radius`` must
        be set if initial flux guesses are not input to the photometry
        class via the ``init_guesses`` keyword.  For tabular PSF models
        (e.g., an `EPSFModel`), you must input the ``aperture_radius``
        keyword.  For analytical PSF models, alternatively you may
        define a FWHM attribute on your input psf_model.
    extra_output_cols : list of str, optional
        List of additional columns for parameters derived by any of the
        intermediate fitting steps (e.g., ``finder``), such as roundness
        or sharpness.

    Notes
    -----
    Note that an ambiguity arises whenever ``finder`` and
    ``init_guesses`` (keyword argument for ``do_photometry``) are both
    not ``None``. In this case, ``finder`` is ignored and initial
    guesses are taken from ``init_guesses``. In addition, an warning is
    raised to remaind the user about this behavior.

    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g., manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at:
        https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract
    """

    def __init__(self, group_maker, bkg_estimator, psf_model, fitshape,
                 finder=None, fitter=LevMarLSQFitter(), aperture_radius=None,
                 extra_output_cols=None):
        self.group_maker = group_maker
        self.bkg_estimator = bkg_estimator
        self.psf_model = psf_model
        self.fitter = fitter
        self.fitshape = fitshape
        self.finder = finder
        self.aperture_radius = aperture_radius
        self._pars_to_set = None
        self._pars_to_output = None
        self._residual_image = None
        self._extra_output_cols = extra_output_cols

    @property
    def fitshape(self):
        return self._fitshape

    @fitshape.setter
    def fitshape(self, value):
        value = np.asarray(value)

        # assume a lone value should mean both axes
        if value.shape == ():
            value = np.array((value, value))

        if value.size == 2:
            if np.all(value) > 0:
                if np.all(value % 2) == 1:
                    self._fitshape = tuple(value)
                else:
                    raise ValueError('fitshape must be odd integer-valued, '
                                     f'received fitshape={value}')
            else:
                raise ValueError('fitshape must have positive elements, '
                                 f'received fitshape={value}')
        else:
            raise ValueError('fitshape must have two dimensions, '
                             f'received fitshape={value}')

    @property
    def aperture_radius(self):
        return self._aperture_radius

    @aperture_radius.setter
    def aperture_radius(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self._aperture_radius = value
        elif value is None:
            self._aperture_radius = value
        else:
            raise ValueError('aperture_radius must be a positive number')

    def get_residual_image(self):
        """
        Return an image that is the result of the subtraction between
        the original image and the fitted sources.

        Returns
        -------
        residual_image : 2D array-like, `~astropy.io.fits.ImageHDU`, \
            `~astropy.io.fits.HDUList`
        """
        return self._residual_image

    def set_aperture_radius(self):
        """
        Set the fallback aperture radius for initial flux calculations
        in cases where no flux is supplied for a given star.
        """
        if hasattr(self.psf_model, 'fwhm'):
            self.aperture_radius = self.psf_model.fwhm.value
        elif hasattr(self.psf_model, 'sigma'):
            self.aperture_radius = (self.psf_model.sigma.value *
                                    gaussian_sigma_to_fwhm)
        # If PSF model doesn't have FWHM or sigma value -- as it
        # is not a Gaussian; most likely because it's an ePSF --
        # then we fall back on fitting a circle of the average
        # size of the fitting box. As ``fitshape`` is the width
        # of the box, we need (width-1)/2 as the radius.
        else:
            self.aperture_radius = float(np.amin((np.asanyarray(
                                         self.fitshape) - 1) / 2))
            warnings.warn('aperture_radius is None and could not '
                          'be determined by psf_model. Setting '
                          'radius to the smallest fitshape size. '
                          'This aperture radius will be used if '
                          'initial fluxes require computing for any '
                          'input stars. If fitshape is significantly '
                          'larger than the psf_model core lengthscale, '
                          'consider supplying a specific aperture_radius.',
                          AstropyUserWarning)

    def __call__(self, image, init_guesses=None,image_weights=None):
        """
        Perform PSF photometry. See `do_photometry` for more details
        including the `__call__` signature.
        """

        return self.do_photometry(image, init_guesses,image_weights=image_weights)

    def do_photometry(self, image, init_guesses=None,image_weights=None):
        """
        Perform PSF photometry in ``image``.

        This method assumes that ``psf_model`` has centroids and flux
        parameters which will be fitted to the data provided in
        ``image``. A compound model, in fact a sum of ``psf_model``,
        will be fitted to groups of stars automatically identified by
        ``group_maker``. Also, ``image`` is not assumed to be background
        subtracted.  If ``init_guesses`` are not ``None`` then this
        method uses ``init_guesses`` as initial guesses for the
        centroids. If the centroid positions are set as ``fixed`` in the
        PSF model ``psf_model``, then the optimizer will only consider
        the flux as a variable.

        Parameters
        ----------
        image : 2D array-like, `~astropy.io.fits.ImageHDU`, \
                `~astropy.io.fits.HDUList`
            Image to perform photometry.
        init_guesses: `~astropy.table.Table`
            Table which contains the initial guesses (estimates) for the
            set of parameters. Columns 'x_0' and 'y_0' which represent
            the positions (in pixel coordinates) for each object must be
            present.  'flux_0' can also be provided to set initial
            fluxes.  If 'flux_0' is not provided, aperture photometry is
            used to estimate initial values for the fluxes. Additional
            columns of the form '<parametername>_0' will be used to set
            the initial guess for any parameters of the ``psf_model``
            model that are not fixed. If ``init_guesses`` supplied with
            ``extra_output_cols`` the initial values are used; if the columns
            specified in ``extra_output_cols`` are not given in
            ``init_guesses`` then NaNs will be returned.

        Returns
        -------
        output_tab : `~astropy.table.Table` or None
            Table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process. Uncertainties on the fitted parameters
            are reported as columns called ``<paramname>_unc`` provided
            that the fitter object contains a dictionary called
            ``fit_info`` with the key ``param_cov``, which contains the
            covariance matrix. If ``param_cov`` is not present,
            uncertanties are not reported.
        """
        if self.bkg_estimator is not None:
            image = image - self.bkg_estimator(image)

        if self.aperture_radius is None:
            self.set_aperture_radius()

        skip_group_maker = False
        if init_guesses is not None:
            # make sure the code does not modify user's input
            init_guesses = init_guesses.copy()

            if self.finder is not None:
                warnings.warn('Both init_guesses and finder are different '
                              'than None, which is ambiguous. finder is '
                              'going to be ignored.', AstropyUserWarning)

            colnames = init_guesses.colnames
            if 'group_id' in colnames:
                warnings.warn('init_guesses contains a "group_id" column. '
                              'The group_maker step will be skipped.',
                              AstropyUserWarning)
                skip_group_maker = True

            if 'flux_0' not in colnames:
                positions = np.transpose((init_guesses['x_0'],
                                          init_guesses['y_0']))
                apertures = CircularAperture(positions,
                                             r=self.aperture_radius)

                init_guesses['flux_0'] = aperture_photometry(
                    image, apertures)['aperture_sum']
            # if extra_output_cols have been given, check whether init_guesses
            # was supplied with extra_output_cols pre-attached and populate
            # columns not given with NaNs
            if self._extra_output_cols is not None:
                for col_name in self._extra_output_cols:
                    if col_name not in init_guesses.colnames:
                        init_guesses[col_name] = np.full(len(init_guesses),
                                                         np.nan)
        else:
            if self.finder is None:
                raise ValueError('Finder cannot be None if init_guesses are '
                                 'not given.')
            sources = self.finder(image)
            if len(sources) > 0:
                positions = np.transpose((sources['xcentroid'],
                                          sources['ycentroid']))
                apertures = CircularAperture(positions,
                                             r=self.aperture_radius)

                sources['aperture_flux'] = aperture_photometry(
                    image, apertures)['aperture_sum']

                # init_guesses should be the initial 3 required
                # parameters (x, y, flux) and then concatenated with any
                # additional sources, if there are any
                init_guesses = QTable(names=['x_0', 'y_0', 'flux_0'],
                                      data=[sources['xcentroid'],
                                            sources['ycentroid'],
                                            sources['aperture_flux']])

                # Currently only needed for the finder, as group_maker and
                # nstar return the original table with new columns, unlike
                # finder
                self._get_additional_columns(sources, init_guesses)

        self._define_fit_param_names()
        for p0, param in self._pars_to_set.items():
            if p0 not in init_guesses.colnames:
                init_guesses[p0] = (len(init_guesses) *
                                    [getattr(self.psf_model, param).value])

        if skip_group_maker:
            star_groups = init_guesses
        else:
            star_groups = self.group_maker(init_guesses)
        
        output_tab, self._residual_image = self.nstar(image, star_groups,image_weights)
        star_groups = star_groups.group_by('group_id')

        if hasattr(output_tab, 'update'):  # requires Astropy >= 5.0
            star_groups.update(output_tab)
        else:
            common_cols = set(star_groups.colnames).intersection(
                output_tab.colnames)
            for name, col in output_tab.items():
                if name in common_cols:
                    star_groups.replace_column(name, col, copy=True)
                else:
                    star_groups.add_column(col, name=name, copy=True)

        star_groups.meta = {'version': _get_version_info()}

        return star_groups

    def nstar(self, image, star_groups,image_weights=None):
        """
        Fit, as appropriate, a compound or single model to the given
        ``star_groups``. Groups are fitted sequentially from the
        smallest to the biggest. In each iteration, ``image`` is
        subtracted by the previous fitted group.

        Parameters
        ----------
        image : numpy.ndarray
            Background-subtracted image.

        star_groups : `~astropy.table.Table`
            This table must contain the following columns: ``id``,
            ``group_id``, ``x_0``, ``y_0``, ``flux_0``.  ``x_0`` and
            ``y_0`` are initial estimates of the centroids and
            ``flux_0`` is an initial estimate of the flux. Additionally,
            columns named as ``<param_name>_0`` are required if any
            other parameter in the psf model is free (i.e., the
            ``fixed`` attribute of that parameter is ``False``).

        Returns
        -------
        result_tab : `~astropy.table.QTable`
            Astropy table that contains photometry results.

        image : numpy.ndarray
            Residual image.
        """
        
        result_tab = QTable()
        for param_tab_name in self._pars_to_output.keys():
            result_tab.add_column(Column(name=param_tab_name))
        unc_tab = QTable()
        for param, isfixed in self.psf_model.fixed.items():
            if not isfixed:
                unc_tab.add_column(Column(name=param + "_unc"))
        
        y, x = np.indices(image.shape)

        star_groups = star_groups.group_by('group_id')
        for n in range(len(star_groups.groups)):
            group_psf = get_grouped_psf_model(self.psf_model,
                                              star_groups.groups[n],
                                              self._pars_to_set)
            usepixel = np.zeros_like(image, dtype=bool)

            for row in star_groups.groups[n]:
                usepixel[overlap_slices(large_array_shape=image.shape,
                                        small_array_shape=self.fitshape,
                                        position=(row['y_0'], row['x_0']),
                                        mode='trim')[0]] = True
            if image_weights is None:
                w = None
            else:
                w = image_weights[usepixel]
            #import matplotlib.pyplot as plt
            #plt.imshow(image[usepixel].reshape(self.fitshape))
            #plt.show()
            #print(image.shape,np.nansum(image[usepixel]),len(np.where(usepixel)[0]))
            fit_model = self.fitter(group_psf, x[usepixel], y[usepixel],
                                    image[usepixel])
                                    #weights = w,maxiter=100)#,maxiter=200)#,acc=1e-06)
            #print(fit_model)
            param_table = self._model_params2table(fit_model,
                                                   star_groups.groups[n])
            result_tab = vstack([result_tab, param_table])

            param_cov = self.fitter.fit_info.get('param_cov', None)
            if param_cov is not None:
                unc_tab = vstack([unc_tab,
                                  self._get_uncertainties(
                                      len(star_groups.groups[n]))])

            # do not subtract if the fitting did not go well
            try:
                image = subtract_psf(image, self.psf_model, param_table,
                                     subshape=self.fitshape)
            except NoOverlapError:
                pass

        if param_cov is not None:
            result_tab = hstack([result_tab, unc_tab])
        
        return result_tab, image

    def _get_additional_columns(self, in_table, out_table):
        """
        Function to parse additional columns from ``in_table`` and add them to
        ``out_table``.
        """
        if self._extra_output_cols is not None:
            for col_name in self._extra_output_cols:
                if col_name in in_table.colnames:
                    out_table[col_name] = in_table[col_name]

    def _define_fit_param_names(self):
        """
        Convenience function to define mappings between the names of the
        columns in the initial guess table (and the name of the fitted
        parameters) and the actual name of the parameters in the model.

        This method sets the following parameters on the ``self`` object:
        * ``pars_to_set`` : Dict which maps the names of the parameters
          initial guesses to the actual name of the parameter in the
          model.
        * ``pars_to_output`` : Dict which maps the names of the fitted
          parameters to the actual name of the parameter in the model.
        """
        xname, yname, fluxname = _extract_psf_fitting_names(self.psf_model)
        self._pars_to_set = {'x_0': xname, 'y_0': yname, 'flux_0': fluxname}
        self._pars_to_output = {'x_fit': xname, 'y_fit': yname,
                                'flux_fit': fluxname}

        for p, isfixed in self.psf_model.fixed.items():
            p0 = p + '_0'
            pfit = p + '_fit'
            if p not in (xname, yname, fluxname) and not isfixed:
                self._pars_to_set[p0] = p
                self._pars_to_output[pfit] = p

    def _get_uncertainties(self, star_group_size):
        """
        Retrieve uncertainties on fitted parameters from the fitter
        object.

        Parameters
        ----------
        star_group_size : int
            Number of stars in the given group.

        Returns
        -------
        unc_tab : `~astropy.table.QTable`
            A table which contains uncertainties on the fitted parameters.
            The uncertainties are reported as one standard deviation.
        """
        unc_tab = QTable()
        for param_name in self.psf_model.param_names:
            if not self.psf_model.fixed[param_name]:
                unc_tab.add_column(Column(name=param_name + "_unc",
                                          data=np.empty(star_group_size)))

        k = 0
        n_fit_params = len(unc_tab.colnames)
        param_cov = self.fitter.fit_info.get('param_cov', None)
        for i in range(star_group_size):
            unc_tab[i] = np.sqrt(np.diag(param_cov))[k: k + n_fit_params]
            k = k + n_fit_params

        return unc_tab

    def _model_params2table(self, fit_model, star_group):
        """
        Place fitted parameters into an astropy table.

        Parameters
        ----------
        fit_model : `astropy.modeling.Fittable2DModel` instance
            PSF or PRF model to fit the data. Could be one of the models
            in this package like `~photutils.psf.sandbox.DiscretePRF`,
            `~photutils.psf.IntegratedGaussianPRF`, or any other
            suitable 2D model.

        star_group : `~astropy.table.Table`
            the star group instance.

        Returns
        -------
        param_tab : `~astropy.table.QTable`
            A table that contains the fitted parameters.
        """
        param_tab = QTable()

        for param_tab_name in self._pars_to_output.keys():
            param_tab.add_column(Column(name=param_tab_name,
                                        data=np.empty(len(star_group))))

        if len(star_group) > 1:
            for i in range(len(star_group)):
                for param_tab_name, param_name in self._pars_to_output.items():
                    # get sub_model corresponding to star with index i as name
                    # name was set in utils.get_grouped_psf_model()
                    # we can't use model['name'] here as that only
                    # searches leaves and we might want a intermediate
                    # node of the tree
                    sub_models = [model for model
                                  in fit_model.traverse_postorder() if model.name == i]
                    if len(sub_models) != 1:
                        raise ValueError('sub_models must have a length of 1')
                    sub_model = sub_models[0]

                    param_tab[param_tab_name][i] = getattr(sub_model,
                                                           param_name).value
        else:
            for param_tab_name, param_name in self._pars_to_output.items():
                param_tab[param_tab_name] = getattr(fit_model,
                                                    param_name).value

        return param_tab
class dao_IterativelySubtractedPSFPhotometry(dao_BasicPSFPhotometry):
    """
    This class implements an iterative algorithm to perform point spread
    function photometry in crowded fields. This consists of applying a
    loop of find sources, make groups, fit groups, subtract groups, and
    then repeat until no more stars are detected or a given number of
    iterations is reached.

    Parameters
    ----------
    group_maker : callable or `~photutils.psf.GroupStarsBase`
        ``group_maker`` should be able to decide whether a given
        star overlaps with any other and label them as belonging
        to the same group. ``group_maker`` receives as input an
        `~astropy.table.Table` object with columns named as ``id``,
        ``x_0``, ``y_0``, in which ``x_0`` and ``y_0`` have the same
        meaning of ``xcentroid`` and ``ycentroid``. This callable must
        return an `~astropy.table.Table` with columns ``id``, ``x_0``,
        ``y_0``, and ``group_id``. The column ``group_id`` should
        contain integers starting from ``1`` that indicate which group a
        given source belongs to. See, e.g., `~photutils.psf.DAOGroup`.
    bkg_estimator : callable, instance of any \
            `~photutils.background.BackgroundBase` subclass, or None
        ``bkg_estimator`` should be able to compute either a scalar
        background or a 2D background of a given 2D image. See, e.g.,
        `~photutils.background.MedianBackground`.  If None, no
        background subtraction is performed.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.sandbox.DiscretePRF`,
        `~photutils.psf.IntegratedGaussianPRF`, or any other suitable 2D
        model.  This object needs to identify three parameters (position
        of center in x and y coordinates and the flux) in order to set
        them to suitable starting values for each fit. The names of
        these parameters should be given as ``x_0``, ``y_0`` and
        ``flux``.  `~photutils.psf.prepare_psf_model` can be used to
        prepare any 2D model to match this assumption.
    fitshape : int or length-2 array-like
        Rectangular shape around the center of a star which will be
        used to collect the data to do the fitting. Can be an integer
        to be the same along both axes. For example, 5 is the same as
        (5, 5), which means to fit only at the following relative pixel
        positions: [-2, -1, 0, 1, 2]. Each element of ``fitshape`` must
        be an odd number.
    finder : callable or instance of any \
            `~photutils.detection.StarFinderBase` subclasses
        ``finder`` should be able to identify stars, i.e., compute a
        rough estimate of the centroids, in a given 2D image.
        ``finder`` receives as input a 2D image and returns an
        `~astropy.table.Table` object which contains columns with names:
        ``id``, ``xcentroid``, ``ycentroid``, and ``flux``. In which
        ``id`` is an integer-valued column starting from ``1``,
        ``xcentroid`` and ``ycentroid`` are center position estimates of
        the sources and ``flux`` contains flux estimates of the sources.
        See, e.g., `~photutils.detection.DAOStarFinder` or
        `~photutils.detection.IRAFStarFinder`.
    fitter : `~astropy.modeling.fitting.Fitter` instance
        Fitter object used to compute the optimized centroid positions
        and/or flux of the identified sources. See
        `~astropy.modeling.fitting` for more details on fitters.
    aperture_radius : float
        The radius (in units of pixels) used to compute initial
        estimates for the fluxes of sources. If ``None``, one FWHM will
        be used if it can be determined from the ```psf_model``.
    niters : int or None
        Number of iterations to perform of the loop FIND, GROUP,
        SUBTRACT, NSTAR. If None, iterations will proceed until no more
        stars remain.  Note that in this case it is *possible* that the
        loop will never end if the PSF has structure that causes
        subtraction to create new sources infinitely.
    extra_output_cols : list of str, optional
        List of additional columns for parameters derived by any of the
        intermediate fitting steps (e.g., ``finder``), such as roundness
        or sharpness.

    Notes
    -----
    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g., manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at:
        https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract
    """

    def __init__(self, group_maker, bkg_estimator, psf_model, fitshape,
                 finder, fitter=LevMarLSQFitter(), niters=3,
                 aperture_radius=None, extra_output_cols=None):

        super().__init__(group_maker, bkg_estimator, psf_model, fitshape,
                         finder, fitter, aperture_radius, extra_output_cols)
        self.niters = niters

    @property
    def niters(self):
        return self._niters

    @niters.setter
    def niters(self, value):
        if value is None:
            self._niters = None
        else:
            try:
                if value <= 0:
                    raise ValueError('niters must be positive.')
                else:
                    self._niters = int(value)
            except ValueError:
                raise ValueError('niters must be None or an integer or '
                                 'convertable into an integer.')

    @property
    def finder(self):
        return self._finder

    @finder.setter
    def finder(self, value):
        if value is None:
            raise ValueError("finder cannot be None for "
                             "IterativelySubtractedPSFPhotometry - you may "
                             "want to use BasicPSFPhotometry. Please see the "
                             "Detection section on photutils documentation.")
        else:
            self._finder = value

    def do_photometry(self, image, init_guesses=None,image_weights=None):
        """
        Perform PSF photometry in ``image``.

        This method assumes that ``psf_model`` has centroids and flux
        parameters which will be fitted to the data provided in
        ``image``. A compound model, in fact a sum of ``psf_model``,
        will be fitted to groups of stars automatically identified by
        ``group_maker``. Also, ``image`` is not assumed to be background
        subtracted.  If ``init_guesses`` are not ``None`` then this
        method uses ``init_guesses`` as initial guesses for the
        centroids. If the centroid positions are set as ``fixed`` in the
        PSF model ``psf_model``, then the optimizer will only consider
        the flux as a variable.

        Parameters
        ----------
        image : 2D array-like, `~astropy.io.fits.ImageHDU`, \
                `~astropy.io.fits.HDUList`
            Image to perform photometry.
        init_guesses: `~astropy.table.Table`
            Table which contains the initial guesses (estimates) for the
            set of parameters. Columns 'x_0' and 'y_0' which represent
            the positions (in pixel coordinates) for each object must be
            present.  'flux_0' can also be provided to set initial
            fluxes.  If 'flux_0' is not provided, aperture photometry is
            used to estimate initial values for the fluxes. Additional
            columns of the form '<parametername>_0' will be used to set
            the initial guess for any parameters of the ``psf_model``
            model that are not fixed. If ``init_guesses`` supplied with
            ``extra_output_cols`` the initial values are used; if the columns
            specified in ``extra_output_cols`` are not given in
            ``init_guesses`` then NaNs will be returned.

        Returns
        -------
        output_table : `~astropy.table.Table` or None
            A table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process. Uncertainties on the fitted parameters
            are reported as columns called ``<paramname>_unc`` provided
            that the fitter object contains a dictionary called
            ``fit_info`` with the key ``param_cov``, which contains the
            covariance matrix.
        """
        if init_guesses is not None:
            table = super().do_photometry(image, init_guesses,image_weights)
            table['iter_detected'] = np.ones(table['x_fit'].shape, dtype=int)

            # n_start = 2 because it starts in the second iteration
            # since the first iteration is above
            output_table = self._do_photometry(n_start=2,image_weights=image_weights)
            output_table = vstack([table, output_table])
        else:
            if self.bkg_estimator is not None:
                self._residual_image = image - self.bkg_estimator(image)
            else:
                self._residual_image = image

            if self.aperture_radius is None:
                self.set_aperture_radius()

            output_table = self._do_photometry()

        output_table.meta = {'version': _get_version_info()}

        return QTable(output_table)

    def _do_photometry(self, n_start=1,image_weights=None):
        """
        Helper function which performs the iterations of the photometry
        process.

        Parameters
        ----------
        n_start : int
            Integer representing the start index of the iteration.  It
            is 1 if init_guesses are None, and 2 otherwise.

        Returns
        -------
        output_table : `~astropy.table.Table` or None
            Table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process.
        """
        output_table = QTable()
        self._define_fit_param_names()

        for (init_parname, fit_parname) in zip(self._pars_to_set.keys(),
                                               self._pars_to_output.keys()):
            output_table.add_column(Column(name=init_parname))
            output_table.add_column(Column(name=fit_parname))

        sources = self.finder(self._residual_image)

        n = n_start
        while((sources is not None and len(sources) > 0) and
              (self.niters is None or n <= self.niters)):
            positions = np.transpose((sources['xcentroid'],
                                      sources['ycentroid']))
            apertures = CircularAperture(positions,
                                         r=self.aperture_radius)
            sources['aperture_flux'] = aperture_photometry(
                self._residual_image, apertures)['aperture_sum']

            init_guess_tab = QTable(names=['id', 'x_0', 'y_0', 'flux_0'],
                                    data=[sources['id'], sources['xcentroid'],
                                          sources['ycentroid'],
                                          sources['aperture_flux']])
            self._get_additional_columns(sources, init_guess_tab)

            for param_tab_name, param_name in self._pars_to_set.items():
                if param_tab_name not in (['x_0', 'y_0', 'flux_0']):
                    init_guess_tab.add_column(
                        Column(name=param_tab_name,
                               data=(getattr(self.psf_model,
                                             param_name) *
                                     np.ones(len(sources)))))

            star_groups = self.group_maker(init_guess_tab)
            table, self._residual_image = super().nstar(
                self._residual_image, star_groups,image_weights)

            star_groups = star_groups.group_by('group_id')
            table = hstack([star_groups, table])

            table['iter_detected'] = n * np.ones(table['x_fit'].shape,
                                                 dtype=int)

            output_table = vstack([output_table, table])

            # do not warn if no sources are found beyond the first iteration
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', NoDetectionsWarning)
                sources = self.finder(self._residual_image)

            n += 1

        return output_table