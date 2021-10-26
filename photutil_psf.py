#!/usr/bin/env python
from __future__ import print_function

import sys, os,re,math
sys.path.append(os.path.join(os.environ['PIPE_SRC'],'pythonscripts'))
import optparse
import numpy as np
import scipy
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
import os

if 'PIPE_SRC' in os.environ:
    sys.path.append(os.environ['PIPE_SRC']+'/pydao')
if 'PIPE_PYTHONSCRIPTS' in os.environ:
    sys.path.append(os.path.join(os.environ['PIPE_PYTHONSCRIPTS'],'tools'))

#from tools import rmfile
from PythonPhot import djs_angle_match
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.datasets import load_simulated_hst_star_image
from photutils.datasets import make_noise_image
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.datasets import load_simulated_hst_star_image
from photutils.datasets import make_noise_image
from photutils.detection import find_peaks
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils import EPSFBuilder, GriddedPSFModel
from photutils.psf import DAOGroup, extract_stars, IterativelySubtractedPSFPhotometry
import pickle

def display_psf_grid(grid, zoom_in=True, figsize=(14, 12), scale_range=1e-4):
    """ Display a PSF grid in a pair of plots
    Shows the NxN grid in NxN subplots, repeated to show
    first the individual PSFs, and then their differences
    from the average PSF.
    At this time, this only visualizes a single GriddedPSFModel object,
    i.e.  covering one detector, not an entire instrument field of view.
    This function returns nothing, but makes some plots.
    Inputs
    -------
    grid : photutils.GriddedPSFModel object
        The grid of PSFs to be displayed.
    scale_range : float
        Dynamic range for display scale. vmin will be set to this factor timex vmax.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    tuple_to_int = lambda t: (int(t[0]), int(t[1]))

    def show_grid_helper(grid, data, title="Grid of PSFs", vmax=0, vmin=0, scale='log'):
        npsfs = grid.data.shape[0]
        n = int(np.sqrt(npsfs))

        fig, axes = plt.subplots(n, n, figsize=figsize)

        # Handle an edge case such that this function doesn't fail
        # for degenerate 1-PSF grids
        if n == 1:
            import warnings
            warnings.warn("Displaying a 1-element 'grid'; this will not be very interesting.")
            axes = np.asarray(axes)
            axes.shape = (1, 1)

        if scale == 'log':
            norm = matplotlib.colors.LogNorm()
        else:
            norm = matplotlib.colors.Normalize()

        for ix in range(n):
            for iy in range(n):
                i = ix*n+iy
                im = axes[n-1-iy, ix].imshow(data[i], vmax=vmax, vmin=vmin, norm=norm)
                axes[n-1-iy, ix].xaxis.set_visible(False)
                axes[n-1-iy, ix].yaxis.set_visible(False)
                axes[n-1-iy, ix].set_title("{}".format(tuple_to_int(grid.grid_xypos[i])))
                if zoom_in:
                    axes[n-1-iy,ix].use_sticky_edges = False
                    axes[n-1-iy,ix].margins(x=-0.25, y=-0.25)
        # plt.suptitle("{} for {} in {} \noversampling: {}x".format(title,
        #                                        grid.meta['detector'][0],
        #                                        grid.meta['filter'][0], grid.oversampling), fontsize=16)


        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("Intensity, relative to PSF sum = 1.0")


    vmax = grid.data.max()
    vmin = vmax*scale_range
    show_grid_helper(grid, grid.data, vmax=vmax, vmin=vmin)

    meanpsf = np.mean(grid.data, axis=0)
    diffs = grid.data - meanpsf
    vmax = np.abs(diffs).max()
    show_grid_helper(grid, diffs, vmax=vmax, vmin=-vmax, scale='linear', title='PSF differences from mean')


def calc_bkg(data,var_bkg=False):
    
    bkgrms = MADStdBackgroundRMS()
    mmm_bkg = MMMBackground()

    if var_bkg:
        print('Using 2D Background')
        sigma_clip = SigmaClip(sigma=3.)
        coverage_mask = (data == 0)

        bkg = Background2D(data, (100, 100), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=mmm_bkg,
                           coverage_mask=coverage_mask, fill_value=0.0)

        data_bkgsub = data.copy()
        data_bkgsub = data_bkgsub - bkg.background

        _, _, std = sigma_clipped_stats(data_bkgsub)

    else:

        std = bkgrms(data)
        bkg = mmm_bkg(data).astype(float)

        data_bkgsub = data.copy().astype(float)
        data_bkgsub -= bkg

    return data_bkgsub, std

def find_stars(data, fwhm,threshold=3, var_bkg=False):
    
    #print('Finding stars --- Detector: {d}, Filter: {f}'.format(f=filt, d=det))
    
    sigma_psf = fwhm

    #print('FWHM for the filter {f}:'.format(f=filt), sigma_psf, "px")
    
    data_bkgsub, std = calc_bkg(data,var_bkg=False)
    
    daofind = DAOStarFinder(threshold=threshold * std, fwhm=sigma_psf)
    found_stars = daofind(data_bkgsub)
    
    print('')
    print('Number of sources found in the image:', len(found_stars))
    print('-------------------------------------')
    print('')
    
    return found_stars

class photclass:
    def __init__(self):

        #import pipeclasses
        #self.params = pipeclasses.paramfileclass()
        #self.params.loadfile(os.environ['PIPE_PARAMS'])
        #self.params.loadfile(os.environ['EXTRAPARAMFILE'],addflag=True)
        
        self.psfrad = 15.0
        self.aprad = 1.0
        self.skyrad = 5.0
        self.PSFSNRthresh = 10.0
        self.SNRthresh = 10.0
        self.bpmval = 0x80
        self.diffimflag = False

        self.maxmaskfrac = 0.75

        self.unclobber = False

        # We'll use this near the end...
        self.sexmatched = False

    def add_options(self, parser=None, usage=None):
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")
        parser.add_option('-v', '--verbose', action="count", dest="verbose",default=1)
        parser.add_option('--debug', default=False, action="store_true",
                          help='debug mode: more output and debug files')
        parser.add_option('--unclobber', default=False, action="store_true",
                          help='don\'t clobber  output files')
        parser.add_option('--noiseim'  , default=None, type="string",
                          help='input noise filename (default=%default)')
        parser.add_option('--maskim'  , default=None, type="string",
                          help='input mask filename (default=%default)')
        parser.add_option('--psfstarlist'  , default=None, type="string",
                          help='list of stars for PSF determination (default=%default)')
        parser.add_option('--psfrad'  , default=15.0, type="float",
                          help='radius for PSF determination in units of pixels (default=%default)')
        parser.add_option('--aprad'  , default=1.0, type="float",
                          help='radius for aperture determination in units of FWHM (default=%default)')
        parser.add_option('--skyrad'  , default=5.0, type="float",
                          help='radius for sky determination in units of FWHM (default=%default)')
        parser.add_option('--PSFSNRthresh'  , default=15.0, type="float",
                          help='signal-to-noise threshold for PSF stars (default=%default)')
        parser.add_option('--SNRthresh'  , default=5.0, type="float",
                          help='signal-to-noise threshold (default=%default)')
        parser.add_option('--forcedSNRthresh'  , default=None, type="float",
                          help='signal-to-noise threshold to return a star in the forced list (default=%default)')
        parser.add_option('--bpmval'  , default=0x80, type="int",
                          help='bad pixel value in image or mask file (default=%default)')
        parser.add_option('--saturation'  , default=None, type="float",
                          help='saturation value (for noise calculation if needed) (default=%default)')
        parser.add_option('--gain'  , default=None, type="float",
                          help='gain value (for noise calculation if needed) (default=%default)')
        parser.add_option('--readnoise'  , default=None, type="float",
                          help='readnoise value (for noise calculation if needed) (default=%default)')
        parser.add_option('-d','--diffimflag', default=False, action="store_true",
                          help='this is a diffim')
        parser.add_option('-f','--forcedflag', default=False, action="store_true",
                          help='this is a diffim')
        parser.add_option('--forcelist'  , default=None, type="string",
                          help='filename for list of position for forced photometry (default=%default)')
        parser.add_option('--inputpsf'  , default=None, type="string",
                          help='input psf filename, used for diffims and/or forced photometry (default=%default)')
        parser.add_option('--fittedpsffilename'  , default=None, type="string",
                          help='outputfilename of psf residuals (default=%default)')
        parser.add_option('--sexstring'  , default=None, type="string",
                          help='commandline options for sextractor (ouch) (default=%default)')
        parser.add_option('--maxmaskfrac'  , default=0.75, type="float",
                          help='throw the towel if more than this fraction of image is masked (default=%default)')
        parser.add_option('--saturation'  , default=np.inf, type="float",
                          help='value above which pixels become saturated (default=%default)')
        parser.add_option('--minpixval'  , default=-1000, type="float",
                          help='minimum value for good pixels (default=%default)')
        parser.add_option('--contamthresh'  , default=3.0, type="float",
                          help='Look for sources around PSF stars above this SNR threshold (default=%default)')
        parser.add_option('--contamradius'  , default=40.0, type="float",
                          help='Radius around PSF stars free of contaminating sources (default=%default)')
        parser.add_option('--minfwhm'  , default=2.0, type="float",
                          help='Minimum image FWHM estimate for determining the PSF fitting radius (default=%default)')
        parser.add_option('--starchi2sigma'  , default=2.0, type="float",
                          help='Objects classified as stars will have PSF-fitting chi2 < starchi2sigma from PSF star mean (default=%default)')
        parser.add_option('--starchi2kw'  , default='MAXCHI2', type="string",
                          help='FITS keyword that contains the max PSF-fitting chi2 for classifying object as a star (default=%default)')
        parser.add_option('--starchi2num'  , default=5, type="float",
                          help='minimum number of PSF stars necessary to compute max chi2 for star classification (default=%default)')
        parser.add_option('--two_psf_iter'  , default=False, action="store_true",
                          help='Try two slightly different star lists for the PSF model (default=%default)')

        parser.add_option('--psftrim'  , default=False, action="store_true",
                          help="""turns on local PSF computation.  PSF stars will be chosen from a region 
nearby an object of interest.  This protects against a spatially varying PSF (default=%default)""")
        parser.add_option('--psftrimSizeDeg'  , default=None, type="float",
                          help='diameter of the region around the object in degrees where PSF stars can be selected (default=%default)')
        parser.add_option('--ObjRA'  , default=None, type="float",
                          help='right ascension of the object.  If provided, this is for local PSF computation (default=%default)')
        parser.add_option('--ObjDec'  , default=None, type="float",
                          help='declination of the object.  If provided, this is for local PSF computation (default=%default)')
        

        return(parser)

    def getmasknoise(self,image,noiseimfilename=None,maskimfilename=None,bpmval=None,
                     gain=None,saturation=None,readnoise=None):
        """Load the mask and noise images"""
        if maskimfilename != None:
            if self.verbose: print('Loading ',maskimfilename)
            try:
                tmp=pyfits.getdata(maskimfilename)
            except Exception as e:
                # This would be a good place for Python 3 exception chaining.
                raise RuntimeError ('failed to read FITS file %s: %s' % (maskimfilename, e))
            image_mask = scipy.where(scipy.logical_and(tmp,bpmval)>0 ,1,0)
        else:
            image_mask = np.zeros(image.shape)
            image_mask[scipy.where(image == bpmval)]=1
            image_mask[scipy.where(image >= saturation)]=1

        if noiseimfilename != None:
            if self.verbose: print('Loading ',noiseimfilename)
            try:
                image_noise=pyfits.getdata(noiseimfilename)
            except Exception as e:
                # This would be a good place for Python 3 exception chaining.
                raise RuntimeError ('failed to read FITS file %s: %s' % (noiseimfilename, e))
        else:
            if gain == None:
                 raise RuntimeError('Cannot determine noise since gain is None!')
            readnoise2=readnoise*readnoise
            image_noise = scipy.where(image_mask>0,0,np.sqrt(image/gain + readnoise2))

        return(image_noise,image_mask)

    def findallstars(self,image,image_mask,roundlim=[-100,100],
#                     sharplim=[-100,100],thresh=5.0,fwhm=None):     Annalisa decreased this from 5 to 3,  Dec. 15, 2015
                     sharplim=[-100,100],thresh=3.0,fwhm=None):
        """Find all stars in an image.  The main source finding happens with SExtractor
        in the runsex code.  This routine serves to find contaminating sources in the vicinity
        of PSF stars.
        """
        from PythonPhot import iterstat,find
        if not fwhm and self.verbose:
            print('Warning : Image FWHM not provided!  Using 2 pix...')
            fwhm = 2.0
        elif self.verbose:
            print('Using FWHM of %.1f pix'%fwhm)

        if fwhm < 0.5: fwhm = 0.5

        yy=np.where(image_mask == 0)
        if len(yy):
            md,std = iterstat.iterstat(image[yy])
        else:
            md,std = iterstat.iterstat(image)
        hmin=thresh*std

        try:
            xposall,yposall,flux,sharp,roundness = find.find(image,
                                                             hmin, fwhm,
                                                             roundlim, sharplim,
                                                             verbose=self.verbose)

        except find.FindError as e:
            print("fatal error finding stars:", e)
            sys.exit (1)

        if self.verbose:
            print("%i stars found with PythonPhot's FIND algorithm"%len(xposall))

        return(xposall,yposall)

    def clipForceList(self,xpos,ypos):
        if not self.ObjRA or not self.ObjDec or not self.psftrimSizeDeg:
            print('Warning : RA/Dec of an object of interest should be given when --psftrim flag is used')
            return(xpos,ypos)

        from astropy import wcs
        from astropy.coordinates import SkyCoord  # High-level coordinates
        from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
        from astropy.coordinates import Angle, Latitude, Longitude  # Angles
        import astropy.units as u
        cObj = SkyCoord(self.ObjRA,self.ObjDec,unit='deg')
        
        imwcs = wcs.WCS(self.hdr)

        maxsep = self.psftrimSizeDeg/2.
        
        iClose = np.array([],dtype='int')
        for i in range(len(xpos)):
            radeclist = imwcs.wcs_pix2world([(xpos[i],ypos[i])],0)
            (ra,dec) = radeclist[0]
            c = SkyCoord(ra,dec,unit='deg')

            SepDeg = c.separation(cObj)
            if SepDeg.value < maxsep:
                iClose = np.append(iClose,i)
        xpos,ypos = xpos[iClose],ypos[iClose]
        return(xpos,ypos)
                
    def clipPSFstars(self):
        if not self.ObjRA or not self.ObjDec or not self.psftrimSizeDeg:
            raise RuntimeError('Error : RA/Dec of an object of interest must be given when --psftrim flag is used')

        from astropy import wcs
        from astropy.coordinates import SkyCoord  # High-level coordinates
        from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
        from astropy.coordinates import Angle, Latitude, Longitude  # Angles
        import astropy.units as u
        cObj = SkyCoord(self.ObjRA,self.ObjDec,unit='deg')

        imwcs = wcs.WCS(self.hdr)

        maxsep = self.psftrimSizeDeg/2.
        
        iClose = np.array([],dtype='int')
        for i in range(len(self.sexdict_psfstars['x'])):
            radeclist = imwcs.wcs_pix2world([(self.sexdict_psfstars['x'][i],self.sexdict_psfstars['y'][i])],0)
            (ra,dec) = radeclist[0]
            c = SkyCoord(ra,dec,unit='deg')

            SepDeg = c.separation(cObj)
            if SepDeg.value < maxsep:
                iClose = np.append(iClose,i)
        for k in list(self.sexdict_psfstars.keys()):
            self.sexdict_psfstars[k] = self.sexdict_psfstars[k][iClose]
            
                
    def getPSFstars(self,psfstarfilename=None):
        """Find PSF stars to make the PSF model.  Looks
        for isolated, unmasked, unsaturated stars starting
        from an input list."""
        result = np.loadtxt(psfstarfilename, unpack=True)
        if result.size < 6:
            # We want at least 3 stars; above the test works correctly for
            # empty files, which lead to us getting an array of shape (0,).
            raise RuntimeError('Not enough PSF stars!!')
        xpos, ypos = result
        #peaks_tbl = find_peaks(self.image, threshold=500.)
        #xpos,ypos = peaks_tbl['x_peak'],peaks_tbl['y_peak']
        # Match stars to SExtractor detections
        # Does this code really match to 2 degrees?
        ntot,index,count=djs_angle_match.djs_angle_match(
            self.sexdict['x'], self.sexdict['y'],
            xpos,ypos,
            5, units="degrees",
            mmax=2)

        matchcols=np.where(index[:,0] > 0)[0]
        if len(matchcols) < 10:
            raise RuntimeError('Error : Not enough stars!!')
        import copy
        self.sexdict_psfstars = copy.deepcopy(self.sexdict)
        for k in list(self.sexdict.keys()):
            self.sexdict_psfstars[k] = self.sexdict[k][matchcols]

        if self.psftrim:
            self.clipPSFstars()
            
        # Remove saturated and masked stars
        # Let's do a few iterations if bright stars are hard to find

        # what do these do?!!!
        # forcex = 1000; forcey = 1000

        # find every possible object - implemented for ATLAS, should help for everything
        if not len(self.sexdict_psfstars['fwhm_image']):
            raise RuntimeError('No PSF stars!!')
        fwhm = np.median(self.sexdict_psfstars['fwhm_image'][self.sexdict_psfstars['fwhm_image'] > 0])
        xposall,yposall = self.findallstars(self.image,self.image_mask,
                                            thresh=self.contamthresh,fwhm=fwhm)

        # brightest stars first
        cols = np.argsort(self.sexdict_psfstars['flux'])[::-1]
        for k in list(self.sexdict_psfstars.keys()):
            self.sexdict_psfstars[k] = self.sexdict_psfstars[k][cols]

        sb = np.sort(self.sexdict_psfstars['flux'])[::-1]
        #(self.sexdict_psfstars['flux'] < sb[5]) &
        bright=np.where((self.sexdict_psfstars['flux']/self.sexdict_psfstars['fluxerr'] > self.PSFSNRthresh) &
                        (self.image_mask[self.sexdict_psfstars['y'].astype(int),
                                         self.sexdict_psfstars['x'].astype(int)] == 0))[0]

        unmaskedbright = []
        masky,maskx = np.shape(self.image_mask)
        for b in bright:
            coords0,coords1 = \
                self.sexdict_psfstars['x'][b].astype(int),self.sexdict_psfstars['y'][b].astype(int)
            n1,n2,n3,n4 = coords1-14,coords1+15,coords0-14,coords0+15
            if n1 < 0: n1 = 0
            if n3 < 0: n3 = 0
            if n2 > masky: n2 = masky
            if n4 > maskx: n4 = maskx
            if np.sum(self.image_mask[n1:n2,n3:n4]) == 0:
                sep = np.sqrt((self.sexdict_psfstars['x'][b] - xposall)**2. + \
                                  (self.sexdict_psfstars['y'][b] - yposall)**2.)
                if len (sep) < 2 or np.sort(sep)[1] > self.contamradius*fwhm:
                    unmaskedbright += [b]
        bright = np.array(unmaskedbright)

        # whoops - this lower part doesn't do anything
        # but I'm going to keep it that way for now, since really we shouldn't be using crowded stars
        if len(bright) < 3:
            if self.verbose: print('Warning : These PSF stars may be too crowded!!!')
            unmaskedbright = []
            for b in bright:
                coords0,coords1 = self.sexdict_psfstars['x'][b].astype(int),self.sexdict_psfstars['y'][b].astype(int)
                n1,n2,n3,n4 = coords1-14,coords1+15,coords0-14,coords0+15
                if n1 < 0: n1 = 0
                if n3 < 0: n3 = 0
                if n2 > masky: n2 = masky
                if n4 > maskx: n4 = maskx
                if np.sum(self.image_mask[n1:n2,n3:n4]) == 0:
                    unmaskedbright += [b]
        bright = np.array(unmaskedbright)

        if len(bright) < 3:
            print('Just %i bright stars to generate PSF'%len(bright))
            raise RuntimeError('Error : cannot find enough bright stars to generate PSF!!')


        self.brightfwhm = self.sexdict_psfstars['fwhm_image'][np.where(self.sexdict_psfstars['flux']/self.sexdict_psfstars['fluxerr'] > 10.0)]
        self.brightx = self.sexdict_psfstars['x'][bright]
        self.brighty = self.sexdict_psfstars['y'][bright]

        # Centroid the stars

    def runsex(self,imagefilename,noiseimfilename,maskimfilename,sexstring):
        """Run SExtractor on the image to find stellar parameters"""
        from subprocess import check_call
        from txtobj import txtobj
        from scipy.stats import norm
        from PythonPhot import cntrd

#        sexcommand = 'sex '
        sexcommand = 'sex '
        sexcommand += sexstring
        sexcommand += imagefilename
        # ADD WEIGHT MAP TO SEXSTRING
        if noiseimfilename:
            if maskimfilename:
                fname,fext = os.path.splitext(noiseimfilename)
                noiseimfilename_tmp = '%s_tmp%s'%(fname,fext)
                mask = pyfits.getdata(maskimfilename)
                hdu = pyfits.open(noiseimfilename)
                hdu[0].data[np.where(mask > 0)] = 1e4
                hdu.writeto(noiseimfilename_tmp,overwrite=True)
                sexcommand += ' -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE %s'%noiseimfilename_tmp
            else:
                sexcommand += ' -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE %s'%noiseimfilename
        try:
            check_call (sexcommand, shell=True)
        except Exception as e:
            print('error: sextractor invocation failed', file=sys.stderr)
            print('command was:', sexcommand, file=sys.stderr)
            raise
        if maskimfilename and noiseimfilename:
            print('removing temporary noise image %s'%noiseimfilename_tmp)
            os.system('rm %s'%noiseimfilename_tmp)

        # Read in the catalog
        sextable = txtobj(sexcommand.split('-CATALOG_NAME ')[-1].split(' ')[0],sexheader=True)

        # Remove stars near boundaries, saturated stars
        goodstars = np.array([],dtype='int')
        for x,y,i in zip(sextable.X_IMAGE,sextable.Y_IMAGE,
                         list(range(len(sextable.X_IMAGE)))):
            cols = np.where((self.image[int(y)-3:int(y)+4,int(x)-3:int(x)+4] < 1000.) &
                            (self.image_mask[int(y)-3:int(y)+4,int(x)-3:int(x)+4] > 0))
            if not len(cols[0]): goodstars = np.append(goodstars,[i])

        for k in list(sextable.__dict__.keys()):
            sextable.__dict__[k] = sextable.__dict__[k][goodstars]

        # Remove stars with large FWHM and ellipticity
        e1 = (sextable.CXX_IMAGE-sextable.CYY_IMAGE)/(sextable.CXX_IMAGE+sextable.CYY_IMAGE)
        e2 = 2*sextable.CXY_IMAGE//(sextable.CXX_IMAGE+sextable.CYY_IMAGE)
        ell = np.sqrt (e1**2+e2**2)
        pa = (np.arctan (e2, e1) / 2.)*180./np.pi+90

        histfw=np.histogram(sextable.FWHM_IMAGE,bins=74,range=[0.2,15.0])
        fwrange=np.arange(74)*.2+.2
        jj = np.where(histfw[0] == max(histfw[0]))[0]
        if len(jj) > 1: jj = jj[0]

        fwhmact=fwrange[jj]
        xe=np.where((ell < .5) &
                    (np.abs(sextable.FWHM_IMAGE-fwhmact) < fwhmact))[0]

        # Flag the stars with the number 1
        flags = np.zeros(len(sextable.X_IMAGE))
        xx = []
        if len(sextable.FWHM_IMAGE[np.where((np.isfinite(sextable.FWHM_IMAGE[xe])) &
                                            (np.abs(sextable.FWHM_IMAGE[xe]-np.median(sextable.FWHM_IMAGE[xe])) < .75*np.median(sextable.FWHM_IMAGE[xe])))[0]]) > 13:
            mu,sigma = norm.fit(sextable.FWHM_IMAGE[np.where((np.isfinite(sextable.FWHM_IMAGE[xe])) &
                                                             (np.abs(sextable.FWHM_IMAGE[xe]-np.median(sextable.FWHM_IMAGE[xe])) < .75*np.median(sextable.FWHM_IMAGE[xe])))])
            xx=np.where((np.abs(sextable.FWHM_IMAGE[xe]-mu) < 3.5*sigma))[0]# &
        if len(xx): flags[xe[xx]] = 1
        elif len(xe): flags[xe] = 1

        # Centroid those stars?
        xcen,ycen = cntrd.cntrd(self.image,sextable.X_IMAGE-1,sextable.Y_IMAGE-1,fwhmact)  #recenter on psf star
        cols = np.where((sextable.FLUX_AUTO == sextable.FLUX_AUTO) &
                        (xcen != -1) &
                        (ycen != -1) &
                        (xcen == xcen) & (ycen == ycen))[0]
        self.sexdict = {'x':xcen[cols],
                        'y':ycen[cols],
                        'fwhm_image':sextable.FWHM_IMAGE[cols],
                        'ellipt':sextable.ELLIPTICITY[cols],
                        'flags':flags[cols], # not the SExtractor flags
                        'class_star':sextable.CLASS_STAR[cols],
                        'fwhm_world':sextable.FWHM_WORLD[cols],
                        'flux':sextable.FLUX_AUTO[cols],
                        'fluxerr':sextable.FLUXERR_AUTO[cols],
                        'flag_sn':flags,
                        'CXX_IMAGE':sextable.CXX_IMAGE[cols],
                        'CXY_IMAGE':sextable.CXY_IMAGE[cols],
                        'CYY_IMAGE':sextable.CYY_IMAGE[cols]}

        pickle.dump(self.sexdict,open('sex_output.pkl','wb'))

    def getPSF(self,
               psfstarlist=None,gain=None,
               inputpsf=None,skipfirststar=False,
               outputpsffilename=None):
        """Load a PSF if one is provided, create a PSF
        from an input starlist if one is not."""

        if inputpsf != None:
            # PSF already exists, load it
            self.psf = pyfits.getdata(inputpsf)
            hpsf = pyfits.getheader(inputpsf)
            self.gauss = [hpsf['GAUSS1'],hpsf['GAUSS2'],hpsf['GAUSS3'],
                     hpsf['GAUSS4'],hpsf['GAUSS5']]
            self.psfmag = hpsf['PSFMAG']
            self.fwhm = 2.355*np.mean([hpsf['GAUSS4'],hpsf['GAUSS5']])
        else:
            # get PSF stars
            # Make sure they aren't offset by 1 pix
            self.getPSFstars(psfstarlist)

            if not self.brightfwhm.size:
                # This can happen, rarely.
                raise RuntimeError ('no bright FWHM stars were found')

            self.fwhm = np.median(self.brightfwhm)
            if self.fwhm < self.minfwhm: self.fwhm = self.minfwhm

            if self.verbose:
                print('Image FWHM for GETPSF set to %.1f pixels'%self.fwhm)

            xpsf,ypsf = self.brightx,self.brighty
            """
            
            peaks_tbl = find_peaks(self.image, threshold=500.)
            peaks_tbl.sort('peak_value')
            peaks_tbl.reverse()
            
            peaks_tbl['peak_value'].info.format = '%.8g'
            
            #x = peaks_tbl['x_peak']  
            #y = peaks_tbl['y_peak']
            
            
            stars = extract_stars(nddata, stars_tbl, size=25)  
            
            sys.exit()
            #norm = simple_norm(self.image, 'sqrt', percent=99.)
            #plt.imshow(self.image, norm=norm, origin='lower', cmap='viridis')
            for i in range(10):
                plt.scatter(peaks_tbl['x_peak'][i],peaks_tbl['y_peak'][i],marker='x',color='r')
                plt.scatter(xpsf[i],ypsf[i],marker='x',color='b')
            plt.show()
            sys.exit()
            """
            
            

            # get the PSF
            from PythonPhot import getpsf,aper
            # bright star aperture magnitudes
            skyrad = [self.skyrad*self.fwhm, (self.skyrad+3.0)*self.fwhm]

            apmag,apmagerr,flux,fluxerr,sky,skyerr,badflag,outstr = aper.aper(
                self.image,xpsf,ypsf,
                phpadu=self.gain,apr=self.aprad*self.fwhm,
                skyrad=skyrad,
                badpix=[self.minpixval,self.saturation],
                verbose=False)
            # Remove those bad stars
            goodphotcols = np.where(badflag == 0)[0]
            magcols = np.argsort(apmag[goodphotcols].reshape(goodphotcols.size))
            if skipfirststar:
                magcols = magcols[0:-1]
            apmag,apmagerr,flux,fluxerr,sky,skyerr,badflag = \
                apmag[goodphotcols][magcols],apmagerr[goodphotcols][magcols],flux[goodphotcols][magcols],\
                fluxerr[goodphotcols][magcols],sky[goodphotcols][magcols],skyerr[goodphotcols][magcols],\
                badflag[goodphotcols][magcols]
            xpsf,ypsf = xpsf[goodphotcols][magcols],ypsf[goodphotcols][magcols]
            # Added by Annalisa, Jan 12, 2016
            from astropy.stats import sigma_clipped_stats
            from photutils.psf import extract_stars
            from photutils.psf import EPSFBuilder
            #print(self.psfrad,self.aprad*self.fwhm - 1)
            #sys.exit()
            mean_val, median_val, std_val = sigma_clipped_stats(self.image, sigma=2.)
            data = self.image-median_val  
            size = self.psfrad
            hsize = (size - 1) / 2
            x,y = xpsf,ypsf
            mask = ((x > hsize) & (x < (self.image.shape[1] -1 - hsize)) &
                    (y > hsize) & (y < (self.image.shape[0] -1 - hsize)))  
            stars_tbl = Table()
            stars_tbl['x'] = x[mask]  
            stars_tbl['y'] = y[mask]   
            nddata = NDData(data=data)  
            stars = extract_stars(nddata, stars_tbl,size=size)  

            # ig, ax = plt.subplots(nrows=5, ncols=5, figsize=(20, 20),
            #             squeeze=True)
            # ax = ax.ravel()
            # for i in range(5*5):
            #     norm = simple_norm(stars[i], 'log', percent=99.)
            #     ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
            # plt.show()
            from photutils.psf import GriddedPSFModel
            psf_method = 'orig'
            nddata.meta['oversampling'] = 4

            
            #print(self.psf)
            #norm = simple_norm(self.psf, 'log', percent=99.)
            #fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10),
            #            squeeze=True)
            #ax = ax.ravel()
            #ax[0].imshow(self.psf, norm=norm, origin='lower', cmap='viridis')
            #plt.show()
            #sys.exit()
            
            #print('X and Y of PSF stars') 
            #print(xpsf,ypsf)
	    
            fitrad = self.aprad*self.fwhm - 1
            if fitrad >= self.psfrad: fitrad=self.psfrad-1
	    
            #print('Fitting radius:') 
            #print(fitrad)
            #print(len(xpsf))
	    
            #print(outputpsffilename)
	        
            
            if psf_method == 'ePSF':
                epsf_builder = EPSFBuilder(oversampling=4, maxiters=20,
                                           progress_bar=False)  
                epsf, fitted_stars = epsf_builder(stars)  
                self.psf=epsf.data
                self.psfmag=1
                self.gauss=None
                pyfits.writeto(self.psf.data,outputpsffilename)
            elif psf_method == 'gridded':
                self.psf = GriddedPSFModel(nddata)
            elif psf_method == 'gaussian':
                sigma_psf = 2
                self.psf = IntegratedGaussianPRF(sigma=sigma_psf)
            else:
                self.gauss,self.psf,self.psfmag = getpsf.getpsf(
                           self.image,xpsf,ypsf,
                           apmag,sky,self.rdnoise,
                           self.gain,list(range(len(xpsf))),
                           self.psfrad,fitrad,
                           outputpsffilename)

                print(self.gauss)
                #from astropy.modeling.models import Gaussian2D
                #mod = Gaussian2D(self.gauss[0],self.gauss[1],self.gauss[2],
                #    self.gauss[3],self.gauss[4])
                #fig,ax = plt.subplots(1,2)
                #ax[0].imshow(self.psf)
                
                #ax[1].imshow
                #plt.show()
                #sys.exit()
            #norm = simple_norm(self.psf, 'sqrt', percent=99.)
            
            #ax[1].imshow(self.psf, norm=norm, origin='lower', cmap='viridis')
            #plt.show()
            #sys.exit()
            if self.psfmag == -1:
                raise RuntimeError ('could not determine PSF')


    def pkfit(self, psffilename, psf, gauss, psfmag, forcelist=None):
        """Do the PSF fitting on all sources in an image. We optionally also fit
        for sources in the negated image, which is intended to catch sources that
        have gotten dimmer in difference images.

        """
        from PythonPhot import pkfit_norecent_noise, aper, rdpsf

        zeropoint = 25.
        fluxscale = 10**(0.4 * (zeropoint - psfmag))

        self.fwhm = 2.355 * np.mean([pyfits.getval(psffilename,'GAUSS4'),
                                     pyfits.getval(psffilename,'GAUSS5')])
        if self.verbose:
            print('PSF FWHM set to %.1f' % self.fwhm)

        if self.forcedflag:
            xpos, ypos = np.loadtxt(forcelist, unpack=True)
            if self.psftrim:
                xpos,ypos = self.clipForceList(xpos,ypos)
            # I think this is correct:
            xpos -= 1
            ypos -= 1
            if not isinstance(xpos, (list,tuple,np.ndarray)): xpos = np.array([xpos])
            if not isinstance(ypos, (list,tuple,np.ndarray)): ypos = np.array([ypos])
        else:
            xpos, ypos = self.sexdict['x'], self.sexdict['y']
        xpos, ypos = xpos[(xpos > 0) & (ypos > 0)], ypos[(xpos > 0) & (ypos > 0)]
        # We need to get the full PSF for our `pkflux` output; also convenient
        # place for debugging.
        fullpsf = rdpsf.rdpsf(psffilename)[0]
        ##pyfits.writeto(psffilename+'.full.test.fits', fullpsf, clobber=True)
        psfmax = fullpsf.max ()
        del fullpsf

        # First do aperture photometry, actually:
        apmag, apmagerr, apflux, apfluxerr, sky, skyerr, badflag, outstr = aper.aper(
            self.image,
            xpos, ypos,
            phpadu=self.gain,
            apr=self.aprad * self.fwhm,
            skyrad=[self.skyrad * self.fwhm, (self.skyrad + 3.0) * self.fwhm],
            badpix=[self.minpixval, self.saturation],
            verbose=False,
            exact=True,
            zeropoint=zeropoint)
        if not isinstance(sky, (list,tuple,np.ndarray)): sky = np.array([sky])
        # Now the PSF fitting.

        pk = pkfit_norecent_noise.pkfit_class(self.image, gauss, psf, self.rdnoise,
                                              self.gain, self.image_noise,
                                              self.image_mask)
        if self.do_neg_fit:
            negpk = pkfit_norecent_noise.pkfit_class(-self.image, gauss, psf,
                                                      self.rdnoise, self.gain,
                                                      self.image_noise,
                                                      self.image_mask)


        # For checking if we're at least k pixels away from a border. pkfit
        # uses a box of half-size fitrad, and the setting of "maxpix" below
        # uses a box of half-size 4, so:

        fitrad = max (self.aprad * self.fwhm - 1, 2)
        ny, nx = self.image.shape
        k = int(max (fitrad, 4))

        flux = np.empty (xpos.size)
        fluxerr = np.empty (xpos.size)
        chi = np.empty (xpos.size)
        sharp = np.empty (xpos.size)
        chi2 = np.empty (xpos.size)
        maxpix = np.empty (xpos.size)

        for arr in flux, fluxerr, chi, sharp, chi2, maxpix:
            arr.fill (np.nan)
            

        for i in range (xpos.size):
            x = xpos[i]
            y = ypos[i]
            s = sky[i]

            if not np.isfinite (x):
                continue

            if self.verbose:
                print('PSF Fitting x = %.2f, y = %.2f' % (x, y))

            ix = int(np.round (x))
            iy = int(np.round (y))
            if (ix < k) or (ix > nx - k) or (iy < k) or (iy > ny - k):
                continue # out of bounds

            if np.min(self.image[iy-k:iy+k+1, ix-k:ix+k+1]) == 0:
                continue # bad pixels, I guess?

            if np.sum(self.image_mask[iy-k:iy+k+1, ix-k:ix+k+1]) > 0:
                continue # masked pixels

            # `info` tuples are (errmag, chi, sharp, niter, scale, chi2)

            info = pk.pkfit_norecent_noise(1, x, y, s, fitrad, maxiter=50,
                                           returnchi2=True, verbose=self.verbose)

            if self.do_neg_fit:
                # Also fit the negated image, and figure out which fit looks
                # better.
                neginfo = negpk.pkfit_norecent_noise(1, x, y, -s, fitrad, maxiter=50,
                                                     returnchi2=True, verbose=self.verbose)
                poschi, possharp = info[1:3]
                negchi, negsharp = neginfo[1:3]

                if negchi <= 0 and poschi <= 0:
                    info = neginfo
                elif (poschi < 0.989 * negchi
                      or (poschi/negchi > 0.989 and poschi/negchi <= 1.01112235
                          and np.abs(possharp) < np.abs(negsharp))):
                    pass # stick with positive-image info
                else:
                    info = (neginfo[0], neginfo[1], neginfo[2], 
                            neginfo[3], -neginfo[4], neginfo[5])

            if np.isfinite(info[0]) and np.isfinite(info[4]) and not np.isnan(info[0]):
                flux[i] = info[4] * fluxscale
                fluxerr[i] = info[0] * fluxscale
            else:
                flux[i] = np.inf
                fluxerr[i] = np.inf

            chi[i] = info[1]
            sharp[i] = info[2]
            chi2[i] = info[5]
            maxpix[i] = np.max(self.image[iy-4:iy+5, ix-4:ix+5])

        # TODO: this would be a great place to use a Pandas DataFrame.
        print(apflux)
        self.pkdict = {
            'x': xpos,
            'y': ypos,
            'psfflux': flux,
            'psffluxerr': fluxerr,
            'chi2': chi2,
            'chi': chi,
            'sharp': sharp,
            'maxpix': maxpix - sky,
            'sky': sky,
            'apmag': apmag,
            'apmagerr': apmagerr,
            'apflux': apflux.reshape(len(xpos)),
            'apfluxerr': apfluxerr.reshape(len(xpos)),
            'pkflux': flux * psfmax / fluxscale
            }

        plt.hist(self.pkdict['chi2'])
        plt.show()
    def matchsex2dao(self,sextable=None,daotable=None):
        """Match up the SExtractor output to the DAO output"""

        keylist = []
        for k in list(sextable.keys()):
            if k not in list(daotable.keys()):
                daotable[k] = np.array([])
                keylist += [k]

        for x,y in zip(daotable['x'],daotable['y']):
            sep = np.sqrt((x-sextable['x'])**2. + (y-sextable['y'])**2.)
            col = np.where(sep == min(sep))[0]
            if len(col) > 1: col = [col[0]]
            for k in keylist:
                daotable[k] = np.append(daotable[k],sextable[k][col])

        return(daotable)

    def getmaxstarchi2(self, psfx, psfy, fittedpsffilename, ddict=None, writetofile=True):
        """Get the maximum chi2 for star classification based on an N-sigma
        cut on measured PSF star chi2 values.

        """
        from PythonPhot import iterstat

        chi2psf = np.empty (psfx.size)

        for i in range (psfx.size):
            sqsep = (ddict['x'] - psfx[i])**2 + (ddict['y'] - psfy[i])**2
            # NOTE: we're not checking that the separation is at all reasonable.
            chi2psf[i] = ddict['chi2'][np.argmin (sqsep)]

        chi2psf = chi2psf[np.isfinite(chi2psf)]

        if chi2psf.size < 3:
            if self.verbose:
                print ('warning: not enough chi2 values for PSF stars to determine maxstarchi2 cutoff')
            # Zero isn't a great indicator for "undefined maxchi2", but it's
            # what's used elsewhere in the code.
            maxchi2 = 0.
        else:
            md,std = iterstat.iterstat(chi2psf,sigmaclip=3.0)
            if not np.isfinite (md):
                if self.verbose:
                    print ('warning: iterstat() failed when determining maxstarchi2 cutoff')
                maxchi2 = 0.
            else:
                maxchi2 = md + self.starchi2sigma * std
                if self.verbose:
                    print('PSF stars have chi2 %.2f +/- %.2f' % (md, std))
                    print('Max chi2 for stars set to %.2f' % maxchi2)

        if writetofile:
            hdu = pyfits.open(fittedpsffilename)
            hdu[0].header[self.starchi2kw] = maxchi2
            hdu.writeto(fittedpsffilename,clobber=True)

        return maxchi2

    def qualcuts(self,obj = None, maxstarchi2 = 0):
        """Flag the bad objects, and try to decide what
        is and isn't a star."""

        # get the SExtractor output
        # this isn't used right now but was used in previous versions and
        # could be useful in the future?
        sexdict = self.matchsex2dao(sextable=self.sexdict,daotable=obj)

        flags=np.zeros(len(obj['x']),dtype=int)
        if self.forcedflag:
            flags[:] = 10

        xx=np.where((obj['psfflux'] == np.inf) |
                    (obj['psfflux'] == -np.inf) |
                    (obj['psfflux'] < 1e-10))[0]
        if len(xx):
            flags[xx] = flags[xx]+80

        xx=np.where((obj['chi2'] < maxstarchi2) &
                    (obj['psfflux'] != 1e6) &
                    (np.abs(obj['psfflux']) < 1e10) &
                    (np.abs(obj['psfflux']) > 0.0) &
                    (np.isfinite(obj['psfflux'])) &
                    (np.isfinite(obj['psffluxerr'])))[0]
        if len(xx): flags[xx]=flags[xx]+1
        print(len(np.where(flags == 1)[0]))
        print(len(flags))

        xx=np.where((obj['chi2'] >= maxstarchi2) &
                    (sexdict['class_star'] > 0.8) &
                    (obj['psfflux'] != 1e6) &
                    (np.abs(obj['psfflux']) < 1e10) &
                    (np.abs(obj['psfflux']) > 0.0) &
                    (np.isfinite(obj['psfflux'])) &
                    (np.isfinite(obj['psffluxerr'])))[0]
        if len(xx): flags[xx]=flags[xx]+7

        xx=np.where((obj['chi2'] >= maxstarchi2) &
                    (sexdict['class_star'] <= 0.8) &
                    (obj['psfflux'] != 1e6) &
                    (np.abs(obj['psfflux']) < 1e10) &
                    (np.abs(obj['psfflux']) > 0.0) &
                    (np.isfinite(obj['psfflux'])) &
                    (np.isfinite(obj['psffluxerr'])))[0]
        if len(xx): flags[xx]=flags[xx]+9

        xx=np.where((obj['psfflux'] == 1e6) |
                    (np.abs(obj['psfflux']) > 1e10) |
                    (np.abs(obj['psfflux']) == 0.0) |
                    (np.isfinite(obj['psfflux']) == 0) |
                    (np.isfinite(obj['psffluxerr']) == 0) |
                    (obj['psfflux'] != obj['psfflux']))[0]
        if len(xx): flags[xx]=flags[xx]+5

        obj['flag'] = flags
        return(obj)

    def writetofile(self,outputcat,ddict=None):
        """Write all the output"""

        # get the SExtractor output
        ddict = self.matchsex2dao(sextable=self.sexdict,daotable=ddict)

        if os.path.exists(outputcat) and noclobber:
            raise RuntimeError('Error : files %s exists!  Not clobbering'%outputcat)
        fout = open(outputcat,'w')
        print('#X Y flux fluxerror '+\
            'type peakval sigx sigxy sigy sky chisqr apphot apphoterr', file=fout)

        for i in range(len(ddict['x'])):
            print("%.2f  %.2f  %.3f  %.3f  0x000000%i  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f"%(
                ddict['x'][i],ddict['y'][i],
                ddict['psfflux'][i],ddict['psffluxerr'][i],
                ddict['flag'][i],ddict['pkflux'][i],
                ddict['CXX_IMAGE'][i],ddict['CXY_IMAGE'][i],ddict['CYY_IMAGE'][i],
                ddict['sky'][i],
                ddict['chi2'][i],ddict['apflux'][i],
                ddict['apfluxerr'][i]), file=fout)

        fout.close()

    def _set_psf_locations(self, num_psfs, psf_location=None):
        """Set the locations on the detector of the fiducial PSFs"""
        import itertools
        self.num_psfs = num_psfs

        if np.sqrt(self.num_psfs).is_integer():
            self.length = int(np.sqrt(self.num_psfs))
        else:
            raise ValueError("You must choose a square number of fiducial PSFs to create (E.g. 9, 16, etc.)")

        # Set the values
        if num_psfs == 1:
            # Want this case to be at the specified location
            location_list = [(psf_location[::-1])]  # tuple of (x,y)
        else:
            max_size = np.min(self.image.shape) - 1
            loc_list = [int(round(num * max_size)) for num in np.linspace(0, 1, self.length, endpoint=True)]
            location_list = list(itertools.product(loc_list, loc_list))  # list of tuples (x,y) (for WebbPSF)

        return location_list

    def to_model(self,data, meta):
        """
        Create a photutils GriddedPSFModel object from input data and meta information
        Parameters
        ----------
        data : ndarray
            3D numpy array of PSFs at different points across the detector
        meta : dict
            Dictionary containing meta data
        Returns
        -------
        model : GriddedPSFModel
            Photutils object with 3D data array and metadata with specified grid_xypos
            and oversampling keys
        """
        try:
            from photutils import GriddedPSFModel
        except ImportError:
            raise ImportError("This method requires photutils >= 0.6")

        ndd = NDData(data, meta=meta, copy=True)

        ndd.meta['grid_xypos'] = [((float(ndd.meta[key][0].split(',')[1].split(')')[0])),
                                  (float(ndd.meta[key][0].split(',')[0].split('(')[1])))
                                  for key in ndd.meta.keys() if "DET_YX" in key]

        ndd.meta['oversampling'] = meta["OVERSAMP"][0]  # just pull the value
        ndd.meta = {key.lower(): ndd.meta[key] for key in ndd.meta}

        model = GriddedPSFModel(ndd)

        return model

    def build_epsf(self, size=11, found_table=None, oversample=4, iters=10,create_grid=False):
        self.oversample=4
        self.num_psfs = 9
        data = self.image

        hsize = (size - 1) / 2
        
        x = self.brightx#found_table['xcentroid']
        y = self.brighty#found_table['ycentroid']
        
        mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) & (y > hsize) & (y < (data.shape[0] - 1 - hsize)))
        import astropy
        stars_tbl = Table()
        stars_tbl['x'] = x[mask]
        stars_tbl['y'] = y[mask]
        
        data_bkgsub, _ = calc_bkg(data)
        
        nddata = NDData(data=data_bkgsub)
        
        epsf_builder = EPSFBuilder(oversampling=self.oversample, maxiters=iters, progress_bar=True)    
        create_grid=True
        do_plot = True
        if create_grid:
            # Create an array to fill ([i, y, x])
            psf_size = self.psfrad * self.oversample
            
            self.location_list = self._set_psf_locations(self.num_psfs)
            psf_arr = np.empty((int(self.length**2), int(psf_size)+1, int(psf_size)+1))
            
            kernel = astropy.convolution.Box2DKernel(width=self.oversample)
            n=0
            m=0
            for i, loc in enumerate(self.location_list):
                if i%self.length<m:
                    n+=1
                m = i%self.length
                print(i,n,m)
                temp_star_tbl = stars_tbl[np.where(np.logical_and(np.logical_and((m+1)*self.image.shape[1]/self.length>=stars_tbl['x'],
                                                                                    stars_tbl['x']>=m*self.image.shape[1]/self.length),
                                                                    np.logical_and((n+1)*self.image.shape[0]/self.length>=stars_tbl['y'],
                                                                        stars_tbl['y']>=n*self.image.shape[0]/self.length)))[0]]
                
                #fig=plt.figure()
                #ax=fig.gca()
                #norm = simple_norm(self.image, 'sqrt', percent=99.)

                #ax.imshow(self.image, norm=norm, cmap='Greys')
                #ax.scatter(temp_star_tbl['x'],temp_star_tbl['y'])
                #plt.show()
                
                stars = extract_stars(nddata, temp_star_tbl, size=self.psfrad)
                epsf, fitted_stars = epsf_builder(stars)                
                #psf_conv = astropy.convolution.convolve(epsf.data, kernel)
                psf_arr[i, :, :] = epsf.data
            psf_arr *= self.oversample**2
            meta = {}
            meta["NUM_PSFS"] = (self.num_psfs, "The total number of fiducial PSFs")
            meta["OVERSAMP"] = (self.oversample, "Oversampling factor for FFTs in computation")
            for h, loc in enumerate(self.location_list):  # these were originally written out in (x,y)
                loc = np.asarray(loc, dtype=float)

                # Even arrays are shifted by 0.5 so they are centered correctly during calc_psf computation
                # But this needs to be expressed correctly in the header
                if self.psfrad % 2 == 0:
                    loc += 0.5  # even arrays must be at a half pixel

                meta["DET_YX{}".format(h)] = (str((loc[1], loc[0])),
                                              "The #{} PSF's (y,x) detector pixel position".format(h))
            epsf_model = self.to_model(psf_arr, meta)
            if do_plot:
                display_psf_grid(epsf_model)
                plt.show()
            
        else:
            #epsf_builder = EPSFBuilder(oversampling=oversample, maxiters=iters, progress_bar=True)
            stars = extract_stars(nddata, stars_tbl, size=size)
            epsf_model, fitted_stars = epsf_builder(stars)
        
        return epsf_model


    def doPhotutilsePSF(self,psfstarlist):
        #self.image=self.image.astype(float)-np.median(self.image)
        self.getPSFstars(psfstarlist)
        font2 = {'family': 'helvetica', 'color': 'black', 'weight': 'normal', 'size': '20'}
        import matplotlib.ticker as ticker

        if not self.brightfwhm.size:
            # This can happen, rarely.
            raise RuntimeError ('no bright FWHM stars were found')

        self.fwhm = np.median(self.brightfwhm)
        if self.fwhm < self.minfwhm: self.fwhm = self.minfwhm

        if self.verbose:
            print('Image FWHM for GETPSF set to %.1f pixels'%self.fwhm)
        found_stars = find_stars(self.image, self.fwhm,threshold=10, var_bkg=False)
        plot1 = False
        plot2 = False
        sh_inf = 0.2
        sh_sup = 0.6
        mag_lim = 0.0
        round_inf = -0.60
        round_sup = 0.60
        if plot1:
            plt.figure(figsize=(12, 8))
            plt.clf()

            ax1 = plt.subplot(2, 1, 1)

            ax1.set_xlabel('mag', fontdict=font2)
            ax1.set_ylabel('sharpness', fontdict=font2)

            xlim0 = np.min(found_stars['mag']) - 0.25
            xlim1 = np.max(found_stars['mag']) + 0.25
            ylim0 = np.min(found_stars['sharpness']) -0.15
            ylim1 = np.max(found_stars['sharpness']) +0.15

            ax1.set_xlim(xlim0, xlim1)
            ax1.set_ylim(ylim0, ylim1)

            ax1.xaxis.set_major_locator(ticker.AutoLocator())
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.yaxis.set_major_locator(ticker.AutoLocator())
            ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

            ax1.scatter(found_stars['mag'], found_stars['sharpness'], s=10, color='k')

            
            ax1.plot([xlim0, xlim1], [sh_sup, sh_sup], color='r', lw=3, ls='--')
            ax1.plot([xlim0, xlim1], [sh_inf, sh_inf], color='r', lw=3, ls='--')
            ax1.plot([mag_lim, mag_lim], [ylim0, ylim1], color='r', lw=3, ls='--')

            ax2 = plt.subplot(2, 1, 2)

            ax2.set_xlabel('mag', fontdict=font2)
            ax2.set_ylabel('roundness', fontdict=font2)

            ylim0 = np.min(found_stars['roundness2']) -0.25
            ylim1 = np.max(found_stars['roundness2']) -0.25

            ax2.set_xlim(xlim0, xlim1)
            ax2.set_ylim(ylim0, ylim1)

            ax2.xaxis.set_major_locator(ticker.AutoLocator())
            ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2.yaxis.set_major_locator(ticker.AutoLocator())
            ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

            

            ax2.scatter(found_stars['mag'], found_stars['roundness2'], s=10, color='k')

            ax2.plot([xlim0, xlim1], [round_sup, round_sup], color='r', lw=3, ls='--')
            ax2.plot([xlim0, xlim1], [round_inf, round_inf], color='r', lw=3, ls='--')
            ax2.plot([mag_lim, mag_lim], [ylim0, ylim1], color='r', lw=3, ls='--')

            plt.tight_layout()
            plt.show()
            sys.exit()
        mask = ((found_stars['mag'] < mag_lim) & (found_stars['roundness2'] > round_inf) & 
        (found_stars['roundness2'] < round_sup) & (found_stars['sharpness'] > sh_inf) 
        & (found_stars['sharpness'] < sh_sup))

        found_stars_sel = found_stars[mask]

        print('Number of stars selected to build ePSF:', len(found_stars_sel))

        # if we include the separation criteria:

        d = []

        # we do not want any stars in a 10 px radius. 

        min_sep = 10

        x_tot = found_stars['xcentroid']
        y_tot = found_stars['ycentroid']

        for xx, yy in zip(found_stars_sel['xcentroid'], found_stars_sel['ycentroid']):

            sep = []
            dist = np.sqrt((x_tot - xx)**2 + (y_tot - yy)**2)
            sep = np.sort(dist)[1:2][0]
            d.append(sep)

        found_stars_sel['min distance'] = d
        mask_dist = (found_stars_sel['min distance'] > min_sep)

        found_stars_sel2 = found_stars_sel[mask_dist]

        print('Number of stars selected to build ePSF \
        including "mimimum distance closest neighbour" selection):', len(found_stars_sel2))
        print('RAD:',self.psfrad)

        epsf = self.build_epsf(size=self.psfrad, found_table=found_stars_sel, oversample=2, iters=15)

        fitter = LevMarLSQFitter()
        mmm_bkg = MMMBackground()
        _,std = calc_bkg(self.image)
        th=10
        daofind = DAOStarFinder(threshold=th * std, fwhm=self.fwhm)
        
        daogroup = DAOGroup(5.0 * self.fwhm)
        phot = IterativelySubtractedPSFPhotometry(finder=daofind, group_maker=daogroup,
                                              bkg_estimator=mmm_bkg, psf_model=epsf,
                                              fitter=fitter,
                                              niters=5, fitshape=[int(self.aprad*self.fwhm - 1)]*2, aperture_radius=self.aprad, 
                                              extra_output_cols=('sharpness', 'roundness2'))
        result = phot(self.image)
        result.write('test_phot.dat',format='ascii',overwrite=True)


        #added to create output similar to daophot.py
        xfit,yfit,fluxfit,fluxerr = np.loadtxt('test_phot.dat',unpack=True,dtype={'names':('x','y','flux','fluxerr'),'formats':(float,float,float,'|S15')},usecols=(0,2,5,10),delimiter=' ',skiprows=1)
        fluxerr=fluxerr.astype('str') 
        dummylist=[0]*len(xfit)

        fout = open('outputcat','w')
        print ('#X Y flux fluxerror type peakval sigx sigxy sigy sky chisqr apphot apphoterr',file=fout)
        for i in range(len(xfit)):
                    print("%.3f  %.3f  %.3f  %s  %s  %i  %i  %i  %i  %i  %i  %i  %i"%(
                        xfit[i],yfit[i],
                        fluxfit[i],fluxerr[i],
                        dummylist[i],dummylist[i],
                        dummylist[i],dummylist[i],dummylist[i],
                        dummylist[i],
                        dummylist[i],dummylist[i],
                        dummylist[i]), file=fout)
        fout.close()

        residual_image = phot.get_residual_image()
        if plot2:
            plt.figure(figsize=(14, 14))

            ax1 = plt.subplot(1, 2, 1)

            plt.xlabel("X [px]", fontdict=font2)
            plt.ylabel("Y [px]", fontdict=font2)

            norm = simple_norm(self.image, 'sqrt', percent=99.)
            ax1.imshow(self.image, norm=norm, cmap='Greys')

            ax2 = plt.subplot(1, 2, 2)

            plt.xlabel("X [px]", fontdict=font2)
            plt.ylabel("Y [px]", fontdict=font2)
            plt.title('residuals', fontdict=font2)

            norm = simple_norm(self.image, 'sqrt', percent=99.)
            ax2.imshow(residual_image, norm=norm, cmap='Greys')

            plt.show()

            sys.exit()
            plt.figure(figsize=(12, 12))

            ax = plt.subplot(1, 1, 1)

            norm_epsf = simple_norm(epsf.data, 'log', percent=99.)
            ax.imshow(epsf.data, norm=norm_epsf)
            plt.tight_layout()
            plt.show()
        sys.exit()
    def doPhotutilsDAO(self,psfstarlist):
        import photutils
        import photutils.psf
        import photutils.psf.sandbox
        fwhm = np.median(self.sexdict['fwhm_image'][self.sexdict['fwhm_image'] > 0])
        self.getPSFstars(psfstarlist)

        if not self.brightfwhm.size:
            # This can happen, rarely.
            raise RuntimeError ('no bright FWHM stars were found')

        self.fwhm = np.median(self.brightfwhm)
        if self.fwhm < self.minfwhm: self.fwhm = self.minfwhm

        if self.verbose:
            print('Image FWHM for GETPSF set to %.1f pixels'%self.fwhm)

        xpsf,ypsf = self.brightx,self.brighty
    
        from PythonPhot import aper
        # bright star aperture magnitudes
        skyrad = [self.skyrad*self.fwhm, (self.skyrad+3.0)*self.fwhm]

        apmag,apmagerr,flux,fluxerr,sky,skyerr,badflag,outstr = aper.aper(
            self.image,xpsf,ypsf,
            phpadu=self.gain,apr=self.aprad*self.fwhm,
            skyrad=skyrad,
            badpix=[self.minpixval,self.saturation],
            verbose=False)
        # Remove those bad stars
        goodphotcols = np.where(badflag == 0)[0]
        magcols = np.argsort(apmag[goodphotcols].reshape(goodphotcols.size))
        
        apmag,apmagerr,flux,fluxerr,sky,skyerr,badflag = \
            apmag[goodphotcols][magcols],apmagerr[goodphotcols][magcols],flux[goodphotcols][magcols],\
            fluxerr[goodphotcols][magcols],sky[goodphotcols][magcols],skyerr[goodphotcols][magcols],\
            badflag[goodphotcols][magcols]
        xpsf,ypsf = xpsf[goodphotcols][magcols],ypsf[goodphotcols][magcols]

        #psf_model = photutils.psf.sandbox.DiscretePRF.create_from_image(self.image-np.median(self.image),
        #                Table([xpsf,ypsf],names=['x_0','y_0']),int(self.psfrad),mask=self.image_mask)
        psf_model = photutils.psf.IntegratedGaussianPRF(sigma=3.0)
        psf_model.sigma.fixed = False
        bkg = photutils.background.MMMBackground()
        thresh = 2.5*bkg(self.image-np.median(self.image))
        photometry = photutils.psf.DAOPhotPSFPhotometry(8,thresh,self.fwhm,psf_model,int((self.psfrad-1)/2),niters=2)
        result_tab = photometry(image=self.image-np.median(self.image))
        residual_image = photometry.get_residual_image()
        norm = simple_norm(residual_image, 'sqrt', percent=99.)
        plt.imshow(residual_image, norm=norm, origin='lower', cmap='viridis')
        plt.show()
        sys.exit()
    def dophotometry(self,imagefilename,outputcat,
                     noiseimfilename=None,maskimfilename=None,
                     gain=None,saturation=None,readnoise=None,
                     psfstarlist=None,
                     forcelist = None,
                     inputpsf =  None,
                     fittedpsffilename = None,
                     sexstring = None,
                     two_psf_iter = False):
        """The main routine.  Loads/creates a PSF model, and
        performs PSF fitting on an input list or SExtractor
        detections."""
        method = 'photutils_psf'
        if self.verbose>1:
            print('Removing ',outputcat)
        os.system('rm %s'%outputcat)
        if fittedpsffilename!=None:
            if inputpsf!=None and os.path.samefile(fittedpsffilename,inputpsf):
                raise RuntimeError('input PSF is the same as output PSF file (%s,%s)!! refusing to go on...' % (fittedpsffilename,inputpsf))
            if self.verbose>1:
                print('Removing fittedpsffilename',fittedpsffilename)
            os.system('rm %s'%fittedpsffilename)

        # load the image
        (self.image,self.hdr)=pyfits.getdata(imagefilename,0,header=True)

        # get the mask and noise
        (self.image_noise,self.image_mask) = self.getmasknoise(self.image,noiseimfilename=noiseimfilename,maskimfilename=maskimfilename,
                                                               bpmval = self.bpmval,
                                                               gain=gain,saturation=saturation,readnoise=readnoise)

        # check how masked the image is
        maskfraction = 1.0*scipy.sum(self.image_mask)/(self.hdr['NAXIS1']*self.hdr['NAXIS2'])
        if self.verbose>2: print('maskfraction',maskfraction)
        if self.maxmaskfrac!=None and maskfraction>self.maxmaskfrac:
            print('WARNING!!!! A large fraction (%f) of pixels are masked, throwing in the towel...' % maskfraction)
            sys.exit(0)


        # Run SExtractor to get star parameters
        #self.runsex(imagefilename,noiseimfilename,maskimfilename,sexstring)
        self.sexdict = pickle.load(open('newsex_ps.pkl','rb'))
        self.sexdict = {key:np.array(self.sexdict[key]) for key in self.sexdict.keys()}
        # creates self.psf_model and self.fitted_phot
        # 
        if method == 'photutils_psf':
            self.doPhotutilsePSF(psfstarlist)


            self.phot_dict = self.create_photutils_dict()
            self.writetofile(phot_dict)
            sys.exit()
        elif method == 'photutils_dao':

            self.doPhotutilsDAO(psfstarlist)
            sys.exit()
    
        # get PSF
        self.getPSF(psfstarlist,gain=gain,inputpsf=self.inputpsf,outputpsffilename=self.fittedpsffilename)

        # do the PSF fitting thing
        if not self.inputpsf:
            self.pkfit(self.fittedpsffilename,self.psf,self.gauss,self.psfmag)
        else:
            self.pkfit(self.inputpsf,self.psf,self.gauss,self.psfmag,forcelist=forcelist)
        # The first PSF star matters a lot; therefore to generate the PSF,
        # we need a second iteration where we omit the first star, and then
        # we compare chi2 between the two
        if not self.inputpsf:
            # Assigning variables like this may not work
            import copy
            if two_psf_iter:
                self.pkdict_firstiter = copy.deepcopy(self.pkdict)
                self.getPSF(psfstarlist,gain=gain,inputpsf=inputpsf,skipfirststar=True,outputpsffilename='%s_2.idlpsf'%imagefilename)
                self.pkfit('%s_2.idlpsf'%imagefilename,self.psf,self.gauss,self.psfmag)

            # Now compare the two iterations
            if two_psf_iter:
                firstcols = np.where((self.pkdict_firstiter['psfflux'] > 10) &
                                     (self.pkdict_firstiter['psfflux'] < self.saturation))
                secondcols = np.where((self.pkdict['psfflux'] > 10) &
                                      (self.pkdict['psfflux'] < self.saturation))
                med = np.median(self.pkdict_firstiter['chi'][cols])
            else:
                cols = np.where((self.pkdict['psfflux'] > 10) &
                                (self.pkdict['psfflux'] < self.saturation))
                med = np.median(self.pkdict['chi'][cols])

            # Removed these lines - only one PSF model is needed
            if two_psf_iter:
                secondmed = np.median(self.pkdict['chi'][secondcols])
                if secondmed < firstmed:
                    os.system('cp %s_2.idlpsf %s'%(imagefilename,self.fittedpsffilename))
                os.system('rm %s_2.idlpsf'%(imagefilename,imagefilename))

            if self.verbose:
                print('Median chi2 for PSF model: %.2f'%med)
                if two_psf_iter:
                    print('Median chi2 for PSF model 2: %.2f'%secondmed)
            if not two_psf_iter and med > 500.: # and secondmed > 500.:
                raise RuntimeError('Error : No good PSF stars!!')
            elif med > 500. and secondmed > 500.:
                raise RuntimeError('Error : No good PSF stars!!')

        # make some cuts, distinguish stars from other
        if psfstarlist:
            # if you just created the PSF, AND there are > 10 PSF stars
            # get the pkfit chi2 distribution of PSF stars and make
            # some kind of sigma-clip
            if len(self.brightx) >= self.starchi2num:
                maxstarchi2 = self.getmaxstarchi2(self.brightx,self.brighty,
                                                  self.fittedpsffilename,ddict=self.pkdict)
            else:
                print('Warning: too few PSF stars for classification! Classifying all objects as non-stars')
                maxstarchi2 = 0
        elif inputpsf:
            # if the PSF is already created, read the max star chi2 from
            # the header
            try:
                maxstarchi2 = pyfits.getval(inputpsf,self.starchi2kw)
            except KeyError:
                print('Warning : Keyword MAXCHI2 not found! Classifying all objects as non-stars')
                maxstarchi2 = 0
        else:
            raise RuntimeError('Error : PSF star list or input PSF must be provided!!!')

        self.qualcuts(obj = self.pkdict,maxstarchi2 = maxstarchi2)

        self.writetofile(outputcat,
                         ddict = self.pkdict)

        print('SUCCESS DAOPHOT!!')

if __name__=='__main__':

    usagestring='USAGE: daophot.py image outputcat'

    dao=photclass()
    #dao.getPSF()

    parser = dao.add_options(usage=usagestring)
    options,  args = parser.parse_args()

    if len(args)!=2:
        parser.parse_args(args=['--help'])
        sys.exit(0)

    (imagefilename,outputcat)=args


    dao.verbose = options.verbose
    dao.debug = options.debug

    dao.psfrad = options.psfrad
    dao.aprad = options.aprad
    dao.skyrad = options.skyrad
    dao.PSFSNRthresh = options.PSFSNRthresh
    dao.SNRthresh = options.SNRthresh
    dao.bpmval = options.bpmval
    dao.diffimflag = options.diffimflag
    dao.maxmaskfrac = options.maxmaskfrac
    dao.inputpsf = options.inputpsf

    dao.unclobber = options.unclobber
    dao.minpixval = options.minpixval
    dao.saturation = options.saturation
    dao.forcedflag = options.forcedflag
    if options.forcelist:
        dao.forcedflag = True
    dao.fittedpsffilename = options.fittedpsffilename
    dao.contamthresh = options.contamthresh
    dao.contamradius = options.contamradius
    dao.minfwhm = options.minfwhm
    dao.starchi2num= options.starchi2num
    dao.starchi2sigma = options.starchi2sigma
    dao.starchi2kw = options.starchi2kw
    dao.do_neg_fit = dao.forcedflag and dao.diffimflag

    dao.psftrim = options.psftrim
    dao.psftrimSizeDeg = options.psftrimSizeDeg
    dao.ObjRA = options.ObjRA
    dao.ObjDec = options.ObjDec
    
    try:
        dao.gain = pyfits.getval(imagefilename,'GAIN')
    except:
        try:
            dao.gain = pyfits.getval(imagefilename,'EGAIN')
        except:
            raise RuntimeError('Error : GAIN not found in image header!!!')
    try:
        dao.rdnoise = pyfits.getval(imagefilename,'RDNOISE')
    except:
        try:
            dao.rdnoise = pyfits.getval(imagefilename,'ENOISE')
        except:
            raise RuntimeError('Error : RDNOISE not found in image header!!!')

    dao.dophotometry(imagefilename,outputcat,
                     noiseimfilename=options.noiseim,maskimfilename=options.maskim,
                     gain=options.gain,saturation=options.saturation,readnoise=options.readnoise,
                     psfstarlist=options.psfstarlist,
                     forcelist = options.forcelist,
                     inputpsf =  options.inputpsf,
                     fittedpsffilename =  options.fittedpsffilename,
                     sexstring = options.sexstring,
                     two_psf_iter = options.two_psf_iter)

    if dao.psftrim:
        print('RA,Dec,size,eventdist: %.7f %.7f %.7f %.7f'%(dao.ObjRA,dao.ObjDec,dao.psftrimSizeDeg,0.0))
    print("SUCCESS daophot.py")
