#!/usr/bin/env python
from __future__ import print_function

import sys, os,re,math
#sys.path.append(os.path.join(os.environ['PIPE_SRC'],'pythonscripts'))
import optparse
import numpy as np
import scipy

import astropy.io.fits as pyfits

import os

if 'PIPE_SRC' in os.environ:
    sys.path.append(os.environ['PIPE_SRC']+'/pydao')
if 'PIPE_PYTHONSCRIPTS' in os.environ:
    sys.path.append(os.path.join(os.environ['PIPE_PYTHONSCRIPTS'],'tools'))

#from tools import rmfile
from PythonPhot import djs_angle_match
from astropy.modeling.fitting import *
from astropy.stats import gaussian_sigma_to_fwhm
import matplotlib.pyplot as plt



from astropy.visualization import simple_norm

from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table

import pickle


from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import stats 
from astroquery.mast import Catalogs
import math
from astroquery.vizier import Vizier
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from astropy.io import fits
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter('ignore')

from photutil_classes import dao_IterativelySubtractedPSFPhotometry
from photutils.datasets import make_model_sources_image 



    
def temp_plant(args):
    i,inds,im_shape,psf_model,sources = args
    
    #print(sources[inds[i]:inds[i+1]])
    return make_model_sources_image(im_shape,psf_model,sources[inds[i]:inds[i+1]])

def create_gridregionfile(x,y,x2,y2,regionname,color,coords='image'):
    with open(regionname, 'w') as f:
        f.write('global dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n'.format(color))
        f.write('%s \n'%coords)
        for star in range(len(x)):
            xval1 = x[star]
            yval1 = y[star]
            xval2 = x2[star]
            yval2 = y2[star]
            f.write('line({ra1},{dec1},{ra2},{dec2}) # color={color}\n'.format(ra1=xval1, dec1=yval1,ra2=xval2,dec2=yval2,color=color[star]))

#     return (ra_wcs,dec_wcs,flux)
    f.close()

def create_pixregionfile(x,y,regionname,color,coords='image'):
        with open(regionname, 'w') as f:
            if isinstance(color,str):
                f.write('global color={0} dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n'.format(color))
                do_col = False
            else:
                do_col = True
                f.write('global dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
            f.write('%s \n'%coords)
            for star in range(len(x)):
                xval = x[star]
                yval = y[star]
                if do_col:
                    f.write('circle({ra},{dec},{radius}") # color={color}\n'.format(ra=xval, dec=yval,radius=12,color=color[star]))
                else:
                    f.write('circle({ra},{dec},{radius}")\n'.format(ra=xval, dec=yval,radius=12))

    #     return (ra_wcs,dec_wcs,flux)
        f.close()

def getPS1cat4table (ra,dec):
# Assume a table of observations, check RA/DEC to make sure overlapping
    #ra = '0:57:30.673'
    #dec = '+30:14:19.993'
    try:
        c = SkyCoord(ra, dec, unit=u.degree, frame='icrs')
    except:
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.degree), frame='icrs')
    

    # Download a catalog for the first coordinate with a large enough radius to
    # overlap with all catalogs
    radius = 1
    RAboxsize = DECboxsize = 0.6
    Mmax = 22.0

    # get the maximum 1.0/cos(DEC) term: used for RA cut
    minDec = c.dec.degree-0.5*DECboxsize
    if minDec<=-90.0:minDec=-89.9
    maxDec = c.dec.degree+0.5*DECboxsize
    if maxDec>=90.0:maxDec=89.9

    invcosdec = max(1.0/math.cos(c.dec.degree*math.pi/180.0),
                            1.0/math.cos(minDec  *math.pi/180.0),
                            1.0/math.cos(maxDec  *math.pi/180.0))

    ramin = c.ra.degree-0.5*RAboxsize*invcosdec
    ramax = c.ra.degree+0.5*RAboxsize*invcosdec
    decmin = c.dec.degree-0.5*DECboxsize
    decmax = c.dec.degree+0.5*DECboxsize
    vquery = Vizier(columns=['RAJ2000', 'DEJ2000',
                             'gmag', 'e_gmag',
                             'rmag', 'e_rmag',
                             'imag', 'e_imag',
#                              'zmag', 'e_zmag',
#                              'ymag', 'e_ymag',
                            'gKmag','e_gKmag',
                            'rKmag','e_rKmag',
                            'iKmag','e_iKmag',
                            'objID'],
    #                         'zKmag','e_zKmag',
    #                         'yKmag','e_yKmag'],
                    column_filters={'gmag':
                                    ('<%f' % Mmax)},
                    row_limit=100000)

    tbdata = vquery.query_region(c, width=('%fd' % radius),
                catalog='II/349/ps1')[0]
    tbdata.rename_column('RAJ2000', 'ra_ps1')
    tbdata.rename_column('DEJ2000', 'dec_ps1')
    tbdata.rename_column('gmag', 'PS1_g')
    tbdata.rename_column('e_gmag', 'PS1_g_err')
    tbdata.rename_column('rmag', 'PS1_r')
    tbdata.rename_column('e_rmag', 'PS1_r_err')
    tbdata.rename_column('imag', 'PS1_i')
    tbdata.rename_column('e_imag', 'PS1_i_err')
    tbdata.rename_column('objID', 'PS1_ID')


    mask = ((tbdata['ra_ps1']<ramax) & (tbdata['ra_ps1']>ramin) &
            (tbdata['dec_ps1']<decmax) & (tbdata['dec_ps1']>decmin) & (tbdata['PS1_i']-tbdata['iKmag']<0.05) )
    tbdata = tbdata[mask]

    # Mask table
    for key in tbdata.keys():
        if key not in ['ra_ps1','dec_ps1']:
            tbdata[key] = [str(dat) for dat in tbdata[key]]
    return(tbdata)


def frompixtoradec (x,y,fitsfile):
    w = WCS(fitsfile)
    ra_wcs=[0]*len(x) #creates a list with x elements
    dec_wcs=[0]*len(x)
    for i in range(len(x)):
        ra_wcs[i], dec_wcs[i] = w.wcs_pix2world(x[i], y[i], 1)
    ra_wcs=np.asarray(ra_wcs)
    dec_wcs=np.asarray(dec_wcs)
    
    return(ra_wcs,dec_wcs)

def compare_phot(ra1, dec1, ra2, dec2): #matches two catalogs
    cf = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    catalogf = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    max_sep = 1.0 * u.arcsec
    idxf, d2df, d3df = cf.match_to_catalog_3d(catalogf)
    sep_constraintf = d2df < max_sep
    c_matchesf = cf[sep_constraintf]
    catalog_matchesf = catalogf[idxf[sep_constraintf]]
    return ([sep_constraintf],[idxf],d2df)

def calc_zpt(mag_catalog, flux,raf,decf,racat,deccat):
    boolval,idx,d2d=compare_phot(raf,decf,racat,deccat)
    magcat=mag_catalog[idx][boolval]
    
    notnanflux=~np.isnan(flux)
    magcat=np.asarray(magcat,dtype=float)

    mageas=-2.5*np.log10(flux[boolval])
    zpt=magcat-mageas
    clipped=sigma_clip(zpt, sigma=3, maxiters=5,masked=False)
    
    return (np.mean(clipped))

def rms(mag1,mag2):
    residuals=mag1-mag2
    print('before clip',len(residuals))
    residuals_m=sigma_clip(residuals, sigma=3, maxiters=5,masked=True)
    residuals_sc = residuals[~residuals_m.mask]
    notnan=~np.isnan(residuals)
    residuals=residuals[notnan]
    notnan_sc=~np.isnan(residuals_sc)
    residuals_sc=residuals_sc[notnan_sc]
    print('after clip',len(residuals_sc))
    return(np.std(residuals),np.std(residuals_sc),len(residuals)-len(residuals_sc),residuals_m.mask)

def analyzedcmp (filename):
#     zp = fits.getval(filename,'ZPTMAG')
#     zp_err = fits.getval(filename,'ZPTMUCER')
    dcmpfile=filename
    final_mag = []
    final_mag_err =[]
    final_flux= []
    final_flux_err =[] 
    data_quality=[]
    dq = 0
    objtype_list =[]
    flag_list =[]
    ut_date_list =[]
    seeings=[]
    skys_list=[]
    filenames=[]
    exptimes=[]
    lenmatches=[]
    clip_perc_stars=[]
    airmasses=[]
    peakfluxes=[]
    chis=[]
    extendednesses=[]
    xs=[]
    ys=[]
    ras=[]
    decs=[]
    
    
    filecols = (0,1,2,4,5,6,7,11,12,13,18,19,21) 
    with open(filename,'r') as f:
        header=f.readline()
        x,y,magnitude,counts,unc,objType,peakflux,sky,chi,objClass,extendedness,objFlag,nMask = np.loadtxt(filename,unpack=True,usecols=filecols,skiprows=1,dtype='str')
        x=[float(i) for i in x]
        y=[float(i) for i in y]
        sky=[float(i) for i in sky]

        x=np.array(x)
        y=np.array(y)
        sky=np.array(sky)
        extendedness=np.array(extendedness)
        return (x,y,extendedness)
    

def residuals_extendedness_plot_PS1(root,phot,fits_fname,mag_PS,magerr_PS,ra_PS,dec_PS,extendedness,ra_PSdcmp,
        dec_PSdcmp,viziertable,ravizierps1,decvizierps1,filt,apflux,label=None):
    if label is None:
        label = ''


    raphot,decphot=frompixtoradec(phot['X'],phot['Y'],fits_fname)

    zpt=calc_zpt(viziertable['PS1_%s'%filt],phot['flux'],raphot,decphot,ravizierps1,decvizierps1)
    apzpt=calc_zpt(viziertable['PS1_%s'%filt],apflux['flux'],raphot,decphot,ravizierps1,decvizierps1)
    mag=-2.5*np.log10(phot['flux'])+zpt
    magerr=2.5*0.434*(phot['fluxerror']/phot['flux'])
    apmag=-2.5*np.log10(apflux['flux']).flatten()+apzpt

    zptfile='out_dir/zpt{0}.dat'.format(root)

    nfile=open(zptfile,'w')
    nfile.write('{0}'.format(zpt))
    nfile.close()


    apmagerr=2.5*0.434*(apflux['fluxerror']/apflux['flux']).flatten()
    boolval,idx,d2d=compare_phot(raphot,decphot,ra_PSdcmp,dec_PSdcmp)

    xs = phot['X'][boolval]
    ys = phot['Y'][boolval]

    mag= mag[boolval]
    magerr=magerr[boolval]
    apmag = apmag[boolval]
    apmagerr = apmagerr[boolval]
    raphot=raphot[boolval]
    decphot=decphot[boolval]

    extendedness=extendedness[idx][boolval]
    
    boolval,idx,d2d=compare_phot(raphot,decphot,ra_PS,dec_PS)
    xs = xs[boolval]
    ys = ys[boolval]
    
    mag= mag[boolval]
    magerr=magerr[boolval]
    apmag = apmag[boolval].flatten()
    apmagerr = apmagerr[boolval]
    extendedness=extendedness[boolval]
    mag_PS=mag_PS[idx][boolval].flatten()
    magerr_PS=magerr_PS[idx][boolval]
    magerr_PS=np.asarray(magerr_PS,float)
    
    apindex=np.argwhere(~np.isnan(apmag-mag_PS))
    index=np.argwhere(~np.isnan(mag-mag_PS))

    magerr=np.asarray(magerr)
    magerr_PS=np.asarray(magerr_PS)
    bad = np.where(np.abs(mag-mag_PS)>.1)[0]
    create_pixregionfile(xs[bad]+1,ys[bad]+1,'out_dir/large_residual{0}.reg'.format(root),color='red')
#     plt.errorbar(extendedness[index],(mag-mag_ap)[index],yerr=np.sqrt(magerr**2+magerr_ap**2)[index],label='{0}; RMS: {1:.3f}. {2} stars'.format(file,rms(mag_ap[index],mag[index]),len(mag[index])),marker='o',ls='none',capsize=4)
    print('epsf')
    phot_rms = rms(np.array(mag_PS[index]),np.array(mag[index]))
    plt.errorbar(extendedness,(mag-mag_PS),yerr=np.sqrt(magerr**2+magerr_PS**2),
                        label='{0}; RMS: {2:.3f} (clipping {1} stars={3:.3f}). {4} stars total'.format(label,phot_rms[2],phot_rms[0],phot_rms[1],len(mag)),marker='o',ls='none',capsize=4)
    print('apmag')
    ap_rms = rms(np.array(mag_PS[apindex]),np.array(apmag[apindex]))
    plt.errorbar(extendedness,(apmag-mag_PS),yerr=np.sqrt(apmagerr**2+magerr_PS**2),
        label='{0}; RMS: {2:.3f} (clipping {1} stars={3:.3f}). {4} stars total'.format('ap.',ap_rms[2],ap_rms[0],ap_rms[1],len(apmag)),marker='o',ls='none',capsize=4)

    res=(mag[index]-mag_PS[index])[~phot_rms[3]]#mag-mag_PS
    res=np.asarray(res)
    psf_extendedness = extendedness[index][~phot_rms[3]]
    apres = (apmag[apindex]-mag_PS[apindex])[~ap_rms[3]]#apmag-mag_PS
    ap_extendedness = extendedness[apindex][~ap_rms[3]]
    apres = np.asarray(apres)

    print (len(raphot[boolval]),len(decphot[boolval]),len(apmag), len(apmagerr),len(apmag[apindex]),len(mag_PS))

    #march302023
    fileap=open('out_dir/aperturephotometry_dao{0}.dat'.format(root),'w')
    fileap.write('# ra dec ap apphoterr\n')
    for i in range(len(apmag)):
        fileap.write('{0:.6f} {1:.6f} {2:.6f} {3:.6f}\n'.format(raphot[boolval][i],decphot[boolval][i],apmag[i],apmagerr[i]))

    fileap.close()


    aptzptfile='out_dir/aptzpt{0}.dat'.format(root)

    apfile=open(aptzptfile,'w')
    apfile.write('{0}'.format(apzpt))
    apfile.close()

    # print (raphot[boolval]).flatten()
    # print (raphot[boolval])


    def func(x, a, b):
        y = a*x + b
        return y
#     print (type(extendedness[index]),type((res)[index]))
#     print (extendedness[index].flatten())
    
    popt, pcov = curve_fit(func, xdata = psf_extendedness.flatten(), ydata = res.flatten())
    #print(file,popt,'rms:',rms(mag_PS,mag))
#     xdata=extendedness[index].flatten()
    xdata=np.linspace(-60,85,100)
    
    plt.plot(xdata, func(xdata, *popt),label='{0}, slope: {1:.5f}'.format(label,popt[0]))
 
    popt, pcov = curve_fit(func, xdata = ap_extendedness.flatten(), ydata = apres.flatten())
    #print(file,popt,'rms:',rms(mag_PS,mag))
#     xdata=extendedness[index].flatten()
    xdata=np.linspace(-60,85,100)
    
    plt.plot(xdata, func(xdata, *popt),label='{0}, slope: {1:.5f}'.format('ap.',popt[0]))    
#     plt.gca().invert_yaxis()

    plt.xlabel('extendedness',fontsize=12)
    plt.ylabel('photutils mag - PS (%s band)'%filt,fontsize=12)
    plt.ylim((-.2,.2))

    plt.title(root)

    plt.axvline(x=0,c='k',linestyle='--')
    plt.axhline(y=0,c='k',linestyle='--')

    plt.legend(loc='lower right',fontsize=12)

def residuals_extendedness_plot(file,mag_ap,magerr_ap,ra_PSdcmp,dec_PSdcmp,extendedness,viziertable,ravizierps1,decvizierps1,ra_ap,dec_ap,label=None):
    if label is None:
        label = os.path.splitext(file)[0]
    phot = Table.read(file,format='ascii')
    raphot,decphot=frompixtoradec(phot['X'],phot['Y'],'F15anh.g.101013_53_1933.sw.fits')
    zpt=calc_zpt(viziertable['PS1_g'],phot['flux'],raphot,decphot,ravizierps1,decvizierps1)
    mag=-2.5*np.log10(phot['flux'])+zpt
    magerr=2.5*0.434*(phot['fluxerror']/phot['flux'])
    boolval,idx,d2d=compare_phot(raphot,decphot,ra_PSdcmp,dec_PSdcmp)
    
    mag= mag[boolval]
    magerr=magerr[boolval]
    raphot=raphot[boolval]
    decphot=decphot[boolval]
    
    
    extendedness=extendedness[idx][boolval]
    
    boolval,idx,d2d=compare_phot(raphot,decphot,ra_ap,dec_ap)
    
    mag= mag[boolval]
    magerr=magerr[boolval]
    extendedness=extendedness[boolval]
    mag_ap=mag_ap[idx][boolval]
    magerr_ap=magerr_ap[idx][boolval]
    
    index=np.argwhere(~np.isnan(mag-mag_ap))
    magerr=np.asarray(magerr)
    magerr_ap=np.asarray(magerr_ap)
    
    
    
    
    
#     plt.errorbar(extendedness[index],(mag-mag_ap)[index],yerr=np.sqrt(magerr**2+magerr_ap**2)[index],label='{0}; RMS: {1:.3f}. {2} stars'.format(file,rms(mag_ap[index],mag[index]),len(mag[index])),marker='o',ls='none',capsize=4)
    plt.errorbar(extendedness,(mag-mag_ap),yerr=np.sqrt(magerr**2+magerr_ap**2),label='{0}; RMS: {1:.3f}. {2} stars'.format(label,rms(mag_ap,mag),len(mag)),marker='o',ls='none',capsize=4)

#     A = np.vstack([extendedness[index], np.ones(len(extendedness[index]))]).T
    
#     m, c = np.linalg.lstsq(A, mag-mag_ap, rcond=None)[0]
    res=mag-mag_ap
    res=np.asarray(res)

    def func(x, a, b):
        y = a*x + b
        return y
#     print (type(extendedness[index]),type((res)[index]))
#     print (extendedness[index].flatten())
    
    popt, pcov = curve_fit(func, xdata = extendedness[index].flatten(), ydata = res[index].flatten())
    print(file,popt,'rms:',rms(mag_ap,mag))
#     xdata=extendedness[index].flatten()
    xdata=np.linspace(-60,85,100)
    
    plt.plot(xdata, func(xdata, *popt),label='{0}, slope: {1:.5f}'.format(label,popt[0]))    
#     plt.gca().invert_yaxis()

    plt.xlabel('extendedness',fontsize=12)
    plt.ylabel('photutils mag - ap phot mag',fontsize=12)
    plt.ylim((-.1,.1))    

    plt.axvline(x=0,c='k',linestyle='--')
    plt.axhline(y=0,c='k',linestyle='--')

    plt.legend(loc='lower right',fontsize=12)

def create_data_for_plot_photutils_aperturephot(file,viziertable,ravizierps1,decvizierps1):
    phot = Table.read(file,format='ascii')
    raphot,decphot=frompixtoradec(phot['X'],phot['Y'],'F15anh.g.101013_53_1933.sw.fits')
    zpt=calc_zpt(viziertable['PS1_g'],phot['apphot'],raphot,decphot,ravizierps1,decvizierps1)
    mag_phot=-2.5*np.log10(phot['apphot'])+zpt
    magerr=2.5*0.434*(phot['apphoterr']/phot['apphot'])


    return (raphot,decphot,mag_phot,magerr)

def display_psf_grid(name,grid, zoom_in=True, figsize=(14, 12), scale_range=1e-4):
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
            norm = matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin)
        else:
            norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)

        for ix in range(n):
            for iy in range(n):
                i = ix*n+iy
                im = axes[n-1-iy, ix].imshow(data[i], norm=norm)
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
    plt.savefig('out_dir/psf_grid{0}.png'.format(name),format='png')
    plt.close()
    meanpsf = np.mean(grid.data, axis=0)
    diffs = grid.data - meanpsf
    vmax = np.abs(diffs).max()
    show_grid_helper(grid, diffs, vmax=vmax, vmin=-vmax, scale='linear', title='PSF differences from mean')
    plt.savefig('out_dir/psf_diff_grid{0}.png'.format(name),format='png')
    plt.close()

def calc_bkg(data,mask=None,fill_value = 0,var_bkg=False):
    from photutils.background import MMMBackground, MADStdBackgroundRMS,Background2D

    bkgrms = MADStdBackgroundRMS()
    mmm_bkg = MMMBackground()

    if var_bkg:
        print('Using 2D Background')
        from astropy.stats import SigmaClip
        
        sigma_clip = SigmaClip(sigma=3.)
        if mask is not None:
            coverage_mask = (mask > 0) & (mask!=fill_value)
            mask = (mask == fill_value)
        else:
            coverage_mask=None
        bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=mmm_bkg,
                           coverage_mask=coverage_mask,mask=mask,fill_value=np.nan)
        print(np.max(data),np.nanmax(bkg.background))

        data_bkgsub = data.copy()
        data_bkgsub = data_bkgsub - bkg.background#_rms

        #_, _, std = sigma_clipped_stats(data_bkgsub)
        return data_bkgsub, bkg.background,bkg.background_rms#std#_rms,std
    else:

        std = bkgrms(data)
        bkg = float(mmm_bkg(data))

        data_bkgsub = data.copy().astype(float)
        data_bkgsub -= bkg

        return data_bkgsub, bkg,std

def find_stars(data, fwhm,threshold=3, var_bkg=False):
    
    #print('Finding stars --- Detector: {d}, Filter: {f}'.format(f=filt, d=det))
    
    sigma_psf = fwhm

    #print('FWHM for the filter {f}:'.format(f=filt), sigma_psf, "px")
    
    data_bkgsub,bkg_calc, std = calc_bkg(data,var_bkg=False)
    
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
        parser.add_option('--psfRoutine'  , default='daopy', type="str",
                          help='Choice of fitting routine (daopy, dao, epsf)')

        parser.add_option('--epsfOversample'  , default=1, type="int",
                          help='EPSF psf model oversampling rate (default=%default)')
        parser.add_option('--epsfFitradius'  , default=None, type="float",
                          help='EPSF psf model fitradius (default is to guess based on fwhm)')
        parser.add_option('--doepsfgrid'  , default=False, action="store_true",
                          help='Use a spatially varying ePSF, a grid (See epsfgridsize; default=%default')
        parser.add_option('--epsfgridsize'  , default=3, type="int",
                          help='Size of grid (on a side) to use for spatially varying ePSF (default=%default')
        parser.add_option('--nepsfiters'  , default=10, type="int",
                          help='Number of fit iterations for the EPSFBuilder (default=%default')
        parser.add_option('--dcmpfilename'  , default=None, type="str",
                          help='Filename for a dcmp file (default=%default')
        parser.add_option('--plantFakes'  , default=False, action="store_true",
                          help='Go through a fake star planting routine (default=%default)')
        parser.add_option('--maskFill'  , default=1, type = float,
                          help='The mask value indicating missing data (e.g., chip gap) (default=%default)')
        parser.add_option('--catmagtransform'  , default=None, type = "str",
                          help='A file providing transformed mags to compare against. (default=%default)')
        parser.add_option('--doapphotclip'  , default=False, action="store_true",
                          help='Do a sigma clip of stars based on aperture photometry agreement with PS1 (default=%default)')
        parser.add_option('--minStarsPerGridCell'  , default=10, type="int",
                          help='Min number of stars per grid cell. (default=%default)')
        parser.add_option('--sn_x'  , default=None, type="int",
                          help='X location of SN for photometry (pixels). (default=%default)')
        parser.add_option('--sn_y'  , default=None, type="int",
                          help='Y location of SN for photometry (pixels). (default=%default)')
        
        
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
            
                
    def getPSFstars(self,psfstarfilename=None,starcat=None):
        """Find PSF stars to make the PSF model.  Looks
        for isolated, unmasked, unsaturated stars starting
        from an input list."""
        if psfstarfilename is not None:
            result = np.loadtxt(psfstarfilename, unpack=True)
            xpos, ypos = result
        else:
            xpos = self.sexdict['x']
            ypos = self.sexdict['y']
        if len(xpos) < 6:
                # We want at least 3 stars; above the test works correctly for
                # empty files, which lead to us getting an array of shape (0,).
                raise RuntimeError('Not enough PSF stars!!')

        
        if starcat is not None:
            xpos,ypos = np.loadtxt(starcat,unpack=True)


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

        unmaskednormal = []
        masky,maskx = np.shape(self.image_mask)
        for b in np.arange(0,len(self.sexdict_psfstars['x']),1):
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
                    unmaskednormal += [b]
        unmaskednormal = np.array(unmaskednormal)
        self.unmaskednormalx = self.sexdict_psfstars['x'][unmaskednormal]
        self.unmaskednormaly = self.sexdict_psfstars['y'][unmaskednormal]
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
        xcen,ycen = sextable.X_IMAGE,sextable.Y_IMAGE#cntrd.cntrd(self.image,sextable.X_IMAGE-1,sextable.Y_IMAGE-1,fwhmact)  #recenter on psf star
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

        pickle.dump(self.sexdict,open('out_dir/sex_output.pkl','wb'))

    def getPSF(self,
               psfstarlist=None,gain=None,
               inputpsf=None,skipfirststar=False,
               outputpsffilename=None,starcat=None):
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
            self.getPSFstars(psfstarlist,starcat)
            
            

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
            #nddata = NDData(data=data)  
            #stars =  stars(nddata, stars_tbl,size=size)  

            # ig, ax = plt.subplots(nrows=5, ncols=5, figsize=(20, 20),
            #             squeeze=True)
            # ax = ax.ravel()
            # for i in range(5*5):
            #     norm = simple_norm(stars[i], 'log', percent=99.)
            #     ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
            # plt.show()
            #from photutils.psf import GriddedPSFModel
            psf_method = 'orig'
            #nddata.meta['oversampling'] = 4

            
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
            if 'out_dir' not in forcelist:
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
        print(apflux,flux)
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

    def build_epsf(self, size=11, found_table=None, oversample=4, iters=10,norm_radius=10,
        create_grid=False,npsf=9):
        from photutils import EPSFBuilder, GriddedPSFModel
        from photutils.psf import DAOGroup, extract_stars
        from astropy.visualization import imshow_norm, MinMaxInterval, SqrtStretch, ZScaleInterval, ImageNormalize
        from photutils import detect_sources, SourceCatalog
        from astropy.stats import (sigma_clip, sigma_clipped_stats,
                           gaussian_fwhm_to_sigma,gaussian_sigma_to_fwhm)
        from photutils import EllipticalAperture, detect_threshold, deblend_sources
        from copy import copy,deepcopy
        from photutil_classes import dao_DAOStarFinder
        self.oversample=oversample
        self.num_psfs = npsf
        data = self.image

        hsize = (size - 1) / 2
        
        x = found_table['x_mean']#self.brightx#self.sexdict['x']#self.brightx#found_table['xcentroid']
        y = found_table['y_mean']#self.brighty#self.sexdict['y']#self.brighty#found_table['ycentroid']
        #mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) & (y > hsize) & (y < (data.shape[0] - 1 - hsize)))
        #import astropy
        stars_tbl = Table()
        stars_tbl['x'] = x#-1#[mask]
        stars_tbl['y'] = y#-1#[mask]
        
        data_bkgsub,bk_calculated, bkg_rms = calc_bkg(data,var_bkg=True,mask = self.image_mask,fill_value=self.maskFill)

        print(np.nanmax(data_bkgsub),np.nanmax(bk_calculated))
        temp = deepcopy(self.fits_image)
        temp[0].data = data_bkgsub
        temp.writeto('out_dir/bk_sub{0}.fits'.format(self.partial_root),overwrite=True) #here
        temp = deepcopy(self.fits_image)
        temp[0].data = bk_calculated
        temp.writeto('out_dir/bk{0}.fits'.format(self.partial_root),overwrite=True)
        norm = ImageNormalize(data_bkgsub, interval=ZScaleInterval())
        plt.imshow(data_bkgsub,norm=norm)
        plt.savefig('out_dir/bk_sub{0}.png'.format(self.partial_root))
        plt.close()
        plt.imshow(bk_calculated)
        plt.savefig('out_dir/bk{0}.png'.format(self.partial_root))
        plt.close()
        nddata = NDData(data=data_bkgsub)

        daofind = dao_DAOStarFinder(threshold=np.median(bkg_rms)*10,fwhm=self.fwhm,xycoords=np.array([stars_tbl['x'],stars_tbl['y']]).T)
        dao_stars = daofind(data_bkgsub)
        to_remove_sharp = sigma_clip(dao_stars['sharpness'],sigma=5,maxiters=5)
        to_remove_round = sigma_clip(dao_stars['roundness2'],sigma=5,maxiters=5)
        

        print('%i stars removed due to sharpness'%len(to_remove_sharp[to_remove_sharp.mask]))
        print('%i stars removed due to roundness'%len(to_remove_round[to_remove_round.mask]))
        print(len(stars_tbl),len(to_remove_sharp),len(to_remove_round),len(np.where(~to_remove_sharp.mask)[0]),len(np.where(to_remove_sharp.mask)[0]),
            len(np.where(~to_remove_round.mask)[0]),len(np.where(to_remove_round.mask)[0]))
        #stars_tbl = stars_tbl[np.where(np.logical_and(~to_remove_sharp.mask,~to_remove_round.mask))[0]]
        #plt.scatter(dao_stars['sharpness'],dao_stars['roundness2'],color='r')
        #plt.show()
        #sys.exit()
        bk_nd = NDData(data=bk_calculated)
        noise_nd = NDData(data=np.sqrt(self.image_noise))
        to_remove = []
        print(len(stars_tbl),"len(stars_tbl)")
        all_stars = extract_stars(nddata, stars_tbl, size=size+5)
        all_bk = extract_stars(bk_nd, stars_tbl, size=size+5)
        all_noise = extract_stars(noise_nd, stars_tbl, size=size+5)
        print (len(all_stars),len(stars_tbl))
        # for i in range(len(stars_tbl)):
        for i in range(len(all_stars)): #change cesar 6/23/2022: len stars_tbl different from len all_stars

            # print (i)

            # print (all_stars[i].data)
            if np.any(np.isnan(all_stars[i].data)):
                to_remove.append(i)

                continue

            # print (to_remove,'here')

            threshold = detect_threshold(all_stars[i].data, nsigma=10,background=np.zeros(all_stars[i].data.shape),error=all_noise[i].data)
            #print(stars_tbl[i])
            segm = detect_sources(all_stars[i].data,
                              threshold, npixels=5)

            # print (i,'here2')

            try:
                scat = SourceCatalog(all_stars[i].data, segm,background=np.zeros(all_stars[i].data.shape),
                                    error=all_noise[i].data)
            except:
                to_remove.append(i)
                continue

            cat = scat.to_table(np.append(scat.default_columns,['fwhm','elongation','ellipticity']))
            if len(cat)>1:
                # print(cat)
                # plt.imshow(all_stars[i].data,origin='lower')
                # plt.show()
                # sys.exit()
                to_remove.append(i)
                #cat = cat[np.argmin(np.sqrt((cat['xcentroid']-size/2)**2+(cat['ycentroid']-size/2)**2))]
            #if cat['elongation'].value>1.2:
            #    to_remove.append(i)

        print('Removing %i stars due to crowding or nans.'%len(to_remove))
        stars_tbl.remove_rows(to_remove)
        print('Fitting %i stars after all cuts.'%len(stars_tbl))
        np.savetxt('out_dir/cut_stars{0}'.format(self.partial_root),np.array(stars_tbl))
        #plt.hist(np.array(to_remove).flatten())
        #plt.show()
        #sys.exit()
        epsf_builder = EPSFBuilder(oversampling=self.oversample, maxiters=iters,norm_radius=norm_radius,recentering_maxiters=100,
            progress_bar=False,shape=[size*self.oversample]*2,recentering_boxsize=5)    
        #create_grid=creat
        do_plot = True
        if create_grid:
            from photutils.psf import GriddedPSFModel
            import astropy,random
            from photutil_classes import dao_GriddedPSFModel
            # Create an array to fill ([i, y, x])
            psf_size = size#self.psfrad * self.oversample
            self.location_list = self._set_psf_locations(self.num_psfs)

            print (self.location_list,"self.location_list")
            
            
            psf_arr = None 
            
            #kernel = astropy.convolution.Box2DKernel(width=self.oversample)
            n=0
            m=0
            meta = {'oversampling': self.oversample, 'grid_xypos': []}
            all_stars_x = []
            all_stars_y = []
            total = 0
            color = []
            line_color = []
            linex1 = []
            linex2 = []
            liney1 = []
            liney2 = []
            done = []
            self.epsfgrid = 'cluster'
            if self.epsfgrid =='uniform' or True:
                print ('uniform epsf')
                print (self.image.shape,'self.image.shape')
                got_final_cells = False

                while not got_final_cells and self.num_psfs>=2:
                    all_xp = []
                    all_yp = []
                    for i, loc in enumerate(self.location_list):
                        if i%self.length<m:
                            n+=1
                        m = i%self.length

                        xp = (m*self.image.shape[1]/self.length-1*m%2+(m+1)*self.image.shape[1]/self.length-1*m%2)/2
                        yp = ((n+1)*self.image.shape[0]/self.length-1*n%2+n*self.image.shape[0]/self.length-1*n%2)/2
                        all_xp.append(xp)
                        all_yp.append(yp)
                    n=0
                    m=0
                    got_final_cells = True
                    for i, loc in enumerate(self.location_list):
                        if i%self.length<m:
                           n+=1
                        m = i%self.length
                        xp = all_xp[i]#((m+1)*self.image.shape[1]/self.length+m*self.image.shape[1]/self.length)/2
                        yp = all_yp[i]#((n+1)*self.image.shape[0]/self.length+n*self.image.shape[0]/self.length)/2
                        to_keep = []
                        for j in range(len(stars_tbl)):
                            if j in done:
                                continue
                            distances = np.sqrt((np.array([tempxp for tempxp in all_xp])-stars_tbl[j]['x'])**2+\
                                (np.array([tempyp for tempyp in all_yp])-stars_tbl[j]['y'])**2)
                            if np.argmin(distances)==i:
                                to_keep.append(j)
                                done.append(j)
                        if len(to_keep)<self.minStarsPerGridCell:
                            print('Not enough stars with %ix%i grid of PSFs'%(np.sqrt(self.num_psfs),np.sqrt(self.num_psfs)))
                            while self.num_psfs>=2:
                                self.num_psfs-=1
                                if np.sqrt(self.num_psfs).is_integer():
                                    break
                            if self.num_psfs<2:
                                raise RuntimeError('Wanted gridded PSF but not enough stars!')
                            print('Trying %ix%i...'%(np.sqrt(self.num_psfs),np.sqrt(self.num_psfs)))
                            self.location_list = self._set_psf_locations(self.num_psfs)


                            done = []
                            n=0
                            m=0
                            got_final_cells = False
                            break

                if self.num_psfs<2:
                    raise RuntimeError('Wanted gridded PSF but not enough stars!')
                else:
                    print('Moving forward with a %ix%i grid.'%(np.sqrt(self.num_psfs),np.sqrt(self.num_psfs)))

                dao.create_root()




            else:
                import sklearn
                from sklearn.cluster import KMeans
                from kneed import KneeLocator   
                wcss = []
                self.mingridsize = 2
                self.maxgridsize = 5
                for i in range(self.mingridsize**2, self.maxgridsize**2):
                    kmeans = KMeans(n_clusters=i)
                    kmeans.fit(stars_tbl.to_pandas())
                    wcss.append(kmeans.inertia_)
                # n_clusters = np.round(KneeLocator(range(self.mingridsize**2, self.maxgridsize**2), wcss, S=1.0, curve="convex", direction="decreasing").elbow)
                n_clusters = 18

                print('Chosen %i clusters with kmeans'%n_clusters)
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(stars_tbl.to_pandas())
                colors = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]) for i in range(n_clusters)]
                for i in range(len(stars_tbl)):
                    plt.scatter(stars_tbl[i]['x'],stars_tbl[i]['y'],color=colors[kmeans.labels_[i]])
                plt.scatter([x[0] for x in kmeans.cluster_centers_],[x[1] for x in kmeans.cluster_centers_],color=colors,s=300)
                plt.close()
                self.location_list = kmeans.cluster_centers_
                self.num_psfs = n_clusters
                dao.create_root()
                plt.savefig('out_dir/star_clusters_{0}.png'.format(self.root))



                kl = KneeLocator(range(self.mingridsize**2, self.maxgridsize**2), wcss, S=1.0, curve="convex", direction="decreasing")
                kl.plot_knee()
                plt.savefig('knee.png')
            done = []
            n=0
            m=0
            for i, loc in enumerate(self.location_list):
                if self.epsfgrid =='uniform' or True:
                    if i%self.length<m:
                       n+=1
                    m = i%self.length
                    xp = all_xp[i]#((m+1)*self.image.shape[1]/self.length+m*self.image.shape[1]/self.length)/2
                    yp = all_yp[i]#((n+1)*self.image.shape[0]/self.length+n*self.image.shape[0]/self.length)/2
                    print(xp,yp)
                    meta['grid_xypos'].append((xp,yp))
                    to_keep = []
                    for j in range(len(stars_tbl)):
                        if j in done:
                            continue
                        distances = np.sqrt((np.array([tempxp for tempxp in all_xp])-stars_tbl[j]['x'])**2+\
                            (np.array([tempyp for tempyp in all_yp])-stars_tbl[j]['y'])**2)
                        if np.argmin(distances)==i:
                            to_keep.append(j)
                            done.append(j)
                    temp_star_tbl = stars_tbl[to_keep]
                else:

                    meta['grid_xypos'].append(loc)
                    #temp_star_tbl = stars_tbl[np.where(np.logical_and(np.logical_and((m+1)*self.image.shape[1]/self.length>=stars_tbl['x'],
                    #                                                                    stars_tbl['x']>=m*self.image.shape[1]/self.length),
                    #                                                    np.logical_and((n+1)*self.image.shape[0]/self.length>=stars_tbl['y'],
                    #                                                        stars_tbl['y']>=n*self.image.shape[0]/self.length)))[0]]
                    
                    temp_star_tbl = stars_tbl[np.where(kmeans.labels_==i)[0]]
                    #print(xp,yp)
                    #print(temp_star_tbl)
                    #sys.exit()

                
                total+=len(temp_star_tbl)
                rand_col = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
                for s in range(len(temp_star_tbl)):
                    color.append(rand_col)
                if self.epsfgrid =='uniform':
                    for s in range(4):
                        line_color.append(rand_col)

                    linex1.append(m*self.image.shape[1]/self.length-1*m%2)
                    linex2.append((m+1)*self.image.shape[1]/self.length-1*m%2)
                    liney1.append(n*self.image.shape[0]/self.length-1*n%2)
                    liney2.append(n*self.image.shape[0]/self.length-1*n%2)

                    linex1.append(m*self.image.shape[1]/self.length-1*m%2)
                    linex2.append((m+1)*self.image.shape[1]/self.length-1*m%2)
                    liney1.append((n+1)*self.image.shape[0]/self.length-1*n%2)
                    liney2.append((n+1)*self.image.shape[0]/self.length-1*n%2)

                    linex1.append(m*self.image.shape[1]/self.length-1*m%2)
                    linex2.append(m*self.image.shape[1]/self.length-1*m%2)                
                    liney1.append(n*self.image.shape[0]/self.length-1*n%2)
                    liney2.append((n+1)*self.image.shape[0]/self.length-1*n%2)

                    linex1.append((m+1)*self.image.shape[1]/self.length-1*m%2)
                    linex2.append((m+1)*self.image.shape[1]/self.length-1*m%2)                
                    liney1.append(n*self.image.shape[0]/self.length-1*n%2)
                    liney2.append((n+1)*self.image.shape[0]/self.length-1*n%2)
                # fig=plt.figure()
                # ax=fig.gca()
                # norm = simple_norm(self.image, 'sqrt', percent=99.)

                # ax.imshow(self.image, norm=norm, cmap='Greys')
                # ax.scatter(temp_star_tbl['x'],temp_star_tbl['y'])
                # ax.scatter(xp,yp,color='r')
                # plt.show()
                # plt.close()
                
                stars = extract_stars(nddata, temp_star_tbl, size=size)
                epsf, fitted_stars = epsf_builder(stars)  

                #print(size,epsf.data.shape,int(self.length**2))
                #plt.imshow(epsf.data*self.oversample**2)  
                #plt.show() 
                #sys.exit()
                #print(fitted_stars[0].flux)
                all_stars_x = np.append(all_stars_x,fitted_stars.center_flat[:,0])
                all_stars_y = np.append(all_stars_y,fitted_stars.center_flat[:,1])
                #meta['grid_xypos'].append((np.mean(temp_star_tbl['x']),np.mean(temp_star_tbl['y'])))
                #psf_conv = astropy.convolution.convolve(epsf.data, kernel)
                if psf_arr is None:
                    psf_arr = np.empty((int(self.num_psfs), int(epsf.data.shape[0]), 
                        int(epsf.data.shape[1])))
                #print(np.sum(epsf.data),np.sum(epsf.data / self.oversample**2))
                #sys.exit()
                psf_arr[i, :, :] = epsf.data# * self.oversample**2
            create_pixregionfile(all_stars_x+1,all_stars_y+1,'out_dir/fit_stars{0}.reg'.format(self.root),color=color)
            # create_pixregionfile(all_stars_x+1,all_stars_y+1,'out_dir/fit_stars.reg',color=color)

            if self.epsfgrid =='uniform':
                create_gridregionfile(linex1,liney1,linex2,liney2,'out_dir/fit_stars_grid{0}.reg'.format(self.root),line_color,coords='image')
            #psf_arr *= self.oversample**2
            print('final fit:',total)
            print(self.num_psfs,psf_arr.shape,len(meta['grid_xypos']))
            meta["NUM_PSFS"] = (self.num_psfs, "The total number of fiducial PSFs")
            meta["OVERSAMP"] = (self.oversample, "Oversampling factor for FFTs in computation")
            for h, loc in enumerate(self.location_list):  # these were originally written out in (x,y)
                loc = np.asarray(loc, dtype=float)

                # Even arrays are shifted by 0.5 so they are centered correctly during calc_psf computation
                # But this needs to be expressed correctly in the header
                if size % 2 == 0:
                    loc += 0.5  # even arrays must be at a half pixel

                meta["DET_YX{}".format(h)] = (str((loc[1], loc[0])),
                                              "The #{} PSF's (y,x) detector pixel position".format(h))
            if self.epsfgrid =='uniform':
                epsf_model = GriddedPSFModel(NDData(psf_arr, meta=meta))#self.to_model(psf_arr, meta)
            else:
                epsf_model = dao_GriddedPSFModel(NDData(psf_arr, meta=meta))#self.to_model(psf_arr, meta)
            if do_plot:
                display_psf_grid(self.root,epsf_model)
                # display_psf_grid(epsf_model)

            #     plt.show()
            #     plt.close()
            
        else:
            #epsf_builder = EPSFBuilder(oversampling=oversample, maxiters=iters, progress_bar=True)'

            #print(size,stars_tbl)
            stars = extract_stars(nddata, stars_tbl, size=size)
            #fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20),
            #            squeeze=True)
            #ax = ax.ravel()
            #for i in range(4):
            #    norm = simple_norm(stars[i], 'log', percent=99.)
            #    ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
            #plt.show()
            epsf_model, fitted_stars = epsf_builder(stars)
            #print(fitted_stars)
            all_stars_x = fitted_stars.center_flat[:,0]
            all_stars_y = fitted_stars.center_flat[:,1]
            #norm = simple_norm(epsf_model.data, 'log', percent=99.)
            #plt.imshow(epsf_model.data, norm=norm, origin='lower', cmap='viridis')
            #plt.show()
            
        return epsf_model,Table({'x_0':all_stars_x,'y_0':all_stars_y}),data_bkgsub,fitted_stars


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
        _,_,std = calc_bkg(self.image)
        th=10
        daofind = DAOStarFinder(threshold=th * std, fwhm=self.fwhm,
            xycoords=np.array([self.brightx,self.brighty]).T)
            #xycoords=np.array([self.sexdict['x'],self.sexdict['y']]).T)

        # epsf.x_0.fixed = True  #this seems to make the epsf photometry worse
        # epsf.y_0.fixed = True

        sources = Table()

        sources['x_mean'] = self.brightx#self.sexdict['x']
        sources['y_mean'] = self.brighty#self.sexdict['y']

        pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],sources['y_mean']])
        print (pos)


        daogroup = DAOGroup(5.0 * self.fwhm)
        phot = IterativelySubtractedPSFPhotometry(finder=daofind, group_maker=daogroup,
                                              bkg_estimator=mmm_bkg, psf_model=epsf,
                                              fitter=fitter,
                                              niters=5, fitshape=[int(self.aprad*self.fwhm - 1)]*2, aperture_radius=self.aprad*self.brightfwhm, 
                                              extra_output_cols=('sharpness', 'roundness2'))
        result = phot(self.image,init_guesses=pos)
        result.write('test_phot.dat',format='ascii',overwrite=True)


        #added to create output similar to daophot.py
        xfit,yfit,fluxfit,fluxerr = np.loadtxt('test_phot.dat',unpack=True,dtype={'names':('x','y','flux','fluxerr'),'formats':(float,float,float,'|S15')},usecols=(0,1,9,10),delimiter=' ',skiprows=1)
        fluxerr=fluxerr.astype('str') 
        dummylist=[0]*len(xfit)

        fout = open('outputcat_photutils','w')
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
    def doPhotutilsDAO(self,psfstarlist,imagefilename,sn_location=None):
        from photutils.psf import IntegratedGaussianPRF, DAOGroup
        from photutils.datasets import load_simulated_hst_star_image
        from photutils.datasets import make_noise_image
        from photutils.background import MMMBackground, MADStdBackgroundRMS,Background2D
        from photutils.datasets import load_simulated_hst_star_image
        from photutils.datasets import make_noise_image
        from photutils.detection import find_peaks
        from photutils.detection import DAOStarFinder
        
        from photutils.psf import DAOGroup

        fwhm = np.median(self.sexdict['fwhm_image'][self.sexdict['fwhm_image'] > 0])
        self.brightfwhm = fwhm
        if not self.brightfwhm.size:
            # This can happen, rarely.
            raise RuntimeError ('no bright FWHM stars were found')

        self.fwhm = np.median(self.brightfwhm)
        if self.fwhm < self.minfwhm: self.fwhm = self.minfwhm
        sources = Table()


        if self.plantFakes:
            sources['x_mean'] = psfstarlist['x_0']# xpsf #
            sources['y_mean'] = psfstarlist['y_0']# # ypsf #
        else:
            print(psfstarlist)
            if psfstarlist is not None:
                
                self.getPSFstars(psfstarlist,starcat=self.forcelist)

            if self.doapphotclip:
                print('Running aperture photometry first time...')
                x_input_ap=self.sexdict['x']
                y_input_ap=self.sexdict['y']

                ap_table=Table([x_input_ap,y_input_ap],names=['x','y'])

                self.photutils_aperture(ap_table,ap_rad=self.aprad*self.fwhm)
                self.catmagtransform_first = Table.read(self.catmagtransform,format='ascii')
                print ('Done with aperture photometry')
                ravizierps1=self.catmagtransform_first['ra']
                decvizierps1=self.catmagtransform_first['dec']
                filt = self.filter
                ps1gmag=self.catmagtransform_first[filt]
                ps1gmagerr=self.catmagtransform_first['d%s'%filt]
                viziertable = Table([ravizierps1,decvizierps1,ps1gmag,ps1gmagerr],names=['ra_ps1','dec_ps1','PS1_%s'%filt,'PS1_%s_err'%filt])

                raphot,decphot=frompixtoradec(self.aperture_result['X'],self.aperture_result['Y'],imagefilename)

                apzpt=calc_zpt(viziertable['PS1_%s'%filt],self.aperture_result['flux'],raphot,decphot,ravizierps1,decvizierps1)
                print (apzpt)

                apmag=-2.5*np.log10(self.aperture_result['flux']).flatten()+apzpt

                boolval,idx,d2d=compare_phot(raphot,decphot,ravizierps1,decvizierps1)

                apmag = apmag[boolval]
                ps1gmag=ps1gmag[idx][boolval]

                ap_first_residuals=apmag-ps1gmag


                # print ("here apphot")

                # print (np.sort(ap_first_residuals))

                print (rms(apmag,ps1gmag)[:-1])

                residuals_sc=sigma_clip(ap_first_residuals, sigma=3, maxiters=5,masked=True)

                print (len(self.sexdict['x'][boolval]),len(residuals_sc),len(~residuals_sc.mask),len(ap_first_residuals))

                # print (len(self.sexdict['x'][boolval][~residuals_sc.mask]),len(self.sexdict['x']))

                sources['x_mean']=self.sexdict['x'][boolval][~residuals_sc.mask]

                sources['y_mean']=self.sexdict['y'][boolval][~residuals_sc.mask]
                
                #sources = sources[np.where(apmag[~residuals_sc.mask]<19)[0]]
                #sources['x_mean'] = self.sexdict['x'] #self.unmaskednormalx#  #self.brightx
                #sources['y_mean'] = self.sexdict['y'] #self.unmaskednormaly#   #self.brighty
                #sources = sources[np.where(apmag[~residuals_sc.mask]>18)[0]]
                fwhm = np.median(self.sexdict['fwhm_image'][boolval][~residuals_sc.mask])
                
            else:
                sources['x_mean'] = self.sexdict['x']#self.sexdict['x'] #self.unmaskednormalx#  #self.brightx
                sources['y_mean'] = self.sexdict['y']#self.sexdict['y'] #self.unmaskednormaly#   #self.brighty

        #sources = sources[5:7]
        #sources.add_row(sources[0])
        #sources.remove_rows([1])
        print('Fitting %i sources...'%len(sources))
        self.brightfwhm = fwhm
        if not self.brightfwhm.size:
            # This can happen, rarely.
            raise RuntimeError ('no bright FWHM stars were found')

        self.fwhm = np.median(self.brightfwhm)
        if self.fwhm < self.minfwhm: self.fwhm = self.minfwhm

        if self.verbose:
            print('Image FWHM for GETPSF set to %.1f pixels'%self.fwhm)

        

        

        #print(sources)
        pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],sources['y_mean']])

        skyrad = [self.skyrad*self.fwhm, (self.skyrad+3.0)*self.fwhm]

        fitter = LevMarLSQFitter()# SLSQPLSQFitter()
        oversample_rate = self.epsfOversample
        if self.epsfFitradius is None:
            epsf_size = self.fwhm*5
        else:
            epsf_size = self.epsfFitradius

        import photutils
        from astropy.nddata.utils import add_array, extract_array
        from photutils.psf.utils import _extract_psf_fitting_names
        
        psf_model,fitted_star_locs,im_sub_bk,star_obj = self.build_epsf(size=epsf_size, found_table=sources, oversample=oversample_rate, \
            iters=self.nepsfiters,norm_radius=self.aprad*self.fwhm,npsf=self.epsfgridsize**2,create_grid=self.doepsfgrid) #HERE
                
        #print(fitted_star_locs)
        self.gridded_epsf = psf_model
        pickle.dump(self.gridded_epsf,open('out_dir/%s'%imagefilename.replace('.fits','_gridded_epsf.pkl'),'wb'))

        if False:
            indices = np.indices(self.image.shape)
            xname, yname, fluxname = _extract_psf_fitting_names(psf_model)
            subbeddata = im_sub_bk.astype(float)#self.image.astype(float)
            fit = star_obj#photutils.psf.EPSFFitter(fitter)(psf_model,star_obj)
            posflux = Table([fit.center_flat[:,0],fit.center_flat[:,1],fit.flux],names=['x_fit','y_fit','flux_fit'])
            for row in posflux:
                subshape = [int(epsf_size)]*2
                x_0, y_0 = row['x_fit'], row['y_fit']

                # float dtype needed for fill_value=np.nan
                y = extract_array(indices[0].astype(float), subshape, (y_0, x_0))
                x = extract_array(indices[1].astype(float), subshape, (y_0, x_0))

                getattr(psf_model, xname).value = x_0
                getattr(psf_model, yname).value = y_0
                getattr(psf_model, fluxname).value = row['flux_fit']
                import matplotlib.pyplot as plt
                from astropy.visualization import imshow_norm, MinMaxInterval, SqrtStretch, ZScaleInterval, ImageNormalize
                fig,ax=plt.subplots(1,3)
                subshape = list(subshape)
                subshape[0] = np.ceil(subshape[0]/2)
                norm = ImageNormalize(subbeddata[int(y_0-subshape[0]):int(y_0+subshape[0]),int(x_0-subshape[0]):int(x_0+subshape[0])], interval=ZScaleInterval())
                ax[0].imshow(subbeddata[int(y_0-subshape[0]):int(y_0+subshape[0]),int(x_0-subshape[0]):int(x_0+subshape[0])],norm=norm)
                ax[1].imshow(psf_model(x, y),norm=ImageNormalize(psf_model(x, y), interval=ZScaleInterval()))
                #subshape[0]*=2
                #print(subbeddata)
                #print(-1.*psf_model(x, y))
                subbeddata = add_array(subbeddata, -1.*psf_model(x, y), (y_0, x_0))
                print(np.nanmedian(subbeddata[int(y_0-subshape[0]):int(y_0+subshape[0]),int(x_0-subshape[0]):int(x_0+subshape[0])]),
                    np.nanmax(subbeddata[int(y_0-subshape[0]):int(y_0+subshape[0]),int(x_0-subshape[0]):int(x_0+subshape[0])]))
                subshape[0]*=1.5
                norm = ImageNormalize(subbeddata[int(y_0-subshape[0]):int(y_0+subshape[0]),int(x_0-subshape[0]):int(x_0+subshape[0])], interval=ZScaleInterval())
                ax[2].imshow(subbeddata[int(y_0-subshape[0]):int(y_0+subshape[0]),int(x_0-subshape[0]):int(x_0+subshape[0])],norm=norm)
                plt.show()
            #pickle.dump(self.gridded_epsf,open('out_dir/psf_model.pkl','wb'))
            sys.exit()
        _,_,std = calc_bkg(self.image,var_bkg=False,mask = self.image_mask,fill_value=self.maskFill)

        th=10
        from astropy.stats import SigmaClip
        daofind = DAOStarFinder(threshold=th * std, fwhm=self.fwhm,
                    #xycoords=np.array([fitted_star_locs['x_0'],fitted_star_locs['y_0']]).T)
                    xycoords=np.array([pos['x_0'],pos['y_0']]).T)


        daogroup = DAOGroup(5.0 * self.fwhm)

        bkg = MMMBackground()


        fitshape = psf_model.data.shape
        if len(fitshape)==3:
            fitshape = [fitshape[1],fitshape[2]]
        fitshape = [int(fitshape[0]/self.oversample),int(fitshape[1]/self.oversample)]
        if fitshape[0]%2==0:
            fitshape[0]+=1
        if fitshape[1]%2==0:
            fitshape[1]+=1
        print('FITSHAPE:',fitshape,'APRAD:', self.aprad*self.fwhm) #here
        phot = dao_IterativelySubtractedPSFPhotometry(finder=daofind, group_maker=daogroup,
                                              bkg_estimator=None, psf_model=psf_model,
                                              fitter=fitter,
                                              niters=1, fitshape=fitshape, 
                                              aperture_radius=self.aprad*self.fwhm, 
                                              extra_output_cols=('sharpness', 'roundness2'))
        from photutils.utils import calc_total_error
        if sn_location is not None:
            print('Including SN for photometry (%f,%f)'%tuple(sn_location))
            fitted_star_locs.add_row({'x_0':sn_location[0],'y_0':sn_location[1]})

            
        self.outputcat_dao = phot(im_sub_bk,init_guesses=fitted_star_locs, #here
            image_weights=1/np.sqrt(self.image_noise))
        if sn_location is not None:
            self.sn_phot = Table(self.outputcat_dao[np.where(np.logical_and(self.outputcat_dao['x_0']==sn_location[0],
                                                                    self.outputcat_dao['y_0']==sn_location[1]))[0][0]])
            self.sn_phot.write('out_dir/sn_photometry{0}.dat'.format(self.root),format='ascii') 
            # self.sn_phot.write('out_dir/sn_photometry.dat',format='ascii')


        # print(self.outputcat_dao,"self.outputcat_dao")
        self.outputcat_dao.rename_column('x_fit','X')
        self.outputcat_dao.rename_column('y_fit','Y')
        np.savetxt('out_dir/force_xy{0}'.format(self.root),np.array([self.outputcat_dao['X'],self.outputcat_dao['Y']]).T) #here
        #create_pixregionfile(self.outputcat_dao['X']+1,self.outputcat_dao['Y']+1,'out_dir/fit_stars.reg','green')
        self.outputcat_dao.rename_column('flux_fit','flux')
        try:
            self.outputcat_dao.rename_column('flux_unc','fluxerror')
        except:
            self.outputcat_dao['fluxerror'] = 0
        pyfits.PrimaryHDU(phot.get_residual_image(),header=self.hdr).writeto('out_dir/test_residual{0}.fits'.format(self.root),overwrite=True)
        self.outputcat_dao.write('out_dir/test_phot_dao{0}.dat'.format(self.root),format='ascii',overwrite=True)
        xfit,yfit,fluxfit,fluxerr = np.loadtxt('out_dir/test_phot_dao{0}.dat'.format(self.root),unpack=True,dtype={'names':('x','y','flux','fluxerr'),'formats':(float,float,float,'|S15')},usecols=(0,1,9,10),delimiter=' ',skiprows=1)
        fluxerr=fluxerr.astype('str')  
        dummylist=[0]*len(xfit)


        fout = open('out_dir/outputcat_dao{0}.dat'.format(self.root),'w')
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

        
    
    def photutils_aperture(self,starlist,ap_rad=None):
        from photutils.aperture import CircularAperture,CircularAnnulus
        from photutils.aperture import aperture_photometry
        fluxes = []
        fluxerrs = []
        try:
            xvals = np.array(starlist["X"])
            yvals = np.array(starlist['Y'])
        except:
            xvals = np.array(starlist["x"])
            yvals = np.array(starlist['y'])
        if ap_rad is None:
            ap_rad = self.fwhm*2
        for x,y in zip(xvals,yvals):
            
            aperture = CircularAperture(np.array([x,y]), r=ap_rad)
            mask_res = aperture_photometry(self._non_fake_image,aperture)['aperture_sum']

            annulus_aperture = CircularAnnulus(np.array([x,y]), r_in=ap_rad*1.5, r_out=ap_rad*3)
            annulus_masks = annulus_aperture.to_mask(method='center')
            annulus_data = annulus_masks.multiply(self._non_fake_image)
            mask = annulus_masks.data
            annulus_data_1d = annulus_data[mask > 0]
            mean_sigclip, _, _ = sigma_clipped_stats(annulus_data_1d)
            apers = [aperture, annulus_aperture]
            phot_table = aperture_photometry(self._non_fake_image,apers,error=np.sqrt(self.image_noise))
            background = mean_sigclip * aperture.area
            fluxes.append(phot_table['aperture_sum_0']-background)
            fluxerrs.append(phot_table['aperture_sum_err_0'])
        self.aperture_result = Table([xvals,yvals,fluxes,fluxerrs],names=['X','Y','flux','fluxerror'])

    def plant_fake_stars(self,psf_model=None,xylocations=None,out_filename=None,
                                             fluxes=None,nplants=50,star_sep=15,
                                             plant_flux=400000):
        from photutils.aperture import CircularAperture,CircularAnnulus
        from photutils.aperture import aperture_photometry
        from photutils.datasets import make_model_sources_image 
        import multiprocessing
        from copy import copy
        import pyParz
        if psf_model is None:
            try:
                psf_model = self.gridded_epsf
            except:
                print('Need to supply psf model.')
                sys.exit()
        im_background = np.median(self._non_fake_image)
        if xylocations is None:
            xpos = []#np.random.uniform(0,self.image.shape[1],nplants)
            ypos = []#np.random.uniform(0,self.image.shape[0],nplants)
            n_tries = 0
            high_fluxes = []
            while len(xpos)<nplants:
                n_tries+=1
                if n_tries > 1000*nplants:
                    print('Cannot plant enough stars.')
                    sys.exit()
                tempx = np.random.uniform(star_sep,self._non_fake_image.shape[1]-star_sep)
                tempy = np.random.uniform(star_sep,self._non_fake_image.shape[0]-star_sep)
        
                aperture = CircularAperture(np.array([tempx,tempy]), r=star_sep)
                mask_res = aperture_photometry(self.image_mask,aperture)['aperture_sum']

                annulus_aperture = CircularAnnulus([tempx,tempy], r_in=star_sep*1.5, r_out=star_sep*3)
                annulus_masks = annulus_aperture.to_mask(method='center')
                annulus_data = annulus_masks.multiply(self._non_fake_image)
                mask = annulus_masks.data
                annulus_data_1d = annulus_data[mask > 0]
                mean_sigclip, _, _ = sigma_clipped_stats(annulus_data_1d)
                apers = [aperture, annulus_aperture]
                phot_table = aperture_photometry(self._non_fake_image,apers,error=np.sqrt(self.image_noise))
                background = mean_sigclip * aperture.area
                flux_res = phot_table['aperture_sum_0']-background#aperture_photometry(self.image,aperture)['aperture_sum'] - im_background  
                err_res = phot_table['aperture_sum_err_0']
                #err_res = aperture_photometry(self.image_noise,aperture)['aperture_sum']    
                if mask_res > 0 or flux_res/err_res > 3:
                    if flux_res/err_res > 15:
                        high_fluxes.append(flux_res)
                    #print(len(xpos),float(mask_res),float(flux_res),float(err_res),im_background)
                    continue
                #else:
                #    print('success: ',float(mask_res),float(flux_res),float(err_res))
                xpos.append(tempx)
                ypos.append(tempy)
            print('Successfully created %i fake plant locations.'%nplants)
            create_pixregionfile(xpos,ypos,'out_dir/fake_plants.reg','red')
            sources = Table()
            sources['x_0'] = np.array(xpos)
            sources['y_0'] = np.array(ypos)
            sources['id'] = np.arange(1,len(sources)+1,1)
            sources['theta'] = [0]*len(sources)
            sources['flux'] = [plant_flux]*len(sources)#np.random.normal(loc=np.mean(high_fluxes),scale=np.std(high_fluxes))
            #sources['flux'] = np.random.normal(loc=np.mean(high_fluxes),scale=np.std(high_fluxes),size=len(sources))

            #model_ims = []

            ncpus = int(np.min([multiprocessing.cpu_count(),nplants/2]))
            print('splitting to %i cpus'%ncpus)
            inds = np.linspace(0,len(sources),ncpus).astype(int)

            #for i in range(lcden(inds)-1):
            #    model_ims.append(temp_plant([i,inds,self.image.shape,psf_model,sources]))
            model_ims = pyParz.foreach(np.arange(0,len(inds)-1,1),temp_plant,args=[inds,self._non_fake_image.shape,psf_model,sources])
            plant_im = np.zeros(self._non_fake_image.shape)
            for im in model_ims:
                plant_im+=im
            #plant_im = make_model_sources_image(self._non_fake_image.shape,psf_model,sources)
            
            print('Planted %i stars successfully.'%nplants)
            self.fake_sources = sources
            self.fake_sources.write('out_dir/fake_plants.cat',format='ascii')
            
            
            self.planted_fits_image = copy(self._non_fake_fits_image)
            self.planted_fits_image[0].data = self._non_fake_image + plant_im
            self.fake_planted_image = self._non_fake_image + plant_im
            if out_filename is None:
                out_filename = 'out_dir/fake_plant_image.fits'
            self.planted_fits_image.writeto(out_filename,overwrite=True)
            #plt.imshow(model_im)
            #plt.show()
            #sys.exit()

    def get_filter(self,imagefilename):
        self.filter = pyfits.getval(imagefilename,'FILTER')
        # print (self.filter,"immmmm")

    def get_ut(self,imagefilename):
            self.utdate = pyfits.getval(imagefilename,'FITSNAME').split('.')[2]
            print (self.utdate,'utttt')

    def get_objname(self,imagefilename):
        self.snname = pyfits.getval(imagefilename,'OBJECT')

    def get_id(self,imagefilename):
        self.id = pyfits.getval(imagefilename,'FITSNAME').split('.')[3].split('_')[0]
        print (self.id,'id')

    def create_partial_root(self,imagefilename):
        self.partial_root='_'+self.snname+'.'+self.filter+'.'+self.utdate+'.'+self.id
        print (self.partial_root,"proot")

    def create_root(self):
        self.root='_'+self.snname+'.'+self.filter+'.'+self.utdate+'.'+self.id+'_f_'+str(self.num_psfs)+'_'
        print (self.root,"root")

   
    

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
        method = dao.psfRoutine
        from astropy import wcs
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
        self._non_fake_fits_image = fits.open(imagefilename.replace('_fake.fits','.fits'))
        
        (self._non_fake_image,self._non_fake_hdr)=pyfits.getdata(imagefilename.replace('_fake.fits','.fits'),0,header=True)

        print(imagefilename)

        self.fits_image = fits.open(imagefilename)
        (self.image,self.hdr)=pyfits.getdata(imagefilename,0,header=True)
        self.image = self.image.astype(float)#/43.
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
        #sys.exit()

        # self.sexdict = pickle.load(open('newsex_ps.pkl','rb'))
        #self.sexdict = pickle.load(open('../dao_swope/newsex_ps.pkl','rb'))
        #self.sexdict = pickle.load(open('../dao_swope/sex_PS_swope.pkl','rb'))
        #self.sexdict = pickle.load(open('../dao_swope/sex_PS_swope_more.pkl','rb'))
        #self.sexdict = pickle.load(open('../dao_swope/sex_PS_swope_imax.pkl','rb'))
        #self.sexdict = pickle.load(open('../dao_swope2/sex_PS_swope.pkl','rb'))
        # self.sexdict = pickle.load(open('sex_PS_swope_c2.pkl','rb'))
        #self.sexdict = pickle.load(open('../ss_lds/sex_PS_swope_ss_lds749b.i.ut180523.0935.pkl','rb'))


        self.sexdict = pickle.load(open('dao_swope/sex_PS_swope{0}.pkl'.format(self.partial_root),'rb'))

        if psfstarlist is not None:
            goodx,goody = np.loadtxt(psfstarlist,unpack=True)
            good_inds = [x in goodx and y in goody for x,y in zip(self.sexdict['x'],self.sexdict['y'])]
            for key in self.sexdict.keys():
                self.sexdict[key] = np.array(self.sexdict[key])[good_inds]
        #self.sexdict = pickle.load(open('../dao_swope2/sex_PS_swope.pkl','rb'))
        
        # missingx = [x for x,y in zip(self.sexdict1['x'],self.sexdict1['y']) if (x not in self.sexdict['x'] or y not in self.sexdict['y'])]
        # missingy = [y for x,y in zip(self.sexdict1['x'],self.sexdict1['y']) if (x not in self.sexdict['x'] or y not in self.sexdict['y'])]
        # create_pixregionfile(missingx,missingy,'out_dir/missing.reg',color='green')
        # sys.exit()

        self.sexdict = {key:np.array(self.sexdict[key]) for key in self.sexdict.keys()}

        if method == 'epsf':
            self.doPhotutilsePSF(psfstarlist)


            self.phot_dict = self.create_photutils_dict()
            self.writetofile(phot_dict)
            sys.exit()
        elif method == 'dao':

            
            star_flux = 400000
            if '_fake' in imagefilename:
                fake_out_name = imagefilename
            else:
                fake_out_name = imagefilename.replace('.fits','_fake.fits')
            if self.plantFakes:
                self.plant_fake_stars(pickle.load(open('out_dir/psf_model.pkl','rb')),out_filename=fake_out_name,
                                           nplants=100,plant_flux=star_flux)
                if '_fake' not in imagefilename:
                   print('make sure to now run with %s as your image name in test_dao.sh'%imagefilename.replace('.fits','_fake.fits'))
                   sys.exit()
                self.doPhotutilsDAO(Table.read('out_dir/fake_plants.cat',format='ascii'))

                fake_res = Table.read('out_dir/outputcat_dao',format='ascii')
                fake_res['flux_diff'] = fake_res['flux']-star_flux
                print('fake recovery precision (raw flux): %.2f'%(np.std(fake_res['flux_diff'])))
                #### REAL UNCERTAINTY CALC,TODO ####
                #fake_res['fluxerror'] = np.sqrt(np.std(fake_res['flux'])**2 + poiss_unc**2)
                fake_res['flux_diff']/=star_flux
                fake_res['flux_diff']*=100
                fake_res['fluxerror']/=star_flux
                fake_res['fluxerror']*=100
                
                fake_res = fake_res[np.abs(fake_res['flux_diff'])<10] # cut crazy outliers, probably hitting mask
                plt.errorbar(np.arange(0,len(fake_res),1),fake_res['flux'],yerr=fake_res['fluxerror'],fmt='.')
                plt.ylabel('percent flux difference')
                plt.xlabel('star index')
                plt.show()
                sys.exit()
            else:
                if self.sn_x is not None and self.sn_y is not None:
                    sn_location = [self.sn_x,self.sn_y]
                else:
                    print('No SN location found in args, just running star psf fit.')
                    sn_location = None
                self.doPhotutilsDAO(psfstarlist,os.path.basename(imagefilename),sn_location=sn_location)
            print('Running aperture photometry...') #here
            self.photutils_aperture(self.outputcat_dao,ap_rad=self.aprad*self.fwhm)


            #march302023
            # print ("running aperture_photometry ****")
            # from astropy import wcs
        
            # im_sc = wcs.WCS(self.fits_image[0].header).pixel_to_world(self.fits_image[0].header['NAXIS1']/2,self.fits_image[0].header['NAXIS2']/2)
            # viziertable=getPS1cat4table(im_sc.ra.value,im_sc.dec.value)
            # pickle.dump(viziertable,open('out_dir/viziertable{0}.out'.format(self.root),'wb'))
            # filt = self.filter
            # if self.catmagtransform is not None:
            #     self.catmagtransform = Table.read(self.catmagtransform,format='ascii')
            #     ravizierps1=self.catmagtransform['ra']
            #     decvizierps1=self.catmagtransform['dec']
            #     ps1gmag=self.catmagtransform[filt]
            #     ps1gmagerr=self.catmagtransform['d%s'%filt]
            #     viziertable = Table([ravizierps1,decvizierps1,ps1gmag,ps1gmagerr],names=['ra_ps1','dec_ps1','PS1_%s'%filt,'PS1_%s_err'%filt])
            # else:    
            #     ravizierps1=viziertable['ra_ps1']
            #     decvizierps1=viziertable['dec_ps1']
            #     ps1gmag=viziertable['PS1_%s'%filt]
            #     ps1gmagerr=viziertable['PS1_%s_err'%filt]

            # ravizierps1=np.asarray(ravizierps1)
            # decvizierps1=np.asarray(decvizierps1)
        
            # ravizierps1=np.asarray(ravizierps1)
            # decvizierps1=np.asarray(decvizierps1)
            # ps1gmag=np.asarray(ps1gmag)
            # ps1gmagerr=np.asarray(ps1gmagerr)
            # raphot,decphot=frompixtoradec(self.aperture_result['X'],self.aperture_result['Y'],imagefilename)

            # apzpt=calc_zpt(viziertable['PS1_%s'%filt],self.aperture_result['flux'],raphot,decphot,ravizierps1,decvizierps1)
            # print (apzpt)

            # apmag=-2.5*np.log10(self.aperture_result['flux']).flatten()+apzpt

            # boolval,idx,d2d=compare_phot(raphot,decphot,ravizierps1,decvizierps1)

            # apmag = apmag[boolval]
            
            # print ("here apphot")

            # print (apmag)








            if self.dcmpfilename is not None:
                from astropy import wcs
                try:
                    im_sc = wcs.WCS(self.fits_image[0].header).pixel_to_world(self.fits_image[0].header['NAXIS1']/2,self.fits_image[0].header['NAXIS2']/2)
                    #ra = 15.0387611
                    #dec = 30.5255061
                    viziertable=getPS1cat4table(im_sc.ra.value,im_sc.dec.value)
                    pickle.dump(viziertable,open('out_dir/viziertable{0}.out'.format(self.root),'wb'))
                    filt = self.filter
                    if self.catmagtransform is not None:
                        self.catmagtransform = Table.read(self.catmagtransform,format='ascii')
                        ravizierps1=self.catmagtransform['ra']
                        decvizierps1=self.catmagtransform['dec']
                        ps1gmag=self.catmagtransform[filt]
                        ps1gmagerr=self.catmagtransform['d%s'%filt]
                        viziertable = Table([ravizierps1,decvizierps1,ps1gmag,ps1gmagerr],names=['ra_ps1','dec_ps1','PS1_%s'%filt,'PS1_%s_err'%filt])
                    else:    
                        ravizierps1=viziertable['ra_ps1']
                        decvizierps1=viziertable['dec_ps1']
                        ps1gmag=viziertable['PS1_%s'%filt]
                        ps1gmagerr=viziertable['PS1_%s_err'%filt]

                    ravizierps1=np.asarray(ravizierps1)
                    decvizierps1=np.asarray(decvizierps1)
                    create_pixregionfile(ravizierps1,decvizierps1,'out_dir/ps{0}.reg'.format(self.root),color='green',coords='icrs')
                    ps1gmag=np.asarray(ps1gmag,float)
                    ps1gmagerr=np.asarray(ps1gmagerr)
                    ps1gmag[2]
                    x_PSdcmp,y_PSdcmp, extendedness=analyzedcmp(self.dcmpfilename)
                    ra_PSdcmp,dec_PSdcmp=frompixtoradec(x_PSdcmp,y_PSdcmp,imagefilename)
                    extendedness=np.asarray(extendedness,float)
                    ravizierps1=np.asarray(ravizierps1)
                    decvizierps1=np.asarray(decvizierps1)
                    ps1gmag=np.asarray(ps1gmag)
                    ps1gmagerr=np.asarray(ps1gmagerr)
                    plt.figure(figsize=(16,8))

                    residuals_extendedness_plot_PS1(self.root,self.outputcat_dao,imagefilename,ps1gmag,ps1gmagerr,ravizierps1,decvizierps1,extendedness,ra_PSdcmp,dec_PSdcmp,
                        viziertable,ravizierps1,decvizierps1,filt,self.aperture_result,label='Gridded ePSF') #aphere
                    #dave's code
                    #residuals_extendedness_plot_PS1('daopy_42.txt',ps1gmag,ps1gmagerr,ravizierps1,decvizierps1,extendedness,ra_PSdcmp,dec_PSdcmp,
                    #    viziertable,ravizierps1,decvizierps1,label='dao.py')
                    plt.savefig('out_dir/phot_ext_comp{0}.png'.format(self.root),format='png') #here
                    plt.close()
                    plt.figure(figsize=(16,8))


                    # print (self.aperture_result,"self.aperture_result")
                    

                    #ra_ap,dec_ap,mag_ap,magerr_ap=create_data_for_plot_photutils_aperturephot('daopy_42.txt',viziertable,ravizierps1,decvizierps1)
                    #residuals_extendedness_plot(self.outputcat_dao,mag_ap,magerr_ap,ra_PSdcmp,dec_PSdcmp,extendedness,viziertable,ravizierps1,decvizierps1,
                    #    ra_ap,dec_ap,label='Gridded ePSF')

                    #residuals_extendedness_plot('daopy_42.txt',mag_ap,magerr_ap,ra_PSdcmp,dec_PSdcmp,extendedness,viziertable,ravizierps1,decvizierps1,
                    #    ra_ap,dec_ap,label='dao.py')
                    #plt.savefig('phot_ext_comp_ap.png',format='png')
                    #plt.close()
                    #new = Table.read('outputcat_dao',format='ascii')
                except RuntimeError:
                    print('Plotting failed!')

            plt.scatter(-2.5*np.log10(self.outputcat_dao['flux']),self.outputcat_dao['flux']/self.outputcat_dao['fluxerror'])
            plt.savefig('out_dir/snr{0}.png'.format(self.root))


            sys.exit()
        
        print(psfstarlist)
        
        # get PSF
        
        #try:
        self.getPSF(psfstarlist,gain=gain,inputpsf=self.inputpsf,outputpsffilename=self.fittedpsffilename,
            starcat=self.forcelist)
        #except:
        #    print('ignoring star cat...')
        #    self.getPSF(psfstarlist,gain=gain,inputpsf=self.inputpsf,outputpsffilename=self.fittedpsffilename)


        # do the PSF fitting thing
        if not self.inputpsf:
            self.pkfit(self.fittedpsffilename,self.psf,self.gauss,self.psfmag,forcelist=self.forcelist)
        else:
            self.pkfit(self.inputpsf,self.psf,self.gauss,self.psfmag,forcelist=self.forcelist)
        # The first PSF star matters a lot; therefore to generate the PSF,
        # we need a second iteration where we omit the first star, and then
        # we compare chi2 between the two
        if not self.inputpsf:
            # Assigning variables like this may not work
            import copy
            if two_psf_iter:
                self.pkdict_firstiter = copy.deepcopy(self.pkdict)
                #try:
                self.getPSF(psfstarlist,gain=gain,inputpsf=inputpsf,skipfirststar=True,outputpsffilename='%s_2.idlpsf'%imagefilename,
                        starcat=Table.read('out_dir/outputcat_dao',format='ascii'))
                #except:
                #    self.getPSF(psfstarlist,gain=gain,inputpsf=inputpsf,skipfirststar=True,outputpsffilename='%s_2.idlpsf'%imagefilename)
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
        self.outputcat_dao = Table.read(outputcat,format='ascii')
        #print('Running aperture photometry...')
        #self.photutils_aperture(self.outputcat_dao,ap_rad=self.aprad*self.fwhm)
        self.aperture_result = self.outputcat_dao.copy()
        for col in self.outputcat_dao.colnames:
            if col not in ['X','Y','apphot','apphoterr']:
                self.aperture_result.remove_column(col)
        self.aperture_result.rename_column('apphot','flux')
        self.aperture_result.rename_column('apphoterr','fluxerror')
        if self.dcmpfilename is not None:
            from astropy import wcs
            try:
                try:
                    viziertable = pickle.load(open('out_dir/viziertable.out','rb'))
                except:
                    im_sc = wcs.WCS(self.fits_image[0].header).pixel_to_world(self.fits_image[0].header['NAXIS1']/2,self.fits_image[0].header['NAXIS2']/2)
                    viziertable=getPS1cat4table(im_sc.ra.value,im_sc.dec.value)
                    pickle.dump(viziertable,open('out_dir/viziertable.out','wb'))
                filt = self.filter
                if self.catmagtransform is not None:
                    self.catmagtransform = Table.read(self.catmagtransform,format='ascii')
                    ravizierps1=self.catmagtransform['ra']
                    decvizierps1=self.catmagtransform['dec']
                    ps1gmag=self.catmagtransform[filt]
                    ps1gmagerr=self.catmagtransform['d%s'%filt]
                    viziertable = Table([ravizierps1,decvizierps1,ps1gmag,ps1gmagerr],names=['ra_ps1','dec_ps1','PS1_%s'%filt,'PS1_%s_err'%filt])
                else:    
                    ravizierps1=viziertable['ra_ps1']
                    decvizierps1=viziertable['dec_ps1']
                    ps1gmag=viziertable['PS1_%s'%filt]
                    ps1gmagerr=viziertable['PS1_%s_err'%filt]

                ravizierps1=np.asarray(ravizierps1)
                decvizierps1=np.asarray(decvizierps1)
                ps1gmag=np.asarray(ps1gmag,float)
                ps1gmagerr=np.asarray(ps1gmagerr)
                ps1gmag[2]
                x_PSdcmp,y_PSdcmp, extendedness=analyzedcmp(self.dcmpfilename)
                ra_PSdcmp,dec_PSdcmp=frompixtoradec(x_PSdcmp,y_PSdcmp,imagefilename)
                extendedness=np.asarray(extendedness,float)
                ravizierps1=np.asarray(ravizierps1)
                decvizierps1=np.asarray(decvizierps1)
                ps1gmag=np.asarray(ps1gmag)
                ps1gmagerr=np.asarray(ps1gmagerr)
                plt.figure(figsize=(16,8))

                residuals_extendedness_plot_PS1(self.root,self.outputcat_dao,imagefilename,ps1gmag,ps1gmagerr,ravizierps1,decvizierps1,extendedness,ra_PSdcmp,dec_PSdcmp,
                    viziertable,ravizierps1,decvizierps1,filt,self.aperture_result,label='daophot')
                #dave's code
                #residuals_extendedness_plot_PS1('daopy_42.txt',ps1gmag,ps1gmagerr,ravizierps1,decvizierps1,extendedness,ra_PSdcmp,dec_PSdcmp,
                #    viziertable,ravizierps1,decvizierps1,label='dao.py')
                plt.savefig('out_dir/phot_ext_comp_%s_daopy.png'%filt,format='png')
                plt.close()
                plt.figure(figsize=(16,8))
                

                #ra_ap,dec_ap,mag_ap,magerr_ap=create_data_for_plot_photutils_aperturephot('daopy_42.txt',viziertable,ravizierps1,decvizierps1)
                #residuals_extendedness_plot(self.outputcat_dao,mag_ap,magerr_ap,ra_PSdcmp,dec_PSdcmp,extendedness,viziertable,ravizierps1,decvizierps1,
                #    ra_ap,dec_ap,label='Gridded ePSF')

                #residuals_extendedness_plot('daopy_42.txt',mag_ap,magerr_ap,ra_PSdcmp,dec_PSdcmp,extendedness,viziertable,ravizierps1,decvizierps1,
                #    ra_ap,dec_ap,label='dao.py')
                #plt.savefig('phot_ext_comp_ap.png',format='png')
                #plt.close()
                #new = Table.read('outputcat_dao',format='ascii')
            except RuntimeError:
                print('Plotting failed!')

        print('SUCCESS DAOPHOT!!')



if __name__=='__main__':

    usagestring='USAGE: daophot.py image outputcat'

    dao=photclass()
    #dao.getPSF()



    parser = dao.add_options(usage=usagestring)
    options,  args = parser.parse_args()
    print(args)
    print(options)
    if len(args)!=2:
        parser.parse_args(args=['--help'])
        sys.exit(0)

    (imagefilename,outputcat)=args

    dao.psfRoutine = options.psfRoutine
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
        dao.forcelist = options.forcelist
    else:
        dao.forcelist = None
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
    dao.epsfOversample = options.epsfOversample
    dao.epsfFitradius = options.epsfFitradius
    dao.doepsfgrid = options.doepsfgrid
    dao.epsfgridsize = options.epsfgridsize
    dao.nepsfiters = options.nepsfiters
    dao.dcmpfilename = options.dcmpfilename
    dao.plantFakes = options.plantFakes
    dao.maskFill = options.maskFill
    dao.catmagtransform = options.catmagtransform
    dao.doapphotclip = options.doapphotclip
    dao.minStarsPerGridCell = options.minStarsPerGridCell
    dao.sn_x = options.sn_x
    dao.sn_y = options.sn_y


    try:
        dao.gain = pyfits.getval(imagefilename,'GAIN')
    except:
        try:
            dao.gain = pyfits.getval(imagefilename,'EGAIN')
        except RuntimeError:
            raise RuntimeError('Error : GAIN not found in image header!!!')
    try:
        dao.rdnoise = pyfits.getval(imagefilename,'RDNOISE')
    except:
        try:
            dao.rdnoise = pyfits.getval(imagefilename,'ENOISE')
        except:
            raise RuntimeError('Error : RDNOISE not found in image header!!!')

    dao.get_filter(imagefilename)

    dao.get_objname(imagefilename)

    dao.get_ut(imagefilename)

    dao.get_id(imagefilename)


    dao.create_partial_root(imagefilename)


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
