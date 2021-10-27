import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import sys, os, os.path, math, re
from astropy.io import fits
from astropy.time import Time
import sys
from operator import itemgetter, attrgetter
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
# from mlxtend.data import iris_data
# from mlxtend.plotting import scatterplotmatrix
# import corner
from scipy import stats 
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table,Column,join
from astropy.stats import sigma_clip
import sys

fieldname=sys.argv[1]

print (fieldname)

#open dcmp/icmp

def analyzedcmp (filename):
    dcmpfile=filename
    exp=fits.getval(filename,'EXPTIME')
    sat=fits.getval(filename,'SATURATE')
    airm=fits.getval(filename,'AIRMASS')
    fwhm=fits.getval(filename,'FWHM')
    flt = fits.getval(filename,'FILTER')
    zp = fits.getval(filename,'ZPTMAG')
    zp_err = fits.getval(filename,'ZPTMUCER')
    mjd = fits.getval(filename,'MJD-OBS')
    ut_date=filename.split(".")[2]
    seeing=fwhm*0.435
    
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
    
    
#     print (exp, airm, fwhm,flt,zp,zp_err)
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

        
        counts=[float(i) for i in counts]
        unc=[float(i) for i in unc]
        magnitude=[float(i) for i in magnitude]
        counts=np.array(counts)
        unc=np.array(unc)
        objecttype = [i for i in objType]
        flag = [i for i in objFlag]
        peakflux=[float(i) for i in peakflux]
        extendedness=[float(i) for i in extendedness]
        chi=[float(i) for i in chi]
        
        print (counts,"counts")
        print (np.std(counts),np.max(counts),np.min(counts))
        
        
        mag_from_flux_zp = -2.5*np.log10(counts) + zp
        division = np.divide(unc,counts)
        photom_error = 2.5*0.434*(division)
        error_mag_from_flux_zp = np.sqrt((photom_error**2 + zp_err**2))
        flux = counts*10**(0.4*(27.5-zp))
        fluxerr = 1./1.086 * error_mag_from_flux_zp * flux
        
        mjd=[mjd]*len(x)
        flt=[flt]*len(x)
        lenmatch=[0]*len(x)
        ut_date=[ut_date]*len(x)
        seeing=[seeing]*len(x)
        exptime=[exp]*len(x)
        airm=[airm]*len(x)
        
        for i in range(len(objecttype)):
            if objecttype[i]=='0x00000001': #only select type1 objects (good dophot point source fit)
                data_quality.append(0)
            else:
                data_quality.append(1)
                
        ra_wcs=[0]*len(x) #creates a list with x elements
        dec_wcs=[0]*len(x)
        new_sw=filename[:-4]+'fits'
        print (new_sw)
        w = WCS(new_sw)
        print (w)
#         print (x[0],y[0])  
        data_quality=np.asarray(data_quality)
        index=np.where(data_quality==0)
        
        for i in range(len(x)):


            ra_wcs[i], dec_wcs[i] = w.wcs_pix2world(x[i], y[i], 1)
            
            
#         print (mag_from_flux_zp,error_mag_from_flux_zp,ra_wcs,dec_wcs)
#         print (len(mag_from_flux_zp),len(mag_from_flux_zp[index]))
        
        ra_wcs=np.asarray(ra_wcs)
        dec_wcs=np.asarray(dec_wcs)
        extendedness=np.asarray(extendedness)
        unswarped_fits=filename[:-7]+'fits.fz'
        print (unswarped_fits,"unswarped_fits")
        
        #wcs_success=fits.getval(unswarped_fits,'WCSNSUC')
        #print (wcs_success)

        w = WCS(unswarped_fits)
        print (w)
        
        x_phys=[0]*len(x) #creates a list with x elements
        y_phys=[0]*len(x)
        
        for i in range(len(x)):
            x_phys[i], y_phys[i] = w.wcs_world2pix(ra_wcs[i], dec_wcs[i], 1)
        x_phys=np.asarray(x_phys)
        y_phys=np.asarray(y_phys)

        print (len(mag_from_flux_zp),len(mag_from_flux_zp[index]))

        print (mag_from_flux_zp,"mags")


        # print (ra_wcs)
        # print (dec_wcs)
        return(mag_from_flux_zp[index],error_mag_from_flux_zp[index],ra_wcs[index],dec_wcs[index],x_phys[index],y_phys[index],extendedness[index],flux[index],fluxerr[index])
    

def readps1cat(file):
    # ra,dec,mag,err=np.loadtxt('/Users/cesar/daophot_test/newdaov2/new/noflags/full/cats/'+file,unpack=True,skiprows=1)
    ra,dec,mag,err=np.loadtxt(file,unpack=True,skiprows=1)
    return (ra,dec,mag,err)

def compare_phot(ra1, dec1, ra2, dec2): #matches two catalogs
    cf = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    catalogf = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)

    max_sep = 1.0 * u.arcsec
    idxf, d2df, d3df = cf.match_to_catalog_3d(catalogf)
    sep_constraintf = d2df < max_sep
    c_matchesf = cf[sep_constraintf]
    catalog_matchesf = catalogf[idxf[sep_constraintf]]
    print ('Number of matches: ', len(c_matchesf), "out of ", len(cf))
    return ([sep_constraintf],[idxf],d2df)


# magsw,magerr,rasw,decsw,x=analyzedcmp('fits_inspection/beforedao/2018hyz.i.ut181124.1129_stch_1.sw.dcmp')

# ps1cat_ra,ps1cat_dec,ps1cat_mag,ps1cat_err=readps1cat('2018hyz.i.PS1.cat')

# magicmp,magerricmp,raicmp,decicmp,x=analyzedcmp('fits_inspection/2018hyz.i.ut181124.1129_stch_1.sw.icmp')


#magsw,magerr,rasw,decsw,xsw,ysw,extsw,fluxsw,fluxerrsw=analyzedcmp('F15anh.g.101013_53_1933.sw.dcmp')

#obj=fieldname.split('.')[0]
#bandfolder=fieldname.split('.')[1]

#ps1cat_ra,ps1cat_dec,ps1cat_mag,ps1cat_err=readps1cat('/data2/crojas/do_dao_comp/rerun/v2/'+obj+'.'+bandfolder+'.PS1.cat')

magicmp,magerricmp,raicmp,decicmp,xicmp,yicmp,exticmp,fluxicmp,fluxerricmp=analyzedcmp('dj_F15anh.g.101013_53_1933.sw.icmp')

#print(magicmp,magerricmp,raicmp,decicmp,xicmp,yicmp,exticmp,fluxicmp,fluxerricmp)

temp_magicmp,temp_magerricmp,temp_raicmp,temp_decicmp,temp_xicmp,temp_yicmp,temp_exticmp,temp_fluxicmp,temp_fluxerricmp=\
    analyzedcmp('temp_F15anh.g.101013_53_1933.sw.icmp')
temp_inds = []
for i in range(len(magicmp)):

    temp_inds.append(np.argmin(np.sqrt((raicmp[i]-temp_raicmp)**2+(decicmp[i]-temp_decicmp)**2)))

#plt.scatter(magicmp,temp_magicmp[temp_inds])
plt.hist(magicmp,density=True,label='dao.py',alpha=.5)
plt.hist(temp_magicmp,density=True,alpha=.5,label='phot_dao')
plt.legend()

plt.show()
sys.exit()
print(magicmp,magerricmp,raicmp,decicmp,xicmp,yicmp,exticmp,fluxicmp,fluxerricmp)
sys.exit()
# magsw,magerr,rasw,decsw,x=analyzedcmp('fits_inspection/beforedao/ss_c26202.i.ut200203.2043_stch_1.sw.dcmp')

# ps1cat_ra,ps1cat_dec,ps1cat_mag,ps1cat_err=readps1cat('ss_c26202.i.PS1.cat')

# magicmp,magerricmp,raicmp,decicmp,x=analyzedcmp('fits_inspection/ss_c26202.i.ut200203.2043_stch_1.sw.icmp')


dcmp_ps1,idx_dcmp_ps1,d2dswps1=compare_phot(rasw,decsw,ps1cat_ra,ps1cat_dec) #compares dcmp with PS catalog (cat)

icmp_ps1,idx_icmp_ps1,d2dicmpps1=compare_phot(raicmp,decicmp,ps1cat_ra,ps1cat_dec) #compares icmp with PS catalog (cat)

dcmp_icmp,idx_dcmp_icmp,d2ddcmpicmp=compare_phot(rasw,decsw,raicmp,decicmp) #compares dcmp with icmp

dcmp_icmp_ps1,idx_dcmp_icmp_ps1,d2ddcmpicmpps1=compare_phot(rasw,decsw,raicmp[icmp_ps1],decicmp[icmp_ps1]) #matches dcmp,icmp and PS




fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6),(ax7,ax8),(ax9,ax10),(ax11,ax12),(ax13,ax14)) = plt.subplots(7, 2, sharex=False, sharey=False,figsize=(10,12),gridspec_kw = {'wspace':0.3, 'hspace':1})

snricmp=fluxicmp/fluxerricmp
# ax1.scatter(magsw[dcmp_ps1],ps1cat_mag[idx_dcmp_ps1][dcmp_ps1])
# # ax1.errorbar(magsw[dcmp_ps1],ps1cat_mag[idx_dcmp_ps1][dcmp_ps1],magerr[dcmp_ps1],ps1cat_err[idx_dcmp_ps1][dcmp_ps1],fmt='.')
# ax1.invert_yaxis()
# ax1.set_xlabel('DCMP mag')
# ax1.set_ylabel('PS1 mag')

res=magsw[dcmp_ps1]-ps1cat_mag[idx_dcmp_ps1][dcmp_ps1]

filtered_data = sigma_clip(res, sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

aa=ax1.scatter(xsw[dcmp_ps1][clipped_mask],ysw[dcmp_ps1][clipped_mask],c=res[clipped_mask],cmap='jet',s=20,vmin=-0.3, vmax=0.3)
fig.colorbar (aa, ax = ax1,label='avg zpt diff with PS1 ')
ax1.set_xlabel('X phys DCMP')
ax1.set_ylabel('Y phys DCMP')




res=magicmp[icmp_ps1]-ps1cat_mag[idx_icmp_ps1][icmp_ps1]
filtered_data = sigma_clip(res, sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

aa=ax2.scatter(xicmp[icmp_ps1][clipped_mask],yicmp[icmp_ps1][clipped_mask],c=res[clipped_mask],cmap='jet',s=20,vmin=-0.3, vmax=0.3)
fig.colorbar (aa, ax = ax2,label='avg zpt diff with PS1 ')
ax2.set_xlabel('X phys ICMP')
ax2.set_ylabel('Y phys ICMP')








res=magsw[dcmp_ps1]-ps1cat_mag[idx_dcmp_ps1][dcmp_ps1]
filtered_data = sigma_clip(res, sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

reserrs=np.sqrt(magerr[dcmp_ps1]**2+ps1cat_err[idx_dcmp_ps1][dcmp_ps1]**2)


# ax2.scatter(magsw[dcmp_ps1][clipped_mask],res[clipped_mask],alpha=0.8)
ax3.errorbar(x=magsw[dcmp_ps1][clipped_mask],y=res[clipped_mask],yerr=reserrs[clipped_mask],alpha=0.5,fmt='.',)
ax3.set_xlabel('DCMP mag')
ax3.set_ylabel('DCMP-PS1 mag')
ax3.set_title('RMS: {0:.3f}'.format(np.std(res[clipped_mask])))


bins=30
count,_,_=stats.binned_statistic(magsw[dcmp_ps1][clipped_mask],res[clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)

bin_med, edges, binnumber=stats.binned_statistic(magsw[dcmp_ps1][clipped_mask],res[clipped_mask], 'median', bins=bins)
ax3.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)
ax3.axhline(y=0,linewidth=3,color='k')




ax5.scatter(extsw[dcmp_ps1][clipped_mask],res[clipped_mask],marker='.')
bins=30
count,_,_=stats.binned_statistic(extsw[dcmp_ps1][clipped_mask],res[clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(extsw[dcmp_ps1][clipped_mask],res[clipped_mask], 'median', bins=bins)
ax5.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)
ax5.set_xlabel('Extendedness')
ax5.set_ylabel('DCMP-PS1 mag')
# ax5.set_title(np.std(res))
ax5.axhline(y=0,linewidth=3,color='k')


########################################################################


res=magicmp[icmp_ps1]-ps1cat_mag[idx_icmp_ps1][icmp_ps1]
notnan=[~np.isnan(res)]
filtered_data = sigma_clip(res[notnan], sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

reserrs=np.sqrt(magerricmp[icmp_ps1]**2+ps1cat_err[idx_icmp_ps1][icmp_ps1]**2)


# ax4.scatter(magicmp[icmp_ps1][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask],alpha=0.8)
ax4.errorbar(x=magicmp[icmp_ps1][~np.isnan(res)][clipped_mask],y=res[~np.isnan(res)][clipped_mask],yerr=reserrs[~np.isnan(res)][clipped_mask],alpha=0.5,fmt='.',)

ax4.set_xlabel('ICMP mag')
ax4.set_ylabel('ICMP-PS1 mag')
ax4.set_title('RMS: {0:.3f}'.format(np.std(res[~np.isnan(res)][clipped_mask])))

bins=30
count,_,_=stats.binned_statistic(magicmp[icmp_ps1][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(magicmp[icmp_ps1][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask], 'median', bins=bins)
ax4.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)
ax4.axhline(y=0,linewidth=3,color='k')




ax12.scatter(magicmp[icmp_ps1][~np.isnan(res)][clipped_mask],snricmp[icmp_ps1][~np.isnan(res)][clipped_mask],marker='.')
ax12.axhline(y=3,linewidth=3,color='k')
ax12.axhline(y=5,linewidth=3,color='k')
ax12.set_xlabel('DAO mag')
ax12.set_ylabel('DAO SNR')
# 12x8.set_ylim(0,15)
bins=30
count,_,_=stats.binned_statistic(magicmp[icmp_ps1][~np.isnan(res)][clipped_mask],snricmp[icmp_ps1][~np.isnan(res)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(magicmp[icmp_ps1][~np.isnan(res)][clipped_mask],snricmp[icmp_ps1][~np.isnan(res)][clipped_mask], 'median', bins=bins)
ax12.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)

ax14.scatter(snricmp[icmp_ps1][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask],marker='.')
ax14.axvline(x=3,linewidth=3,color='k')
ax14.axvline(x=5,linewidth=3,color='k')
ax14.set_xlabel('DAO SNR')
ax14.set_ylabel ('DAO mag - PS1 ')

bins=30
count,_,_=stats.binned_statistic(snricmp[icmp_ps1][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(snricmp[icmp_ps1][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask], 'median', bins=bins)
ax14.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)
ax14.axhline(y=0,c='k')

ax10.scatter(d2dicmpps1[icmp_ps1][~np.isnan(res)][clipped_mask]*3600,res[~np.isnan(res)][clipped_mask],marker='.')
ax10.set_xlabel('ICMP/PS1 offset (arcsec)')
ax10.set_ylabel ('DAO mag - PS1')
ax10.axhline(y=0,c='k')


bins=30
count,_,_=stats.binned_statistic(d2dicmpps1[icmp_ps1][~np.isnan(res)][clipped_mask]*3600,res[~np.isnan(res)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(d2dicmpps1[icmp_ps1][~np.isnan(res)][clipped_mask]*3600,res[~np.isnan(res)][clipped_mask], 'median', bins=bins)
ax10.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)

########################################################################

res=magsw[dcmp_icmp]-magicmp[idx_dcmp_icmp][dcmp_icmp]
notnan=[~np.isnan(res)]
filtered_data = sigma_clip(res[notnan], sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

reserrs=np.sqrt(magerr[dcmp_icmp]**2+magerricmp[idx_dcmp_icmp][dcmp_icmp]**2)


# ax6.scatter(magsw[dcmp_icmp][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask],alpha=0.8)
ax7.errorbar(x=magsw[dcmp_icmp][~np.isnan(res)][clipped_mask],y=res[~np.isnan(res)][clipped_mask],yerr=reserrs[~np.isnan(res)][clipped_mask],alpha=0.5,fmt='.',)

ax7.set_xlabel('DCMP mag')
ax7.set_ylabel('DCMP - ICMP mag')
ax7.set_title('RMS: {0:.3f}'.format(np.std(res[~np.isnan(res)][clipped_mask])))

bins=30
count,_,_=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(res)][clipped_mask],res[~np.isnan(res)][clipped_mask], 'median', bins=bins)
ax7.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)
ax7.axhline(y=0,linewidth=3,color='k')
ax7.set_ylim(-0.4,0.4)

##########################################
reserr=magerr[dcmp_icmp]-magerricmp[idx_dcmp_icmp][dcmp_icmp]

notnan=[~np.isnan(reserr)]
filtered_data = sigma_clip(reserr[notnan], sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask


ax8.errorbar(x=magsw[dcmp_icmp][~np.isnan(reserr)][clipped_mask],y=reserr[~np.isnan(reserr)][clipped_mask],alpha=0.5,fmt='.')
ax8.set_xlabel('DCMP mag')
ax8.set_ylabel('DCMP magerr - \n ICMP magerr')
# ax8.set_title('RMS: {0:.3f}'.format(np.std(reserr[~np.isnan(reserr)][clipped_mask])))

bins=30
count,_,_=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(reserr)][clipped_mask],reserr[~np.isnan(reserr)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(reserr)][clipped_mask],reserr[~np.isnan(reserr)][clipped_mask], 'median', bins=bins)
ax8.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=3)
ax8.axhline(y=0,linewidth=3,color='k')
# ax8.set_ylim(-0.4,0.4)


################################
res=magicmp[idx_dcmp_icmp_ps1][dcmp_icmp_ps1]-ps1cat_mag[idx_icmp_ps1][icmp_ps1][idx_dcmp_icmp_ps1][dcmp_icmp_ps1]
filtered_data = sigma_clip(res, sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

ax6.scatter(extsw[dcmp_icmp_ps1][clipped_mask],res[clipped_mask],marker='.')

bins=30
count,_,_=stats.binned_statistic(extsw[dcmp_icmp_ps1][clipped_mask],res[clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(extsw[dcmp_icmp_ps1][clipped_mask],res[clipped_mask], 'median', bins=bins)
ax6.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)
ax6.set_xlabel('Extendedness')
ax6.set_ylabel('ICMP-PS1 mag')
ax6.axhline(y=0,linewidth=3,color='k')
# ax7.set_title(np.std(res))

########################################################################
#DOPHOT ********************

# ax3.errorbar(x=magsw[dcmp_ps1][clipped_mask],y=res[clipped_mask],yerr=reserrs[clipped_mask],alpha=0.5,fmt='.',)
snrdcmp=fluxsw/fluxerrsw


res=magsw[dcmp_ps1]-ps1cat_mag[idx_dcmp_ps1][dcmp_ps1]
filtered_data = sigma_clip(res, sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask



ax11.scatter(magsw[dcmp_ps1][clipped_mask],snrdcmp[dcmp_ps1][clipped_mask],marker='.')
ax11.axhline(y=3,linewidth=3,color='k')
ax11.axhline(y=5,linewidth=3,color='k')
ax11.set_xlabel('DOPHOT mag')
ax11.set_ylabel('DOPHOT SNR')
# 12x8.set_ylim(0,15)
bins=30
count,_,_=stats.binned_statistic(magsw[dcmp_ps1][clipped_mask],snrdcmp[dcmp_ps1][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(magsw[dcmp_ps1][clipped_mask],snrdcmp[dcmp_ps1][clipped_mask], 'median', bins=bins)
ax11.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)

ax13.scatter(snrdcmp[dcmp_ps1][clipped_mask],res[clipped_mask],marker='.')
ax13.axvline(x=3,linewidth=3,color='k')
ax13.axvline(x=5,linewidth=3,color='k')
ax13.set_xlabel('DOPHOT SNR')
ax13.set_ylabel ('DOPHOT mag - PS1 ')

bins=30
count,_,_=stats.binned_statistic(snrdcmp[dcmp_ps1][clipped_mask],res[clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(snrdcmp[dcmp_ps1][clipped_mask],res[clipped_mask], 'median', bins=bins)
ax13.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)
ax13.axhline(y=0,c='k')

ax9.scatter(d2dswps1[dcmp_ps1][clipped_mask]*3600,res[clipped_mask],marker='.')
ax9.set_xlabel('DCMP/PS1 offset (arcsec)')
ax9.set_ylabel ('DOPHOT mag - PS1')
ax9.axhline(y=0,c='k')


bins=30
count,_,_=stats.binned_statistic(d2dswps1[dcmp_ps1][clipped_mask]*3600,res[clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(d2dswps1[dcmp_ps1][clipped_mask]*3600,res[clipped_mask], 'median', bins=bins)
ax9.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=6)

########################################



# ax8.axis('off')


plt.minorticks_on()
# fig = plt.gcf()



# plt.show()
fig.suptitle(fieldname, fontsize=14)

plt.savefig('/data2/crojas/do_dao_comp/rerun/v2/{0}.png'.format(fieldname),dpi=200,bbox_inches='tight')
plt.close()



res=magicmp[icmp_ps1]-ps1cat_mag[idx_icmp_ps1][icmp_ps1]
notnan=[~np.isnan(res)]
filtered_data = sigma_clip(res[notnan], sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask


raicmpps1_used=raicmp[icmp_ps1][~np.isnan(res)][clipped_mask]
decicmpps1_used=decicmp[icmp_ps1][~np.isnan(res)][clipped_mask]


with open('/data2/crojas/do_dao_comp/rerun/v2/passcuts_{0}.reg'.format(fieldname), 'w') as f:
    f.write('global color=green dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
    f.write('fk5 \n')
    for star in range(len(raicmpps1_used)):
        xval = raicmpps1_used[star]
        yval = decicmpps1_used[star]
        f.write('circle({ra},{dec},{radius}")\n'.format(ra=xval, dec=yval,radius=12))
f.close()  

print (len(raicmpps1_used))


clipped_icmp_ps1res=res[~np.isnan(res)][clipped_mask]

# index=np.where((clipped_icmp_ps1res>0.2 ) | (clipped_icmp_ps1res < 0.2))
index=np.where(clipped_icmp_ps1res>0.2)
highres_ra=raicmp[icmp_ps1][index]
highres_dec=decicmp[icmp_ps1][index]

with open('/data2/crojas/do_dao_comp/rerun/v2/high_residuals_{0}.reg'.format(fieldname), 'w') as f:
    f.write('global color=red dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
    f.write('fk5 \n')
    for star in range(len(highres_ra)):
        xval = highres_ra[star]
        yval = highres_dec[star]
        f.write('circle({ra},{dec},{radius}")\n'.format(ra=xval, dec=yval,radius=12))
f.close() 


index=np.where(clipped_icmp_ps1res<-0.2)
lowres_ra=raicmp[icmp_ps1][index]
lowres_dec=decicmp[icmp_ps1][index]
with open('/data2/crojas/do_dao_comp/rerun/v2/low_residuals_{0}.reg'.format(fieldname), 'w') as f:
    f.write('global color=cyan dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
    f.write('fk5 \n')
    for star in range(len(lowres_ra)):
        xval = lowres_ra[star]
        yval = lowres_dec[star]
        f.write('circle({ra},{dec},{radius}")\n'.format(ra=xval, dec=yval,radius=12))
f.close() 

print (len(clipped_icmp_ps1res),"total after cuts")


print (len(highres_ra),"res above 0.2")

print (len(lowres_ra),"res below 0.2")


######new plot

plt.figure()

uncertainty_ratio=magerr[dcmp_icmp]/magerricmp[idx_dcmp_icmp][dcmp_icmp]

notnan=[~np.isnan(uncertainty_ratio)]
filtered_data = sigma_clip(uncertainty_ratio[notnan], sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

print (len(uncertainty_ratio))

print (len(magsw[dcmp_icmp]))

print (len(magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask]))

plt.errorbar(x=magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask],y=uncertainty_ratio[~np.isnan(uncertainty_ratio)][clipped_mask],alpha=0.5,fmt='.')

plt.xlabel('DOPHOT mag')
plt.ylabel ('DOPHOT magerr/DAOPHOT magerr')
plt.axhline(y=1,c='k')

bins=30
count,_,_=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask],uncertainty_ratio[~np.isnan(uncertainty_ratio)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask],uncertainty_ratio[~np.isnan(uncertainty_ratio)][clipped_mask], 'median', bins=bins)
plt.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=3)


plt.savefig('/data2/crojas/do_dao_comp/rerun/v2/{0}_uncertainty_ratio.png'.format(fieldname),dpi=200,bbox_inches='tight')

plt.close()

# ######new plot

plt.figure()

uncertainty_ratio=fluxerrsw[dcmp_icmp]/fluxerricmp[idx_dcmp_icmp][dcmp_icmp]

notnan=[~np.isnan(uncertainty_ratio)]
filtered_data = sigma_clip(uncertainty_ratio[notnan], sigma=3, maxiters=5,masked=True)
clipped_mask=~filtered_data.mask

print (len(uncertainty_ratio))

print (len(magsw[dcmp_icmp]))

print (len(magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask]))

plt.errorbar(x=magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask],y=uncertainty_ratio[~np.isnan(uncertainty_ratio)][clipped_mask],alpha=0.5,fmt='.')

plt.xlabel('DOPHOT mag')
plt.ylabel ('DOPHOT fluxerr/ICMP fluxerr')
plt.axhline(y=1,c='k')

bins=30
count,_,_=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask],uncertainty_ratio[~np.isnan(uncertainty_ratio)][clipped_mask], 'count', bins=bins)
count_bool=[]
for i in range(len(count)):
        if count[i]>=3:
            count_bool.append(True)
        else:
            count_bool.append(False)
bin_med, edges, binnumber=stats.binned_statistic(magsw[dcmp_icmp][~np.isnan(uncertainty_ratio)][clipped_mask],uncertainty_ratio[~np.isnan(uncertainty_ratio)][clipped_mask], 'median', bins=bins)
plt.errorbar(((edges[1:]+edges[:-1])/2.)[count_bool],bin_med[count_bool],0,fmt ='r-o',ms=3)


plt.savefig('/data2/crojas/do_dao_comp/rerun/v2/{0}_uncertainty_ratio_flux.png'.format(fieldname),dpi=200,bbox_inches='tight')


