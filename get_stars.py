from astroquery.mast import Catalogs
from astropy.io import fits
import sys
from astropy.coordinates import SkyCoord
import astropy.units as u
import pickle
from astropy.wcs import WCS
import numpy as np

sex = pickle.load(open('sex_output.pkl','rb'))
new_sex = {k:[] for k in sex.keys()}
x,y = sex['x'],sex['y']
w = WCS('F15anh.g.101013_53_1933.sw.fits')
for i in range(len(x)):
    ra_wcs, dec_wcs = w.wcs_pix2world(x[i], y[i], 1)
    sc = SkyCoord(ra_wcs,dec_wcs,unit=u.deg)
    catalog_data = Catalogs.query_region(sc, radius=5*u.arcsec,
                                      catalog="Panstarrs",objType="STAR")

    if len(catalog_data)>0:
        try:
            mags = [np.nanmean(np.array(catalog_data[x])) for x in catalog_data.colnames if x[1:]=='MeanApMag' and np.nanmean(np.array(catalog_data[x]))>0]
        except:
            print([catalog_data[x] for x in catalog_data.colnames if x[1:]=='MeanApMag'])
            sys.exit()

        if len(mags)==0 or np.nanmax(mags)>25 or np.nanmin(mags)<10:
            continue
        

        for key in sex.keys():
            new_sex[key].append(sex[key][i])
print(len(new_sex['x']))
for key in sex.keys():
    new_sex[key] = np.array(new_sex[key])

pickle.dump(new_sex,open('new_sex_output.pkl','wb'))
sys.exit()
#dat = fits.open('F15anh.g.101013_53_1933.sw.fits')[0].header
#ra = dat['FPA.RA']
#dec = dat['FPA.DEC']
sc = SkyCoord(ra,dec,unit=(u.hourangle, u.deg))

catalog_data = Catalogs.query_region(sc, radius=.5,
                                      catalog="Panstarrs",objType="STAR")

catalog_data = catalog_data[catalog_data['']]
print(catalog_data)