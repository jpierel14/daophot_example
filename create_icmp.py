from astropy.table import Table
from astropy.io import fits
import numpy as np
import os,sys

# icmp_header_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NPLYTERM', 'HISTORY', 'PSLIB_V', 'HISTORY', 
# 					'HISTORY', 'MODULE_V', 'HISTORY', 'HISTORY', 'PHOT_V', 'HISTORY', 'HISTORY', 'STATS_V', 'HISTORY', 
# 					'HISTORY', 'WARP_V', 'HISTORY', 'HISTORY', 'MD5_SkyChip_SkyCell_0', 'SRC_0000', 'SRC_0001', 'SRC_0002', 
# 					'SRC_0003', 'SRC_0004', 'SRC_0005', 'SRC_0006', 'SEC_0000', 'SEC_0001', 'SEC_0002', 'SEC_0003', 'SEC_0004', 
# 					'SEC_0005', 'SEC_0006', 'MPX_0000', 'MPY_0000', 'MPX_0001', 'MPY_0001', 'MPX_0002', 'MPY_0002', 'MPX_0003', 
# 					'MPY_0003', 'MPX_0004', 'MPY_0004', 'MPX_0005', 'MPY_0005', 'MPX_0006', 'MPY_0006', 'CTYPE1', 'CTYPE2', 
# 					'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'PC001001', 'PC001002', 'PC002001', 'PC002002', 
# 					'PSCAMERA', 'PSFORMAT', 'FPA.TELESCOPE', 'FPA.INSTRUMENT', 'FPA.DETECTOR', 'FPA.COMMENT', 'FPA.OBS.MODE', 
# 					'FPA.OBS.GROUP', 'FPA.FOCUS', 'AIRMASS', 'FPA.FILTERID', 'FPA.FILTER', 'FPA.POSANGLE', 'FPA.ROTANGLE', 
# 					'FPA.RADECSYS', 'FPA.RA', 'FPA.DEC', 'FPA.LONGITUDE', 'FPA.LATITUDE', 'FPA.ELEVATION', 'FPA.OBSTYPE', 'FPA.OBJECT', 
# 					'FPA.ALT', 'FPA.AZ', 'FPA.TEMP', 'FPA.M1X', 'FPA.M1Y', 'FPA.M1Z', 'FPA.M1TIP', 'FPA.M1TILT', 'FPA.M2X', 'FPA.M2Y', 
# 					'FPA.M2Z', 'FPA.M2TIP', 'FPA.M2TILT', 'FPA.ENV.TEMP', 'FPA.ENV.HUMID', 'FPA.ENV.WIND', 'FPA.ENV.DIR', 'FPA.TELTEMP.M1', 
# 					'FPA.TELTEMP.M1CELL', 'FPA.TELTEMP.M2', 'FPA.TELTEMP.SPIDER', 'FPA.TELTEMP.TRUSS', 'FPA.TELTEMP.EXTRA', 'FPA.PON.TIME', 
# 					'FPA.BURNTOOL.APPLIED', 'FPA.ZP', 'CHIP.XSIZE', 'CHIP.YSIZE', 'CHIP.TEMP', 'CHIP.TEMPERATURE', 'CHIP.ID', 'CHIP.SEEING', 
# 					'CHIP.VIDEOCELL', 'CELL.GAIN', 'CELL.READNOISE', 'CELL.SATURATION', 'CELL.BAD', 'EXPTIME', 'CELL.DARKTIME', 'CELL.XBIN', 
# 					'CELL.YBIN', 'MJD-OBS', 'CELL.XSIZE', 'CELL.YSIZE', 'CELL.XWINDOW', 'CELL.YWINDOW', 'CELL.TRIMSEC', 'CELL.BIASSEC', 
# 					'OBSTYPE', 'GAIN', 'RDNOISE', 'FILTER', 'SATURATE', 'CHECKSUM', 'DATASUM', 'CPSCALE', 'PEDESTAL', 'SKYADU', 'SKYSIG', 
# 					'BADPVAL', 'NOISEIM', 'MASKIM', 'SOFTNAME', 'PIXSCALE', 'SW_PLTSC', 'PHOTCODE', 'SUBDIR', 'FITSNAME', 'MASKN', 'MASKP', 
# 					'RA', 'DEC', 'SW_RA', 'SW_DEC', 'WCSCAT', 'WCSEXE', 'WCSDB', 'WCSFLAG', 'WCSNSUC', 'PASTRO', 'APRECPIX', 'POLYFIT', 
# 					'WCSRASH0', 'WCSDESH0', 'WCSRADEC', 'EWCSXAS', 'EWCSYAS', 'WCSRAD', 'NASTRO', 'WCSRASH', 'WCSDECSH', 'CD1_1', 'CD1_2', 
# 					'CD2_1', 'CD2_2', 'EWCSXPIX', 'EWCSYPIX', 'ROTGASTR', 'WCSFULL', 'BSCALE', 'BZERO', 'CMPAPERT', 'CMPNPIX', 'PHOTTYPE', 
# 					'CPIX1X', 'CPIX1Y', 'FWHM', 'NDETECT', 'MAGZERO', 'ZPTMAG', 'ZPTMUCER', 'ZPTNSTAR', 'ZPTSOURC', 'ZPTSTDEV', 'ZPTNCLIP', 
# 					'ZPTRX2', 'DPSIGX', 'DPSIGXY', 'DPSIGY', 'DPFWHM1', 'DPFWHM2', 'DPTILT', 'DPTYPE1', 'DPTYPE2', 'DPTYPE3', 'DPTYPE5', 
# 					'DPTYPE7', 'DPTYPE9', 'PIXCHK', 'NCOLTBL', 'COLTBL1', 'COLTBL2', 'COLTBL3', 'COLTBL4', 'COLTBL5', 'COLTBL6', 'COLTBL7', 
# 					'COLTBL8', 'COLTBL9', 'COLTBL10', 'COLTBL11', 'COLTBL12', 'COLTBL13', 'COLTBL14', 'COLTBL15', 'COLTBL16', 'COLTBL17', 
# 					'COLTBL18', 'COLTBL19', 'COLTBL20', 'COLTBL21', 'COLTBL22', 'COLTBL23', 'COLTBL24', 'PSFTRIM', 'ZPDEG', 'EVNTDIST', 
# 					'ZPRACEN', 'ZPDECCEN', 'ZPTFILE', 'FORCEZPT', 'FZPTMAG']


def convert_to_icmp(phot_output):
	#x,y,magnitude,counts,unc,objType,peakflux,sky,chi,objClass,extendedness,objFlag,nMask
	#filecols = (0,1,2,4,5,6,7,11,12,13,18,19,21)
	print('opening')
	with open('F15anh.g.101013_53_1933.sw.icmp','r') as f:
		dat = f.read()
	split = dat.split('\n')
	with open('temp_F15anh.g.101013_53_1933.sw.icmp','w') as f:
		f.write(split[0]+'\n')
		for i in range(0,len(phot_output)):
			row = [phot_output[i][0],phot_output[i][1],1,1,phot_output[i][2],
					phot_output[i][3],'0x00000001',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			row_str = [str(x) for x in row]
			if '--' in row_str:
				continue
			f.write("  ".join(row_str)+'\n')



convert_to_icmp(Table.read('outputcat_dao',format='ascii'))