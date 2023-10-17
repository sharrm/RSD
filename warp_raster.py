# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:29:34 2023

@author: sharrm

https://www.ncei.noaa.gov/products/etopo-global-relief-model

Used to ensure exact overlap between two raster data sets in spatial resolution,
projection, and spatial extent.

"""

import affine
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT


# %% - globals

# interpolation method
resampling_method = Resampling.bilinear


# %% - based on the 'match_tif' extents, will mask, resample, and reproject the 'tif' so the two spatially overlap
# based on https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html
# should not need to modify anything below

def resample(tif, match_tif, out_etopo):
                                                        
    with rasterio.open(match_tif, 'r') as dest:
        dst_crs = dest.crs
        dst_bounds = dest.bounds
        dst_height = dest.height
        dst_width = dest.width
    
    print(f'Using {match_tif} extents...')
    
    # Output image transform based on input match tif
    left, bottom, right, top = dst_bounds
    xres = (right - left) / dst_width
    yres = (top - bottom) / dst_height
    dst_transform = affine.Affine(xres, 0.0, left,
                                  0.0, -yres, top)
    
    vrt_options = {
        'resampling': resampling_method, 
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
        'nodata': 0
    }
    
    print(f'Warping {tif} to overlap...')
    
    with rasterio.open(tif) as src:
    
        with WarpedVRT(src, **vrt_options) as vrt:
    
            # At this point 'vrt' is a full dataset with dimensions,
            # CRS, and spatial extent matching 'vrt_options'.
    
            # Read all data into memory.
            data = vrt.read()
    
            # Process the dataset in chunks.  Likely not very efficient.
            for _, window in vrt.block_windows():
                data = vrt.read(window=window)
    
            # Dump the aligned data into a new file.  A VRT representing
            # this transformation can also be produced by switching
            # to the VRT driver.
            rio_shutil.copy(vrt, out_etopo, driver='GTiff')
    
    return out_etopo


if __name__ == '__main__':
        tif = r"P:\_RSD\Data\ETOPO\All_ETOPO2022_15s_IceSurf_EXT_01_m100_LZW.tif"
        # tif = r"P:\_RSD\Data\VIIRS\VRSUCW_2023100_DAILY_SNPP_KD490_CD_750M.tif"
        # match_tif = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
        # match_tif = r"C:\_Turbidity\Imagery\_turbidTestingETOPO\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_492.tif"
        # match_tif = r"P:\_RSD\Data\Imagery\_turbidTraining_rhos\Chesapeake_20230316\S2A_MSI_2023_03_16_16_02_50_T18SUG_L2R_rhos_492.tif"
        # match_tif = r"C:\_Turbidity\Imagery\_turbidTestingVIIRS\Chesapeake_20230410\S2B_MSI_2023_04_10_16_02_54_T18SUG_L2R_rhos_492.tif"
        # match_tif = r"P:\_RSD\Data\Imagery\_turbidTraining_rhos\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_492.tif"
        match_tif= r"P:\_RSD\Data\ETOPO\Imagery\CapeCod\S2A_MSI_2023_05_29_15_41_33_T19TCG_L2R_rhos_492.tif"
        # out_etopo = os.path.join(r'P:\_RSD\Data\VIIRS', 'VIIRS_Lookout_bilinear.tif')
        out_etopo = os.path.join(r'P:\_RSD\Data\ETOPO', 'ETOPO_CapeCod_bilinear.tif')
        
        outfile = resample(tif, match_tif, out_etopo)
        print(f'\nOutput: {outfile}')