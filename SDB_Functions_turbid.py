# -*- coding: utf-8 -*-
"""
@author: sharrm

Updated: 20Mar2023
"""

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
import rasterio.mask
from rasterio.plot import show
from rasterio.transform import from_bounds
# import richdem as rd
from osgeo import gdal
from scipy.ndimage import uniform_filter
# from scipy.ndimage import generic_filter
# from scipy.ndimage import sobel, prewitt, laplace, gaussian_gradient_magnitude
# from scipy.ndimage import convolve
import sys
# from skimage import feature, filters
# from osgeo import gdal


# %% - check boundary
def check_bounds(rgbnir_dir, shapefile):
    raster = [os.path.join(rgbnir_dir, r) for r in os.listdir(rgbnir_dir) if r.endswith('.tif')]
    raster_bounds = rasterio.open(raster[0]).bounds
    
    for r in raster[1:]:
        if raster_bounds != rasterio.open(r).bounds:
            print(f'Unexpected boundary for {r} in {os.path.dirname(rgbnir_dir)}')
        
    # check if shapefiles point locations are inside the bounds of the raster
    shp_bounds = fiona.open(shapefile, 'r').bounds
    
    # check bounds
    eastings_within = np.logical_and(shp_bounds[0] > raster_bounds[0], # left
                                     shp_bounds[2] < raster_bounds[2]) # right
    northings_within = np.logical_and(shp_bounds[1] > raster_bounds[1], # bottom
                                      shp_bounds[3] < raster_bounds[3]) # top
    
    if np.all([eastings_within, northings_within]):
        print(f'{os.path.basename(shapefile)} within bounds of {os.path.basename(rgbnir_dir)} imagery\n')
        return True
    else:
        return False


# %% - features

def mask_imagery(band, in_shp, masked_raster_name):
    #open bounding shapefile
    with fiona.open(in_shp, 'r') as shapefile:
        shape = [feature['geometry'] for feature in shapefile]

    # read raster, extract spatial information, mask the raster using the input shapefile
    with rasterio.open(band) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        out_meta = src.meta
        
    # writing information
    out_meta.update({"driver": "GTiff",
                      "dtype": 'float32',
                      "height": out_image.shape[1],
                      "width": out_image.shape[2],
                      "nodata": 0,
                      "count": 1,
                      "transform": out_transform})

    # write masked raster to a file
    with rasterio.open(masked_raster_name, "w", **out_meta) as dest:
        dest.write(out_image)

    # close the file
    dest = None
    
    return True, masked_raster_name

def band_comparisons(blue_name, green_name, red_name, red_704_name, red_740_name, nir_name, output_dir):
                     # nir_783_name, 
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read(1)
    green_band = rasterio.open(green_name).read(1)
    red_band = rasterio.open(red_name).read(1)
    red_704_band = rasterio.open(red_704_name).read(1)
    red_740_band = rasterio.open(red_740_name).read(1)
    # nir_783_band = rasterio.open(nir_783_name).read(1)
    # nir_band = rasterio.open(nir_name).read(1)
    
    # psdbr = np.log(blue_band * 1000.0) / np.log(red_band * 1000.0)
    # nd_red_vs_nir = ((red_band + red_704_band + red_740_band) - (nir_783_band + nir_band)) / (red_band + red_704_band + red_740_band + nir_783_band + nir_band)
    # nd_740_704 = (red_740_band - red_704_band) #/ (red_740_band + red_704_band)
    # nd_704_blue = (red_704_band - blue_band) / (red_704_band + blue_band)
    # nd_pSDBr_704 = (psdbr - red_704_band) #/ (psdbr + red_704_band)
    # dd_560_704 = green_band - red_704_band
    # b3_b4 = green_band + red_band
    # b2_b4 = blue_band + red_band
    # add_740_704 = red_704_band + red_740_band
    # DDWI = Green - NIR https://www.mdpi.com/2072-4292/14/3/557
    red_740_green = red_740_band / green_band
    
    comparison_dict = {
        # 'nd_red_vs_nir': nd_red_vs_nir,
        # 'dd_740_704': nd_740_704, 
        # 'nd_704_blue': nd_704_blue, 
        # 'dd_560_704': dd_560_704,
        # 'dd_pSDBr_704': nd_pSDBr_704, # changed from normalized difference
        # 'b3_plus_b4': b3_b4,
        # 'b2_plus_b4': b2_b4,
        # 'add_740_704': add_740_704
        'red_740_green': red_740_green
        }
    
    comparison_names = []
    
    for _arr in comparison_dict.keys():
        outraster_name = os.path.join(output_dir, _arr + '.tif')
        comparison_names.append(outraster_name)
        
        with rasterio.open(outraster_name, "w", **out_meta) as dest:
            dest.write(comparison_dict[_arr],1)
    
    return True, comparison_names, comparison_dict

def rgb_to_cmyk(red_name, green_name, blue_name, output_dir):
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read(1)
    green_band = rasterio.open(green_name).read(1)
    red_band = rasterio.open(red_name).read(1)
    
    CMYK_SCALE = 100

    # sentinel 2 values out of 1: RGB -> CMYK
    ck = 1 - red_band
    mk = 1 - green_band
    yk = 1 - blue_band

    # extract out k [0, 1]
    k = np.minimum.reduce([ck, mk, yk]) # element-wise minimum 
    c = (ck - k) / (1 - k)
    m = (mk - k) / (1 - k)
    y = (yk - k) / (1 - k)

    # rescale to the range [0,CMYK_SCALE]
    cmyk = [c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE]
    
    cyan_out = os.path.join(output_dir, 'cyan.tif')
    magenta_out = os.path.join(output_dir, 'magenta.tif')
    yellow_out = os.path.join(output_dir, 'yellow.tif')
    black_out = os.path.join(output_dir, 'black.tif')
    
    cmyk_name = [cyan_out, magenta_out, yellow_out, black_out]
    
    for cmyk_arr, outraster_name in zip(cmyk, cmyk_name):
        
        with rasterio.open(outraster_name, "w", **out_meta) as dest:
            dest.write(cmyk_arr,1)

    return True, *cmyk_name

# https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/
def rgb_to_hsv(red_name, green_name, blue_name, output_dir):
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read(1)
    green_band = rasterio.open(green_name).read(1)
    red_band = rasterio.open(red_name).read(1)
  
    # h, s, v = hue, saturation, value
    cmax = np.maximum.reduce([red_band, green_band, blue_band])    # maximum of r, g, b
    cmin = np.minimum.reduce([red_band, green_band, blue_band])    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
  
    # if cmax and cmin are equal then h = 0
    # if np.array_equal(cmax, cmin): 
    #     h = 0
      
    # # if cmax equal r then compute h
    # elif np.array_equal(cmax, red_band): 
    #     h = (60 * ((green_band - blue_band) / diff) + 360) % 360
  
    # # if cmax equal g then compute h
    # elif np.array_equal(cmax,green_band):
    #     h = (60 * ((blue_band - red_band) / diff) + 120) % 360
  
    # # if cmax equal b then compute h
    # elif np.array_equal(cmax, blue_band):
    #     h = (60 * ((red_band - green_band) / diff) + 240) % 360
  
    h = np.where(cmax == cmin, 0, 0)
    h = np.where(cmax == red_band, (60 * ((green_band - blue_band) / diff) + 360) % 360, h)
    h = np.where(cmax == green_band, (60 * ((blue_band - red_band) / diff) + 120) % 360, h)
    h = np.where(cmax == blue_band, (60 * ((red_band - green_band) / diff) + 240) % 360, h)
  
    # # if cmax equal zero
    # if not np.any(cmax): # cmax == 0:
    #     s = 0
    # else:
    #     s = (diff / cmax) * 100
  
    # # compute v
    # v = cmax * 100
    
    s = 1 - ((3 / (red_band + blue_band + green_band)) * cmin)
    v = (1/3) * (red_band + blue_band + green_band)
    
    hsv = [h, s, v]
    
    hue_out = os.path.join(output_dir, 'hue.tif')
    saturation_out = os.path.join(output_dir, 'saturation.tif')
    value_out = os.path.join(output_dir, 'value.tif')

    hsv_name = [hue_out, saturation_out, value_out]
    
    for hsv_arr, outraster_name in zip(hsv, hsv_name):
        
        with rasterio.open(outraster_name, "w", **out_meta) as dest:
            dest.write(hsv_arr,1)

    return True, *hsv_name

def odi_1(blue_name, green_name, odi_1_name, output_dir):
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read(1)
    green_band = rasterio.open(green_name).read(1)
    
    odi_1_arr = (green_band * green_band) / blue_band
    
    out_meta.update({"driver": "GTiff",
                     # "dtype": 'float32',
                      "height": blue_band.shape[0],
                      "width": blue_band.shape[1],
                      "nodata": 0,
                      "count": 1})
    
    # output raster filename with path
    outraster_name = os.path.join(output_dir, odi_1_name)
    
    # write masked raster to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(odi_1_arr, 1)
    
    dest = None
    return True, outraster_name

def odi_2(blue_name, green_name, odi_2_name, output_dir):
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read()
    green_band = rasterio.open(green_name).read()
    
    odi_2_arr = (green_band - blue_band) / (green_band + blue_band)
    
    out_meta.update({"driver": "GTiff",
                     # "dtype": 'float32',
                      "height": blue_band.shape[1],
                      "width": blue_band.shape[2],
                      "nodata": 0,
                      "count": 1})
    
    # output raster filename with path
    outraster_name = os.path.join(output_dir, odi_2_name)
    
    # write masked raster to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(odi_2_arr)
    
    dest = None
    return True, outraster_name

def pSDBn (band1, band2, rol_name, output_dir):

    # read first band
    with rasterio.open(band1) as band1_src:
        band1_image = band1_src.read(1)

    # read second band
    with rasterio.open(band2) as band2_src:
        band2_image = band2_src.read(1)
        out_meta = band2_src.meta

    # Stumpf et al algorithm (2003)
    ratioArrayOutput = np.log(band1_image * 1000.0) / np.log(band2_image * 1000.0)
    
    # output raster filename with path
    outraster_name = os.path.join(output_dir, rol_name)
    
    # writing information  
    ratioArrayOutput[np.isnan(ratioArrayOutput)] = 0.0
    ratioArrayOutput[np.isinf(ratioArrayOutput)] = 0.0
    out_meta.update({"dtype": 'float32', "nodata": 0.0})
    
    # write ratio between bands to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(ratioArrayOutput, 1)

    # close the file
    dest = None

    return True, outraster_name

# in general, when writing a file use one to specify number of bands.
def slope(pSDB_str, slope_name, output_dir):
    slope_output = os.path.join(output_dir, slope_name)
    gdal.DEMProcessing(slope_output, pSDB_str, 'slope') # writes directly to file
    return True, slope_output

# from: https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
def window_stdev(pSDB_slope, window, stdev_name, output_dir):
    out_meta = rasterio.open(pSDB_slope).meta
    pSDB_slope = rasterio.open(pSDB_slope).read(1)
    pSDB_slope[pSDB_slope == -9999.] = 0.
    
    # https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    c1 = uniform_filter(pSDB_slope, window, mode='reflect')
    c2 = uniform_filter(pSDB_slope * pSDB_slope, window, mode='reflect')
    std = np.sqrt(c2 - c1*c1)
    std[std == np.nan] = 0.
    
    stdev_output = os.path.join(output_dir, stdev_name)
    
    out_meta.update({"driver": "GTiff",
                      "height": pSDB_slope.shape[0],
                      "width": std.shape[1],
                      "count": 1,
                      "nodata": 0})

    # write stdev slope edges to file
    with rasterio.open(stdev_output, "w", **out_meta) as dest:
        dest.write(std, 1)
        
    dest = None
    
    return True, stdev_output

def roughness(pSDB_str, roughness_name, output_dir):
    roughness_output = os.path.join(output_dir, roughness_name)
    gdal.DEMProcessing(roughness_output, pSDB_str, 'Roughness') # writes directly to file
    return True, roughness_output

def log_chl(coastal, red, out_name, output_dir):
    # read first band
    with rasterio.open(coastal) as coastal_src:
        coastal_image = coastal_src.read(1)

    # read second band
    with rasterio.open(red) as red_src:
        red_image = red_src.read(1)
        out_meta = red_src.meta
        
    log_chl = 1.1578 + -2.5984*np.log10(coastal_image/red_image) + 1.6643*np.log10(coastal_image/red_image)**2 + -0.4915*np.log10(coastal_image/red_image)**3

    chl_a = log_chl # 10**log_chl

    chl_output = os.path.join(output_dir, out_name)

    # write to file
    with rasterio.open(chl_output, "w", **out_meta) as dest:
        dest.write(chl_a, 1)
        
    dest = None
    
    return True, chl_output   

# based on https://publications.gc.ca/collections/collection_2020/mpo-dfo/Fs97-6-3366-eng.pdf

def log_tsm(green_name, red_name, nir_name, output_dir):
    with rasterio.open(green_name) as green_src:
        green_image = green_src.read(1)
    
    with rasterio.open(red_name) as red_src:
        red_image = red_src.read(1)
        out_meta = red_src.meta
        
    with rasterio.open(nir_name) as nir_src:
        nir_image = nir_src.read(1)        
    
    x = (np.log10(green_image) / np.log10(red_image)) + (np.log10(nir_image) / np.log10(red_image)) / np.log10(red_image)
    
    log10_tsm = 40.70320 * x**2 + 57.79984 * x + 21.88387
    
    tsm_output = os.path.join(output_dir, 'log10_tsm.tif')

    # write to file
    with rasterio.open(tsm_output, "w", **out_meta) as dest:
        dest.write(log10_tsm, 1)
        
    dest = None
    
    return True, tsm_output
    
def log_ssd(blue_name, green_name, red_name, output_dir):
    with rasterio.open(green_name) as green_src:
        green_image = green_src.read(1)
    
    with rasterio.open(red_name) as red_src:
        red_image = red_src.read(1)
        out_meta = red_src.meta
        
    with rasterio.open(blue_name) as blue_src:
        blue_image = blue_src.read(1)   
        
        
    log10_ssd = -1.0867 + 0.6417 * np.log10(blue_image) - 1.4111 * np.log10(green_image) - 0.0289 * np.log10(red_image)
    
    ssd_output = os.path.join(output_dir, 'log10_ssd.tif')

    # write to file
    with rasterio.open(ssd_output, "w", **out_meta) as dest:
        dest.write(log10_ssd, 1)
        
    dest = None
    
    return True, ssd_output

def mci(red_name, red_704_name, red_740_name, output_dir):
        # MCI=𝑅𝑟𝑠704−𝑅𝑟𝑠665+(𝑅𝑟𝑠665−𝑅𝑟𝑠740)∗((𝑅𝑟𝑠704−𝑅𝑟𝑠665)/(𝑅𝑟𝑠740−𝑅𝑟𝑠665))
        
        with rasterio.open(red_name) as red_src:
            red_img = red_src.read(1)
        
        with rasterio.open(red_704_name) as red_704_src:
            red704_img = red_704_src.read(1)
            out_meta = red_704_src.meta
            
        with rasterio.open(red_740_name) as red_740_src:
            red740_img = red_740_src.read(1)
            out_meta = red_740_src.meta
        
        # https://www.mdpi.com/2072-4292/12/3/451#B53-remotesensing-12-00451
        # https://www.tandfonline.com/doi/full/10.1080/01431161003639660?casa_token=rL4WPtHtiLcAAAAA%3AhZM7_AZnEQqDT5HPWP1d3Kf24pclaVnL2tzBaToKlC2YIar7qBkGMf5STQH2m5r4GMLaWc2VLldk
        # mci = red704_img - red_img - (red_img - red740_img) * ((red704_img - red_img) / (red740_img - red_img))
        mci = red704_img - red_img - (709 - 681) * (red740_img - red_img) / (754 - 681)
        
        mci_output = os.path.join(output_dir, 'mci.tif')

        # write to file
        with rasterio.open(mci_output, "w", **out_meta) as dest:
            dest.write(mci, 1)
            
        dest = None
        
        return True, mci_output


# %% - composite

# in a multi-band raster, make band dimension first dimension in array, and use no band number when writing
def composite(dir_list, output_composite_name):
    
    bands = []
    
    need_meta_trans = True
    meta_transform = []
    
    for file in dir_list:
        band = rasterio.open(file)
        
        print(f'--Merging {file} to composite')
        
        if need_meta_trans:
            out_meta = band.meta
            out_transform = band.transform  
            meta_transform.append(out_meta)
            meta_transform.append(out_transform)
            need_meta_trans = False
        
        bands.append(band.read(1))        
        band = None
        
    # remember to update count, and shift the np depth axis to the beginning
    # method for creating composite
    comp = np.dstack(bands)
    comp = np.rollaxis(comp, axis=2)
    
    out_meta, out_transform = meta_transform
    
    out_meta.update({"driver": "GTiff",
                      "height": comp.shape[1],
                      "width": comp.shape[2],
                      "count": comp.shape[0],
                      "nodata": 0,
                      # 'compress': 'lzw',
                      "transform": out_transform})
    
    with rasterio.open(output_composite_name, "w", **out_meta) as dest:
        dest.write(comp) # had to specify '1' here for some reason

    dest = None
    return True, comp.shape[0]


# %% - labeled polygon to raster

def polygon_to_raster(polygon, composite_raster, binary_raster):
    # Load polygon
    vector = gpd.read_file(polygon)
    raster = rasterio.open(composite_raster)
    out_transform = raster.transform 
    out_meta = raster.meta
    out_shape = raster.shape
    
    geom = []
    for shape in range(0, len(vector['geometry'])):
        geom.append((vector['geometry'][shape], vector['Truthiness'][shape]))
    
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html
    rasterized = features.rasterize(geom,
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        all_touched=False,
        default_value=1,
        dtype=None)
    
    # Plot raster
    fig, ax = plt.subplots(1, figsize = (10, 10))
    show(rasterized, ax = ax)

    # raster_name = os.path.basename(polygon).replace('shp', 'tif')
    # output_raster = os.path.join(os.path.abspath(os.path.join(os.path.dirname(polygon),
    #                                                           '..', 'Raster')), raster_name)
    
    out_meta.update({"driver": "GTiff",
                      "height": out_shape[0],
                      "width": out_shape[1],
                      "count": 1,
                      "nodata": 0.,
                      "transform": out_transform})
    
    with rasterio.open(binary_raster, 'w', **out_meta) as dest:
        dest.write(rasterized, 1)
        
    dest = None
    
    # print(f"\nWrote {output_raster}")
    print(f"\nWrote {binary_raster}")
    return None


# %% - notes

# def canny(pSDB, canny_name, output_dir, out_meta):
#     canny_edges = feature.canny(pSDB, sigma=1.0)
#     canny_output = os.path.join(output_dir, canny_name)
    
#     # write canny edges to file
#     with rasterio.open(canny_output, "w", **out_meta) as dest:
#         dest.write(canny_edges, 1)
    
#     return True, canny_output

# Zevenbergen, L.W., Thorne, C.R., 1987. Quantitative analysis of land surface topography. Earth surface processes and landforms 12, 47–56.
# def curvature(pSDB, curvature_name, output_dir, out_meta):
#     curvature_output = os.path.join(output_dir, curvature_name)
    
#     # sys.stdout = open(os.devnull, 'w')
#     rda = rd.rdarray(pSDB, no_data=0.0)
#     curve = rd.TerrainAttribute(rda, attrib='curvature')
#     # sys.stdout = sys.__stdout__
    
#     curve = np.array(curve)
#     curve[curve == -9999] = 0.0
    
#     # write curvature to file
#     with rasterio.open(curvature_output, "w", **out_meta) as dest:
#         dest.write(curve, 1)
#     return True, curvature_output

# def tri(pSDB_str, tri_name, output_dir):
#     tri_output = os.path.join(output_dir, tri_name)
#     gdal.DEMProcessing(tri_output, pSDB_str, 'TRI', options=gdal.DEMProcessingOptions(alg='Wilson')) # writes directly to file
#     return True, tri_output

# def tpi(pSDB_str, tpi_name, output_dir):
#     tpi_output = os.path.join(output_dir, tpi_name)
#     gdal.DEMProcessing(tpi_output, pSDB_str, 'TPI') # writes directly to file
#     return True, tpi_output

# def sobel_filt(pSDB, sobel_name, output_dir, out_meta):
    
#     sobel_input = rasterio.open(pSDB).read(1)
    
#     sobel_edges = sobel(sobel_input)
#     sobel_output = os.path.join(output_dir, sobel_name)
    
#     # write sobel edges to file
#     with rasterio.open(sobel_output, "w", **out_meta) as dest:
#         dest.write(sobel_edges, 1)
        
#     sobel_input = None
#     dest = None
#     return True, sobel_output, sobel_edges

# def prewitt_filt(pSDB, prewitt_name, output_dir, out_meta):
#     prewitt_edges = prewitt(pSDB)
#     prewitt_output = os.path.join(output_dir, prewitt_name)
    
#     # write prewitt edges to file
#     with rasterio.open(prewitt_output, "w", **out_meta) as dest:
#         dest.write(prewitt_edges, 1)
#     return True, prewitt_output, prewitt_edges

# def laplace_filt(pSDB, laplace_name, output_dir, out_meta):
    
#     laplace_input = rasterio.open(pSDB).read(1)
    
#     laplace_result = laplace(laplace_input)
#     laplace_output = os.path.join(output_dir, laplace_name)
    
#     # write sobel edges to file
#     with rasterio.open(laplace_output, "w", **out_meta) as dest:
#         dest.write(laplace_result, 1)
        
#     laplace_input = None
#     dest = None
#     return True, laplace_output

# def gaussian_gradient_magnitude_filt(pSDB, gauss_name, output_dir, out_meta):
    
#     gauss_input = rasterio.open(pSDB).read(1)
    
#     gauss_result = gaussian_gradient_magnitude(gauss_input, sigma=1)
#     gauss_output = os.path.join(output_dir, gauss_name)
    
#     # write sobel edges to file
#     with rasterio.open(gauss_output, "w", **out_meta) as dest:
#         dest.write(gauss_result, 1)
        
#     gauss_input = None
#     dest = None
#     return True, gauss_output

# def highpass_filt(pSDB, highpass_name, output_dir, out_meta):
    
#     kernel = np.array([[-1, -1, -1],
#                        [-1,  8, -1],
#                        [-1, -1, -1]])
    
#     highpass_input = rasterio.open(pSDB).read(1)
    
#     highpass_result = convolve(highpass_input, kernel)
#     highpass_output = os.path.join(output_dir, highpass_name)
    
#     # write sobel edges to file
#     with rasterio.open(highpass_output, "w", **out_meta) as dest:
#         dest.write(highpass_result, 1)
        
#     highpass_input = None
#     dest = None
#     return True, highpass_output

# def stdev_slope(pSDB, window_size, stdev_name, output_dir, out_meta):
#     # window_size = window_size
#     rows, cols = pSDB.shape
#     total_rows = np.arange(window_size, rows+window_size, 1)
#     total_columns = np.arange(window_size, cols+window_size, 1)
#     stdev_slope = np.zeros(pSDB.shape)
#     pSDB = np.pad(pSDB, window_size, mode='constant', constant_values=0.0)

#     # v = np.lib.stride_tricks.sliding_window_view(pSDB, (window_size, window_size))
#     # stdev_slope = np.array([np.std(b) for b in v])
    
#     for i in total_rows:
#         for j in total_columns:
#             window = pSDB[i - window_size : i + window_size, j - window_size : j + window_size]
#             stdev = np.std(window)
#             stdev_slope[i - window_size, j - window_size] = stdev

#     stdev_output = os.path.join(output_dir, stdev_name)

#     # write stdev slope edges to file
#     with rasterio.open(stdev_output, "w", **out_meta) as dest:
#         dest.write(stdev_slope, 1)
    
#     return True, stdev_output, stdev_slope

    # x = np.array([np.arange(5), np.arange(5) + 5, np.arange(5) + 10, np.arange(5) + 20])
    # print(f'x: {x}')
    # x.shape
    # v = np.lib.stride_tricks.sliding_window_view(x, (3,3)) # all windows
    # v.shape
    # print(f'v: {v}')
    # for b in v: # windows in v
    #     print(b+1)
    # r = np.array([b+2 for b in v])    # perform some operation
    # print(f'r: {r}')
    
    # stdev_slope = np.zeros([rows + window_size, cols + window_size])
    
    # for i in total_rows:
    #     for j in total_columns:
    #         window = pSDB[i-window_size : i+window_size+1, j-window_size : j+window_size+1]
    #         stdev = np.std(window)
    #         stdev_slope[i,j] = stdev


# def read_band(band):
#     with rasterio.open(masked_rasters['blue']) as src:
#         read_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
#         out_meta = src.meta

#     return read_image, out_meta, out_transform

# def relative_bathymetry(band1, band2):
#     band1, ref, transform = read_band(band1)
#     band2, ref, transform = read_band(band2)

#     # Stumpf algorithm
#     ratiologs = np.log(1000 * band1) / np.log(1000 * band2)

#     return ratiologs, ref, transform

# def write_raster(band1, band2):
#     output_rol = relative_bathymetry(band1, band2)

#     # output raster filename with path
#     outraster_name = os.path.join(os.path.dirname(band1), 'ratio_of_logs.tif')

#     # write ratio between bands to a file
#     with rasterio.open(outraster_name, "w", **out_meta) as dest:
#         dest.write(ratioImage)

#     # close the file
#     dest = None

#     return None

# def stdev(pSDB_str, window, stdev_name, output_dir, out_meta):
#     pSDB_slope = rasterio.open(pSDB_str).read(1)
#     pSDB_slope[pSDB_slope == -9999.] = 0. # be careful of no data values
    
#     print(f'Computing standard deviation of slope within a {window} window...')
    
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html
#     std = generic_filter(pSDB_slope, np.std, size=window)
       
#     stdev_output = os.path.join(output_dir, stdev_name)
    
#     with rasterio.open(stdev_output, "w", **out_meta) as dest:
#         dest.write(std, 1)
    
#     pSDB_slope = None
#     dest = None
    
#     return True, stdev_output