#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:50:29 2016

@author: cparr
"""

import glob
from osgeo import gdal, osr
import pandas as pd
import re

index = []

def make_index(path_to_data):
    
    """
    Use filenames to create index for pd DataFrame
    """
    
    file_list = glob.glob(str(path_to_data) + '*.tif')
    
    for f in file_list:
        
        fname = f.split('/')[-1]
        index.append(fname)

    return file_list
        
f_list = make_index('/home/cparr/snow_surfaces/level_1_surfaces/')

surface_metadata = pd.DataFrame(columns=['Year','DOY',
                                         'Projected CRS','Bands','Columns',
                                         'Rows','ULX','No Data Value',
                                        'ULY','Pixel Width',
                                        'Pixel Height','Data Type',
                                        'Format', 'Min.','Max.'], index = index)

f_list_name = zip(f_list,index)

def get_metadata():
    
    """
    Return list of .tif files from the specified folder.
    Extract GDAL metadata to a table
    """
    
    for f in f_list_name:
                
        datafile = gdal.Open(f[0])
        geoinformation = datafile.GetGeoTransform()
        prj=datafile.GetProjection()
        srs=osr.SpatialReference(wkt=prj)
        band = datafile.GetRasterBand(1)
        acqtime = re.findall('\d+', f[1] )
        
#        surface_metadata.ix[f[1]]['Year'] = acqtime[0]
#        surface_metadata.ix[f[1]]['DOY'] = acqtime[1]
        surface_metadata.ix[f[1]]['Bands'] = datafile.RasterCount
        surface_metadata.ix[f[1]]['Columns'] = datafile.RasterXSize
        surface_metadata.ix[f[1]]['Rows'] = datafile.RasterYSize
        surface_metadata.ix[f[1]]['ULX'] = geoinformation[0]
        surface_metadata.ix[f[1]]['Pixel Width'] = geoinformation[1]
        surface_metadata.ix[f[1]]['Pixel Height'] = geoinformation[5]
        surface_metadata.ix[f[1]]['ULY'] = geoinformation[3]
        surface_metadata.ix[f[1]]['Format'] = datafile.GetDriver().LongName
        surface_metadata.ix[f[1]]['Data Type'] = gdal.GetDataTypeName(band.DataType)
        surface_metadata.ix[f[1]]['Projected CRS'] = srs.GetAttrValue('projcs')
        surface_metadata.ix[f[1]]['No Data Value'] = band.GetNoDataValue()
        surface_metadata.ix[f[1]]['Min.'] = round(band.ComputeRasterMinMax()[0])
        surface_metadata.ix[f[1]]['Max.'] = round(band.ComputeRasterMinMax()[1])
                       
                
get_metadata()

surface_metadata.to_csv('/home/cparr/snow_surfaces/level_1_surfaces/metadata.csv')

