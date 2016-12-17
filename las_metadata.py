#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:45:44 2016

@author: cparr
"""

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
import laspy

index = []

def make_index(path_to_data):
    
    """
    Use filenames to create index for pd DataFrame
    """
    
    file_list = glob.glob(str(path_to_data) + '*.las')
    
    for f in file_list:
        
        fname = f.split('/')[-1]
        index.append(fname)

    return file_list
        
f_list = make_index('/home/cparr/level_0_surfaces/')

las_metadata = pd.DataFrame(columns=['Date','Projected CRS','Bands','Columns',
                                     'ULX','No Data Value','Total Points',
                                     'ULY','Format', 'Min. XYZ','Max. XYZ'], index = index)
#
f_list_name = zip(f_list,index)

def get_metadata():
    
    """
    Return list of .tif files from the specified folder.
    Extract LAS header metadata to a table
    """
    
    for f in f_list_name:
        
        las = laspy.file.File(f[0], mode = 'r')
        hdr = las.header
        las_metadata.ix[f[1]]['Min. XYZ'] = hdr.min
        las_metadata.ix[f[1]]['Max. XYZ'] = hdr.max
        las_metadata.ix[f[1]]['Date'] = hdr.date
        las_metadata.ix[f[1]]['Total Points'] = hdr.point_records_count
        
        headerformat = las.header.header_format
        
        #for spec in headerformat:
            #print(spec.name)
        
        
#        datafile = gdal.Open(f[0])
#        geoinformation = datafile.GetGeoTransform()
#        prj=datafile.GetProjection()
#        srs=osr.SpatialReference(wkt=prj)
#        band = datafile.GetRasterBand(1)
#        acqtime = re.findall('\d+', f[1] )
#        
#        surface_metadata.ix[f[1]]['Year'] = acqtime[0]
#        surface_metadata.ix[f[1]]['DOY'] = acqtime[1]
#        surface_metadata.ix[f[1]]['Bands'] = datafile.RasterCount
#        surface_metadata.ix[f[1]]['Columns'] = datafile.RasterXSize
#        surface_metadata.ix[f[1]]['Rows'] = datafile.RasterYSize
#        surface_metadata.ix[f[1]]['ULX'] = geoinformation[0]
#        surface_metadata.ix[f[1]]['Pixel Width'] = geoinformation[1]
#        surface_metadata.ix[f[1]]['Pixel Height'] = geoinformation[5]
#        surface_metadata.ix[f[1]]['ULY'] = geoinformation[3]
#        surface_metadata.ix[f[1]]['Format'] = datafile.GetDriver().LongName
#        surface_metadata.ix[f[1]]['Data Type'] = gdal.GetDataTypeName(band.DataType)
#        surface_metadata.ix[f[1]]['Projected CRS'] = srs.GetAttrValue('projcs')
#        surface_metadata.ix[f[1]]['No Data Value'] = band.GetNoDataValue()

#                       
#                
get_metadata()
#
#surface_metadata.to_csv('/home/cparr/level_0_surfaces/metadata.csv')

