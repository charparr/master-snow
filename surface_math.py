#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:07:54 2016

@author: cparr
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cv2

#use rasterio to import and read into np arrays
#then subtract summer surface from each winter surface
#and then inspect for nans or negative values

clpx_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/clpx_2012_157_bare_earth_demv2.tif')
clpx_bare = clpx_src.read(1)
clpx_bare_meta = clpx_src.meta
clpx_bare = np.ma.masked_array(clpx_bare, clpx_bare == clpx_src.nodatavals)
clpx_bare = cv2.blur(clpx_bare, (5,5))

clpx_snow_src12 = rasterio.open('/home/cparr/surfaces/level_1_surfaces/clpx_2012_106_snow_on_dem.tif')
clpx_snow12 = clpx_snow_src12.read(1)
clpx_snow12_meta = clpx_snow_src12.meta
clpx_snow12 = np.ma.masked_array(clpx_snow12, clpx_snow12 == clpx_snow_src12.nodatavals)
clpx_snow12_trim_to_bare = clpx_snow12[::,:-1]

clpx_depth_12 = clpx_snow12_trim_to_bare - clpx_bare + 2
plt.imshow(clpx_depth_12, vmin=0, vmax = 5)
plt.title('CLPX 2012 Snow Depth w/ +2 m offset')
plt.colorbar()

clpx_snow_src13 = rasterio.open('/home/cparr/surfaces/level_1_surfaces/clpx_2013_102_snow_on_dem.tif')
clpx_snow13 = clpx_snow_src13.read(1)
clpx_snow13_meta = clpx_snow_src13.meta
clpx_snow13 = np.ma.masked_array(clpx_snow13, clpx_snow13 == clpx_snow_src13.nodatavals)
clpx_snow13_trim_to_bare = clpx_snow13[:-1,:-1]

## 2013 Depth +2 offset

clpx_depth_13_2 = clpx_snow13_trim_to_bare - clpx_bare + 2
g = plt.imshow(clpx_depth_13_2, vmin=0, vmax = 5)
plt.colorbar()
plt.title('CLPX 2013 Snow Depth w/ +2 m offset', fontsize = 14)
cs = plt.contour(clpx_bare, levels = range(650,1000,25), colors = 'black')
plt.clabel(cs, fmt = '%.0f', inline = False)

## 2013 Depth No Offset

clpx_depth_13_0 = clpx_snow13_trim_to_bare - clpx_bare + 0
plt.imshow(clpx_depth_13_0, vmin=-3, vmax = 3)
plt.colorbar()
plt.title('CLPX 2013 Snow Depth w/ no offset', fontsize = 18)
cs = plt.contour(clpx_bare, levels = range(650,1000,10), colors = 'black')
plt.clabel(cs, fmt = '%.0f', inline = False)
plt.axhline(y=3150, linewidth=2, color = 'k')

## Bare Earth

plt.imshow(clpx_bare, cmap = 'terrain')
plt.colorbar()
plt.title('CLPX Bare Earth [m]')
cs = plt.contour(clpx_bare, levels = range(650,1000,25), colors = 'black')
plt.clabel(cs, fmt = '%.0f', inline = True)


###
plt.plot(bare_t)
plt.plot(snow13_t)
plt.legend(['bare earth','2013 snow surface'], loc = 'upper left')

