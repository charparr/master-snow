"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
Created on Tue Dec  6 09:54:51 2016

@author: cparr
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pylab

hv_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/hv/hv_2012_107_snow_on_dem.tif')
hv_2012 = hv_src.read(1)
hv_2012 = np.ma.masked_array(hv_2012, hv_2012 == hv_src.nodatavals)

hv_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/hv/hv_2013_103_snow_on_dem.tif')
hv_2013 = hv_src.read(1)
hv_2013 = np.ma.masked_array(hv_2013, hv_2013 == hv_src.nodatavals)

hv_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/hv/hv_2015_096_snow_on_dem.tif')
hv_2015 = hv_src.read(1)
hv_2015 = np.ma.masked_array(hv_2015, hv_2015 == hv_src.nodatavals)

hv_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/hv/hv_2016_096_snow_on_dem.tif')
hv_2016 = hv_src.read(1)
hv_2016 = np.ma.masked_array(hv_2016, hv_2016 == hv_src.nodatavals)

#cmap = pylab.cm.get_cmap('viridis', 10)

plt.figure()
plt.suptitle("Winter Surfaces")
plt.subplots_adjust(wspace = 0.30,hspace = 0.8)

plt.subplot(1,4,1)
plt.imshow(hv_2012[0:5700], cmap = 'viridis', vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2012', fontsize = 8)

plt.subplot(1,4,2)
plt.imshow(hv_2013[0:5700], cmap = 'viridis', vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2013',fontsize = 8)

plt.subplot(1,4,3)
plt.imshow(hv_2015[0:5700], cmap = 'viridis', vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2015',fontsize = 8)

plt.subplot(1,4,4)
plt.imshow(hv_2016[0:5700], cmap = 'viridis', vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2016',fontsize = 8)
plt.colorbar()

plt.savefig('/home/cparr/surfaces/hv__winter_surfaces.png', dpi = 500, bbox_inches = 'tight')
