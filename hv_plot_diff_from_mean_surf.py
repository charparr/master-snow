#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:17:00 2016

@author: cparr
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 09:54:51 2016

@author: cparr
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pylab

hv_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/hv/hv_2012_158_bare_earth_dem.tif')
hv_bare = hv_src.read(1)
hv_bare = np.ma.masked_array(hv_bare, hv_bare == hv_src.nodatavals)
hv_bare = hv_bare[0:5700]

hv_src = rasterio.open('/home/cparr/scratch/hv/hv_2012_minus_mean.tif')
hv_2012 = hv_src.read(1)
hv_2012 = np.ma.masked_array(hv_2012, hv_2012 == hv_src.nodatavals)
hv_2012 = hv_2012[0:5700]

hv_src = rasterio.open('/home/cparr/scratch/hv/hv_2013_minus_mean.tif')
hv_2013 = hv_src.read(1)
hv_2013 = np.ma.masked_array(hv_2013, hv_2013 == hv_src.nodatavals)
hv_2013 = hv_2013[0:5700]

hv_src = rasterio.open('/home/cparr/scratch/hv/hv_2015_minus_mean.tif')
hv_2015 = hv_src.read(1)
hv_2015 = np.ma.masked_array(hv_2015, hv_2015 == hv_src.nodatavals)
hv_2015 = hv_2015[0:5700]

hv_src = rasterio.open('/home/cparr/scratch/hv/hv_2016_minus_mean.tif')
hv_2016 = hv_src.read(1)
hv_2016 = np.ma.masked_array(hv_2016, hv_2016 == hv_src.nodatavals)      
hv_2016 = hv_2016[0:5700]



#horiztonal slice of difference maps
delta15 = hv_2015[0:50][1]

delta13 = hv_2013[0:50][1]

delta12 = hv_2012[0:50][1]

delta16 = hv_2016[0:50][1]

plt.plot(delta12,label='2012')
plt.plot(delta13,label='2013')
plt.plot(delta15,label='2015')
plt.plot(delta16,label='2016')
plt.ylim(-0.5,0.5)
plt.legend()
#####






cmap = pylab.cm.get_cmap('coolwarm', 5)

plt.figure()
plt.suptitle("Winter Surfaces - Mean Winter Surface [m]")
plt.subplots_adjust(wspace = 0.30,hspace = 0.8)

plt.subplot(1,4,1)
plt.imshow(hv_2012[0:5700], cmap = cmap, vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2012 $\mu$ = ' + str(round(np.nanmean(hv_2012[0:5700]),2)),fontsize = 8)

plt.subplot(1,4,2)
plt.imshow(hv_2013[0:5700], cmap = cmap, vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2013 $\mu$ = ' + str(round(np.nanmean(hv_2013[0:5700]),2)),fontsize = 8)

plt.subplot(1,4,3)
plt.imshow(hv_2015[0:5700], cmap = cmap, vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2015 $\mu$ = ' + str(round(np.nanmean(hv_2015[0:5700]),2)),fontsize = 8)

plt.subplot(1,4,4)
plt.imshow(hv_2016[0:5700], cmap = cmap, vmin = -0.5, vmax = 0.5)
plt.xticks([])
plt.yticks([])
plt.title('2016 $\mu$ = ' + str(round(np.nanmean(hv_2016[0:5700]),2)),fontsize = 8)
plt.colorbar()

plt.savefig('/home/cparr/surfaces/hv_diffs_from_mean_winter_surface.png', dpi = 500, bbox_inches = 'tight')

plt.figure(8,5)
plt.subplot 