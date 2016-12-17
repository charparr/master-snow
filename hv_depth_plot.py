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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

hv_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/hv/hv_2012_158_bare_earth_dem.tif')
hv_bare = hv_src.read(1)
hv_bare = np.ma.masked_array(hv_bare, hv_bare == hv_src.nodatavals)
hv_bare = hv_bare[0:5700]

hv_src = rasterio.open('/home/cparr/surfaces/depth_ddems/hv/hv_2012_107_depth.tif')
hv_2012 = hv_src.read(1)
hv_2012 = np.ma.masked_array(hv_2012, hv_2012 == hv_src.nodatavals)
hv_2012 = hv_2012[0:5700]

hv_src = rasterio.open('/home/cparr/surfaces/depth_ddems/hv/hv_2013_103_depth.tif')
hv_2013 = hv_src.read(1)
hv_2013 = np.ma.masked_array(hv_2013, hv_2013 == hv_src.nodatavals)
hv_2013 = hv_2013[0:5700]

hv_src = rasterio.open('/home/cparr/surfaces/depth_ddems/hv/hv_2015_096_depth.tif')
hv_2015 = hv_src.read(1)
hv_2015 = np.ma.masked_array(hv_2015, hv_2015 == hv_src.nodatavals)
hv_2015 = hv_2015[0:5700]

hv_src = rasterio.open('/home/cparr/surfaces/depth_ddems/hv/hv_2016_096_depth.tif')
hv_2016 = hv_src.read(1)
hv_2016 = np.ma.masked_array(hv_2016, hv_2016 == hv_src.nodatavals)      
hv_2016 = hv_2016[0:5700]

cmap = pylab.cm.get_cmap('viridis', 8)

ylabels = ['0','2','4','6','8','10']
xlabels = ['0','1','2']
extent = [0, hv_bare.shape[1], 0, hv_bare.shape[0]] # full dimensions
cracks = [610, 730, 2750, 2950] # crack subset dimensions
water_track = [450, 600, 3320, 3450] # bolt subset dimensions
creek = [480,580, 900, 1100] # creek subset dimensions

def make_snow_inset(ax, zoom, location, img, coords, linecorner1, linecorner2):
    
    axins = zoomed_inset_axes( ax, zoom, loc=location, borderpad = 0.05)
    [i.set_color('#65665C') for i in axins.spines.itervalues()]
    [i.set_alpha(0.5) for i in axins.spines.itervalues()]
    [i.set_linewidth(1) for i in axins.spines.itervalues()]
    axins.set_xticks([])
    axins.set_yticks([])
    axins.imshow(img, extent = extent, cmap = 'viridis', vmin=0, vmax=2)
    axins.set_xlim(coords[0],coords[1])
    axins.set_ylim(coords[2],coords[3])
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=linecorner1, loc2=linecorner2, lw = 1, ec='k',
               alpha = 0.33, facecolor = 'none')


plt.figure(figsize=(8,5))

ax1 = plt.subplot(141)
plt.imshow(hv_2012, cmap = cmap, extent = extent, vmin=0, vmax=2 )
plt.yticks( [0,1000,2000,3000,4000,5000] )
plt.xticks( [0, 500,1000] )
ax1.set_xticklabels(xlabels)
ax1.set_yticklabels(ylabels)
[i.set_linewidth(2) for i in ax1.spines.itervalues()]
plt.tick_params(
    axis='both',
    which='both',
    right = 'off',
    top='off',
    labelsize = 8,
    length = 4,
    width = 1)
plt.xlabel( 'km', fontsize = 10)
plt.ylabel( 'km', fontsize = 10)
plt.title( '2012 $\mu$ = ' + str(round(np.nanmean(hv_2012),2)) + 
          ', $\sigma$ = ' + str(round(np.nanstd(hv_2012),2)),fontsize = 8)

make_snow_inset(ax1, 5, 6, hv_2012, cracks, 1, 4)
make_snow_inset(ax1, 5, 1, hv_2012, water_track, 2, 3)
make_snow_inset(ax1, 5, 3, hv_2012, creek, 1, 4)

###

ax2 = plt.subplot(142, sharey=ax1)
plt.imshow(hv_2013, cmap = cmap, extent = extent, vmin=0, vmax=2 )
plt.tick_params(
    axis='both',
    which='both',
    bottom='off',
    left = 'off',
    right = 'off',
    top='off',
    labelbottom='off',
    labelleft='off')
[i.set_linewidth(2) for i in ax2.spines.itervalues()]
plt.title( '2013 $\mu$ = ' + str(round(np.nanmean(hv_2013),2)) + 
          ', $\sigma$ = ' + str(round(np.nanstd(hv_2013),2)),fontsize = 8)

make_snow_inset(ax2, 5, 6, hv_2013, cracks, 1, 4)
make_snow_inset(ax2, 5, 1, hv_2013, water_track, 2, 3)
make_snow_inset(ax2, 5, 3, hv_2013, creek, 1, 4)
###

ax3 = plt.subplot(143, sharey=ax1)
plt.imshow(hv_2015, cmap = cmap, extent = extent, vmin=0, vmax=2 )
plt.tick_params(
    axis='both',
    which='both',
    bottom='off',
    left = 'off',
    right = 'off',
    top='off',
    labelbottom='off',
    labelleft='off')
[i.set_linewidth(2) for i in ax3.spines.itervalues()]
plt.title( '2015 $\mu$ = ' + str(round(np.nanmean(hv_2015),2)) + 
          ', $\sigma$ = ' + str(round(np.nanstd(hv_2015),2)),fontsize = 8)

make_snow_inset(ax3, 5, 6, hv_2015, cracks, 1, 4)
make_snow_inset(ax3, 5, 1, hv_2015, water_track, 2, 3)
make_snow_inset(ax3, 5, 3, hv_2015, creek, 1, 4)
###

ax4 = plt.subplot(144, sharey=ax1)
plt.imshow(hv_2016, cmap = cmap, extent = extent, vmin=0, vmax=2 )
plt.tick_params(
    axis='both',
    which='both',
    bottom='off',
    left = 'off',
    right = 'off',
    top='off',
    labelbottom='off',
    labelleft='off')
plt.title( '    2016 $\mu$ = ' + str(round(np.nanmean(hv_2016),2)) + 
          ', $\sigma$ = ' + str(round(np.nanstd(hv_2016),2)),fontsize = 8)

[i.set_linewidth(2) for i in ax4.spines.itervalues()]
cbar = plt.colorbar(ticks = [0,0.5,1,1.5,2])

make_snow_inset(ax4, 5, 6, hv_2016, cracks, 1, 4)
make_snow_inset(ax4, 5, 1, hv_2016, water_track, 2, 3)
make_snow_inset(ax4, 5, 3, hv_2016, creek, 1, 4)

plt.subplots_adjust(wspace=0, hspace=0, top = 0.85)
plt.suptitle("Happy Valley Snow Depth Patterns [m]",fontsize = 14)
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_depth_map.png',dpi = 300)