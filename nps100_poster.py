#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

"""
Created on Tue Oct 11 13:51:28 2016

@author: cparr
"""

'''
Script to create primary figure for NPS 100 Centennial Conference.
Poster will show peristence of tundra snow patterns using Happy Valley
lidar / SfM data.

'''
import rasterio
from collections import defaultdict
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LightSource

hv_snow = defaultdict(dict)
file_list = glob.glob('/home/cparr/Snow_Patterns/snow_data/happy_valley/raster/snow_on/full_extent/' + '*.tif')

def snow_process(list_of_rasters):
    
    for f in list_of_rasters:

        src = rasterio.open( f )
        name = f.split('/')[-1].rstrip('_snow_full.tif')
        name = name.strip('hv_')
        snow_mask = name+'_mask'
        subset = name+'_clipped'
        std = name+'_std'
        
        hv_snow[name] =  src.read(1)
        hv_snow[snow_mask] = np.ma.masked_values( hv_snow[name], src.nodata )
        hv_snow[subset] = hv_snow[snow_mask][2000:6000]
        hv_snow[subset] = hv_snow[subset][::,140::]
        hv_snow[std] = (hv_snow[subset] - hv_snow[subset].mean()) / hv_snow[subset].std()
                
snow_process(file_list)

### to get exact same shapes of data

nan12 = np.isnan(hv_snow['2012_std'])
nan13 = np.isnan(hv_snow['2013_std'])
nan15 = np.isnan(hv_snow['2015_std'])
nan1213 = np.logical_and(nan12,nan13)
nan1215 = np.logical_and(nan12,nan15)
nan1315 = np.logical_and(nan13,nan15)
nan1213_1215 = np.logical_and(nan1213, nan1215)
nans = np.logical_and(nan1213_1215,nan1315)
nan_mask = np.ma.make_mask(nans)

hv_snow_2012 = np.ma.masked_array(hv_snow['2012_std'], nan_mask)
hv_snow_2013 = np.ma.masked_array(hv_snow['2013_std'], nan_mask)
hv_snow_2015 = np.ma.masked_array(hv_snow['2015_std'], nan_mask)

elevation_src = rasterio.open('/home/cparr/Snow_Patterns/snow_data/happy_valley/raster/snow_free/dem/hv_2m_snowfree_dem_watermasked.tif')
elevation =  elevation_src.read(1)
elevation = elevation[2000:6000]
elevation = elevation[::,140::]#
elevation = np.ma.masked_less( elevation, 0, copy = True )
elev = np.ma.masked_array(elevation, nan_mask)
ls = LightSource(azdeg=280, altdeg=35)
cmap = plt.cm.terrain
rgb = ls.shade(elev, cmap=cmap, vert_exag=5, vmin=360, vmax=460, blend_mode='soft')

#hv_snow_2012 = cv2.bilateralFilter(hv_snow_2012,3,15,15)
#hv_snow_2013 = cv2.bilateralFilter(hv_snow_2013,3,15,15)
#hv_snow_2015 = cv2.bilateralFilter(hv_snow_2015,3,15,15)

###

ylabels = ['0','2','4','6','8']
xlabels = ['0','1','2']
extent = [0, elev.shape[1], 0, elev.shape[0]] # full dimensions
cracks = [350, 450, 2520, 2600] # crack subset dimensions
water_track = [340, 420, 3170, 3390] # bolt subset dimensions
creek = [680,840, 1150, 1210] # creek subset dimensions


###

def make_topo_inset(ax, zoom, location, img, coords, linecorner1, linecorner2):
    
    axins = zoomed_inset_axes( ax, zoom, loc=location, borderpad = 0.05)
    [i.set_color('#65665C') for i in axins.spines.itervalues()]
    [i.set_alpha(0.5) for i in axins.spines.itervalues()]
    [i.set_linewidth(7) for i in axins.spines.itervalues()]
    axins.set_xticks([])
    axins.set_yticks([])
    axins.imshow(img, extent = extent)
    axins.set_xlim(coords[0],coords[1])
    axins.set_ylim(coords[2],coords[3])
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=linecorner1, loc2=linecorner2, lw = 1*6, ec='k',
               alpha = 0.33, facecolor = 'none')

def make_snow_inset(ax, zoom, location, img, coords, linecorner1, linecorner2):
    
    axins = zoomed_inset_axes( ax, zoom, loc=location, borderpad = 0.05)
    [i.set_color('#65665C') for i in axins.spines.itervalues()]
    [i.set_alpha(0.5) for i in axins.spines.itervalues()]
    [i.set_linewidth(7) for i in axins.spines.itervalues()]
    axins.set_xticks([])
    axins.set_yticks([])
    axins.imshow(img, extent = extent, cmap = 'viridis', vmin=-3, vmax=3)
    axins.set_xlim(coords[0],coords[1])
    axins.set_ylim(coords[2],coords[3])
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=linecorner1, loc2=linecorner2, lw = 1*6, ec='k',
               alpha = 0.33, facecolor = 'none')

###
plt.figure(figsize=(48,30))
# baseline is (8,5)

ax1 = plt.subplot(141)
plt.imshow(rgb, extent = extent, interpolation='bilinear')
plt.yticks( [0,1000,2000,3000,4000] )
plt.xticks( [0, 500,1000] )
ax1.set_xticklabels(xlabels)
ax1.set_yticklabels(ylabels)
[i.set_linewidth(5) for i in ax1.spines.itervalues()]
plt.tick_params(
    axis='both',
    which='both',
    right = 'off',
    top='off',
    labelsize = 8*6,
    length = 24,
    width = 6)
plt.xlabel( 'km', fontsize = 10*6 )
plt.ylabel( 'km', fontsize = 10*6 )
plt.title( "Terrain", fontsize = 12*6)

make_topo_inset(ax1, 5, 6, rgb, cracks, 1, 2)
make_topo_inset(ax1, 5, 1, rgb, water_track, 1, 4)
make_topo_inset(ax1, 5, 3, rgb, creek, 1, 2)

###

ax2 = plt.subplot(142, sharey=ax1)
plt.imshow(hv_snow_2012, cmap = 'viridis', extent = extent, vmin=-3, vmax=3 )
plt.tick_params(
    axis='both',
    which='both',
    bottom='off',
    left = 'off',
    right = 'off',
    top='off',
    labelbottom='off',
    labelleft='off')
[i.set_linewidth(5) for i in ax2.spines.itervalues()]
plt.title( "2012", fontsize = 12*6)

make_snow_inset(ax2, 5, 6, hv_snow_2012, cracks, 1, 2)
make_snow_inset(ax2, 5, 1, hv_snow_2012, water_track, 2, 3)
make_snow_inset(ax2, 5, 3, hv_snow_2012, creek, 1, 2)
###

ax3 = plt.subplot(143, sharey=ax1)
plt.imshow(hv_snow_2013, cmap = 'viridis', extent = extent, vmin=-3, vmax=3 )
plt.tick_params(
    axis='both',
    which='both',
    bottom='off',
    left = 'off',
    right = 'off',
    top='off',
    labelbottom='off',
    labelleft='off')
[i.set_linewidth(5) for i in ax3.spines.itervalues()]
plt.title( "2013", fontsize = 12*6)

make_snow_inset(ax3, 5, 6, hv_snow_2013, cracks, 1, 2)
make_snow_inset(ax3, 5, 1, hv_snow_2013, water_track, 2, 3)
make_snow_inset(ax3, 5, 3, hv_snow_2013, creek, 1, 2)
###

ax4 = plt.subplot(144, sharey=ax1)
plt.imshow(hv_snow_2015, cmap = 'viridis', extent = extent, vmin=-3, vmax=3 )
plt.tick_params(
    axis='both',
    which='both',
    bottom='off',
    left = 'off',
    right = 'off',
    top='off',
    labelbottom='off',
    labelleft='off')
plt.title( "2015", fontsize = 12*6 )
[i.set_linewidth(5) for i in ax4.spines.itervalues()]
cbar = plt.colorbar(ticks = [-3,-2,-1,0,1,2,3])
cbar.ax.set_yticklabels(['-3$\sigma$','-2$\sigma$','-1$\sigma$','0','1$\sigma$','2$\sigma$','3$\sigma$'], fontsize = 10*6)


make_snow_inset(ax4, 5, 6, hv_snow_2015, cracks, 1, 2)
make_snow_inset(ax4, 5, 1, hv_snow_2015, water_track, 2, 3)
make_snow_inset(ax4, 5, 3, hv_snow_2015, creek, 1, 2)
###

plt.subplots_adjust(wspace=0, hspace=0, top = 0.85)
plt.suptitle("Happy Valley Snow Patterns\n Depth = Standard Deviation from the Annual Mean",fontsize = 14*6)
plt.savefig('/home/cparr/Snow_Patterns/figures/parr_nps100_poster.png',dpi = 400)

del ylabels
del xlabels
del creek
del water_track
del cracks
del extent
del file_list
del elevation
del nan12
del nan13
del nan15
del nan1213
del nan1215
del nan1315
del nan1213_1215
del nans