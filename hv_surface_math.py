import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pylab

hv_src = rasterio.open('/home/cparr/surfaces/level_1_surfaces/hv/hv_2012_158_bare_earth_dem.tif')
hv_bare = hv_src.read(1)
hv_bare = np.ma.masked_array(hv_bare, hv_bare == hv_src.nodatavals)
hv_bare = hv_bare[::,:-1]

hv_src = rasterio.open('/home/cparr/surfaces/depth_dems/hv/hv_2012_107_depth.tif')
hv_2012 = hv_src.read(1)
hv_2012 = np.ma.masked_array(hv_2012, hv_2012 == hv_src.nodatavals)

hv_src = rasterio.open('/home/cparr/surfaces/depth_dems/hv/hv_2013_107_depth.tif')
hv_2013 = hv_src.read(1)
hv_2013 = np.ma.masked_array(hv_2013, hv_2013 == hv_src.nodatavals

src = rasterio.open('/home/cparr/surfaces/depth_dems/hv/hv_2015_096_depth.tif')
hv_2015 = hv_src.read(1)
hv_2015 = np.ma.masked_array(hv_2015, hv_2015 == hv_src.nodatavals

hv_src = rasterio.open('/home/cparr/surfaces/depth_dems/hv/hv_2016_096_depth.tif')
hv_2016 = hv_src.read(1)
hv_2016 = np.ma.masked_array(hv_2016, hv_2016 == hv_src.nodatavals       

cmap = pylab.cm.get_cmap('spectral', 9)    # 11 discrete colors

plt.figure()
plt.subplots_adjust(wspace = 0.30,hspace = 0.85)

plt.subplot(1,4,1)
plt.imshow(hv_2012, cmap = cmap, vmin = -0.25, vmax = 2)
plt.xticks([])
plt.yticks([])
plt.title('2012')

plt.subplot(1,4,2)
plt.imshow(hv_2013, cmap = cmap, vmin = -0.25, vmax = 2)
plt.xticks([])
plt.yticks([])
plt.title('2013')

plt.subplot(1,4,3)
plt.imshow(hv_2015, cmap = cmap, vmin = -0.25, vmax = 2)
plt.xticks([])
plt.yticks([])
plt.title('2015')

plt.subplot(1,4,4)
plt.imshow(hv_2016, cmap = cmap, vmin = -0.25, vmax = 2)
plt.xticks([])
plt.yticks([])
plt.title('2016')


plt.savefig('/home/cparr/surfaces/hv_depths_from_las.tif', dpi = 300, bbox_inches = 'tight')
