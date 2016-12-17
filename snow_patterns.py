
# coding: utf-8

# In[1]:
    
import phasepack
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.transform import *
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.morphology import square
from skimage.util import random_noise
from scipy.spatial import procrustes
from scipy import signal
from scipy import fftpack
from matplotlib import six
from skimage.transform import warp


# In[2]:

### Similarity Test Functions ###

def procrustes_analysis(data):
    for d in data:
        mtx1, mtx2, disparity = procrustes(data[0], d)
        # disparity is the sum of the square errors
        # mtx2 is the optimal matrix transformation
        disp_vals.append(disparity.round(3))
        pd_maps.append(mtx2)
        
def structural_sim(data):
    
    for d in data:
        ssim_vals.append( ssim ( data[0], d ).round( 2 ))
        
        ssim_maps.append( ssim ( data[0], d, full  = True )[1] )
        
def reg_mse(data):
    for d in data:
        mse_vals.append(( mse ( data[0], d )).round(2))
        mse_maps.append((data[0] - d) ** 2)

def cw_ssim_value(data, width):
        """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
        image to the target image.
        Args:
          target (str or PIL.Image): Input image to compare the reference image
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).
          width: width for the wavelet convolution (default: 30)
        Returns:
          Computed CW-SSIM float value.
        """

        # Define a width for the wavelet convolution
        widths = np.arange(1, width+1)

        for d in data:
        
            # Use the image data as arrays
            sig1 = np.asarray(data[0].ravel())
            sig2 = np.asarray(d.ravel())

            # Convolution
            cwtmatr1 = signal.cwt(sig1, signal.ricker, widths)
            cwtmatr2 = signal.cwt(sig2, signal.ricker, widths)

            # Compute the first term
            c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
            c1_2 = np.square(abs(cwtmatr1))
            c2_2 = np.square(abs(cwtmatr2))
            num_ssim_1 = 2 * np.sum(c1c2, axis=0) + 0.01
            den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + 0.01

            # Compute the second term
            c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
            num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + 0.01
            den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + 0.01

            # Construct the result
            ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)
            ssim_map = ssim_map.reshape(512,512)
            cw_ssim_maps.append(ssim_map)

            # Average the per pixel results
            index = round( np.average(ssim_map), 2) 
            cw_ssim_vals.append(index)
            
def disccost( data ):
    
    for d in data:
        y = fftpack.dct( d )
        dct_maps.append( y )
        yc = y.mean(axis = 1 )
        dct_curves.append( yc )
        
def convolve(image, kernel):
    
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
 
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
 
    pad = ( kW - 1 ) / 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
 
	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
     
    for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
      
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
      
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
 
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
 
			k = (roi * kernel).sum()
 
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
 
			output[y - pad, x - pad] = k
    return output

def gmsd(data):
    
    """
    Return a map of Gradient Magnitude Similarity and a global index measure.
    
    Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014). 
    Gradient magnitude similarity deviation: A highly efficient perceptual
    image quality index. IEEE Transactions on Image Processing, 23(2), 668–695.
    http://doi.org/10.1109/TIP.2013.2293423
    """
    for d in data:
        
        reference = data[0]
        # Prewitt kernels given in paper
        h_x = [0.33, 0, -0.33,0.33, 0, -0.33, 0.33, 0, -0.33]
        h_x = np.array(h_x).reshape(3,3)                  
        h_y = np.flipud( np.rot90(h_x) )
        
        # Create gradient magnitude images with Prewitt kernels
        ref_conv_hx = convolve(reference, h_x)
        ref_conv_hy = convolve(reference, h_y)
        ref_grad_mag = np.sqrt( ( ref_conv_hx**2 ) + ( ref_conv_hy**2 ) )
        
        dst_conv_hx = convolve(d, h_x)
        dst_conv_hy = convolve(d, h_y)
        dst_grad_mag = np.sqrt( ( dst_conv_hx**2 ) + ( dst_conv_hy**2 ) )
            
        c = 0.0026  # Constant provided by the authors
        
        gms_map = ( 2 * ref_grad_mag * dst_grad_mag + c ) / ( ref_grad_mag**2 + dst_grad_mag**2 + c )
        
        gms_index = round(( np.sum(( gms_map-gms_map.mean() )**2 ) / gms_map.size )**0.5, 3 )    
        
        gms_vals.append(gms_index)
        gms_maps.append(gms_map)
    
def feature_sim(data):
    
    """
    Return the Feature Similarity Index (FSIM).
    Can also return FSIMc for color images
    
    Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). 
    FSIM: A feature similarity index for image quality assessment. 
    IEEE Transactions on Image Processing, 20(8), 2378–2386. 
    http://doi.org/10.1109/TIP.2011.2109730
    """
    
    # Convert the input images to YIQ color space
    # Y is the luma compenent, i.e. B & W
    # imgY = 0.299 * r + 0.587 * g + 0.114 * b

    for d in data:
        
        reference = data[0]
        # Constants provided by the authors
        t1 = 0.85
        t2 = 160
        
        # Phase congruency (PC) images. "PC...a dimensionless measure for the
        # significance of local structure.
        
        pc1 = phasepack.phasecong(reference, nscale = 4, norient = 4, 
                                  minWaveLength = 6, mult = 2, sigmaOnf=0.55)
                                  
        pc2 = phasepack.phasecong(d, nscale = 4, norient = 4,
                                  minWaveLength = 6, mult = 2, sigmaOnf=0.55)
                                  
        pc1 = pc1[0]  # Reference PC map
        pc2 = pc2[0]  # Distorted PC map
        
        # Similarity of PC components
        s_PC = ( 2*pc1 + pc2 + t1 )  / ( pc1**2 + pc2**2 + t1 )
        
        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        refgradX = cv2.Sobel(reference, cv2.CV_64F, dx = 1, dy = 0, ksize = -1)
        refgradY = cv2.Sobel(reference, cv2.CV_64F, dx = 0, dy = 1, ksize = -1)
        
        targradX = cv2.Sobel(d, cv2.CV_64F, dx = 1, dy = 0, ksize = -1)
        targradY = cv2.Sobel(d, cv2.CV_64F, dx = 0, dy = 1, ksize = -1)
        
        refgradient = np.maximum(refgradX, refgradY)    
        targradient = np.maximum(targradX, targradY)   
        
        #refgradient = np.sqrt(( refgradX**2 ) + ( refgradY**2 ))
        
        #targradient = np.sqrt(( targradX**2 ) + ( targradY**2 ))
    
        # The gradient magnitude similarity
    
        s_G = (2*refgradient + targradient + t2) / (refgradient**2 + targradient**2 + t2)
        
        s_L = s_PC * s_G  # luma similarity
        
        pcM = np.maximum(pc1,pc2)
            
        fsim = round( np.nansum( s_L * pcM) / np.nansum(pcM), 3)
        
        fsim_vals.append(fsim)
    
### Plotting Functions

def plot_snow(names, data):
    
    fig, axes = plt.subplots( nrows = 2, ncols = 5 )
    fig.suptitle('Test Snow Patterns', color = 'white')
    
    for p, dat, ax in zip( names, data, axes.flat ):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest', vmin = 0, vmax = 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(p,fontsize = 8, color = 'white')
    
    # Make an axis for the colorbar on the bottom
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ( [0,1,2] ) )
    cax.tick_params(labelsize = 8, colors = 'white')
    
    
def plot_tests(names, test_vals, test_name, data, rows, cols, cmin, cmax):
    
    fig, axes = plt.subplots( nrows = 2, ncols = 5 )
    fig.suptitle( test_name + 'Fidelity Test of Snow Patterns' )
    
    for p, v, dat, ax in zip( names, test_vals, data, axes.flat ):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest', vmin = cmin, vmax = cmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(p + " " + test_name + str(v), fontsize = 6, color = 'white' )
    
    # Make an axis for the colorbar on the bottom
    
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ( [cmin, cmax] ) )
    cax.tick_params(labelsize = 6, colors = 'white')
    
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#236192', row_colors=['#C7C9C7', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText = data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='#FFCD00')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax


# In[3]:

#Snow

src1 = rasterio.open( '/home/cparr/Snow_Patterns/snow_data/happy_valley/raster/snow_on/hv_snow_watertrack_square2012.tif' )
snow_test = src1.read(1)
snow_test = snow_test.astype('float64')


# In[4]:

'''
Snow Data Test
'''

snow_data = []

# Initialize lists for metrics.

mse_vals = []
ssim_vals = []
disp_vals = []
mse_maps = []
ssim_maps = []
mse_maps = []
cw_ssim_vals = []
cw_ssim_maps = []
dct_maps = []
dct_curves = []
pd_maps = []
fsim_vals = []
gms_vals = []
gms_maps = []

# Create the test snows.
def warp_snow(snow):
    
    snow_data.append(snow)
    
    rows, cols = snow.shape
    mu = snow.mean()
    sigma = snow.std()

    # 90 degree rotation
    rotate90 = np.rot90(snow)
    snow_data.append(rotate90)
    
    #45 degree rotation
    oblique = rotate(snow, 45)
    b = oblique == 0
    oblique[b] = np.random.normal(mu, sigma, size=b.sum())
    snow_data.append(oblique)
    
    # morphological dilation and erosion

    selem = square(7)
    morph_dilation = dilation(snow, selem)
    morph_erosion = erosion(snow, selem)
    snow_data.append(morph_dilation)
    snow_data.append(morph_erosion)
    
    # flip up and down, basically a phase shift
    inverse = np.flipud(snow)
    snow_data.append(inverse)
    
    # a shift or translation
    shift_M = np.float32([[1,0,1],[0,1,0]])
    shifted = cv2.warpAffine(snow,shift_M,(cols,rows))
    snow_data.append(shifted)
    
    # Random Affine Transformation    
    c = np.round(np.random.rand( 3,2 ), 2)
    m = np.append( c, ( 0,0,1 ) )
    m = m.reshape( 3,3 )
    aff_t = AffineTransform( matrix = m )
    random_aff_warp = warp( snow, aff_t )
    b = random_aff_warp == 0
    random_aff_warp[b] = np.random.normal(mu, sigma, size=b.sum())
    snow_data.append(random_aff_warp)
    
    # Additive Gaussian Noise
    noise = random_noise( snow, mode = 'gaussian' )
    snow_data.append( noise )
    
    # More Additive Gaussian Noise
    more_noise = random_noise(random_noise(random_noise(random_noise( noise, mode = 'gaussian' ))))
    snow_data.append( more_noise )
    
# Plot Titles and dictionary keys
snow_names = ['Reference', 'Rotate 90', 'Rotate 45', 'Dilation',
                 'Erosion', 'Y - Reflection', 'X Shift',
                 'Random Affine', 'Add Gaussian Noise','More Noise']

# Call It.
warp_snow( snow_test )

# Call Metrics on list of test snows

structural_sim( snow_data )
reg_mse( snow_data )
procrustes_analysis( snow_data )
cw_ssim_value(snow_data, 20)
disccost( snow_data )
feature_sim( snow_data )
gmsd( snow_data )

# Zip names, data, metrics, quadrants into a mega list!
# Generally this is indavisable because it relies on indexing...in the next cell we will make a dictionary.
snow_zip = zip(snow_names,snow_data,mse_vals, ssim_vals, disp_vals, 
              mse_maps, ssim_maps, cw_ssim_vals, cw_ssim_maps, dct_maps,
              dct_curves, pd_maps, fsim_vals, gms_vals, gms_maps )


# In[5]:

snow_dict = defaultdict(dict)

'''
# Making a look up dictionary for patterns and their comparison scores.
'''

def to_dict_w_hists( data_dict, keys, data_zip ):

    i = 0
    while i < len(keys):

        data_dict[keys[i]]['name'] = data_zip[i][0]

        data_dict[keys[i]]['arrays'] = {}

        data_dict[keys[i]]['arrays']['full'] = {}
        data_dict[keys[i]]['arrays']['full']['array'] = data_zip[i][1]
        
        data_dict[keys[i]]['MSE'] = round(data_zip[i][2], 2)
        data_dict[keys[i]]['SSIM'] = round(data_zip[i][3], 2)
        data_dict[keys[i]]['Procrustres Disparity'] = round(data_zip[i][4], 2)
        data_dict[keys[i]]['MSE Map'] = data_zip[i][5]
        data_dict[keys[i]]['SSIM Map'] = data_zip[i][6]
        data_dict[keys[i]]['CW SSIM'] = data_zip[i][7]
        data_dict[keys[i]]['CW SSIM Map'] = data_zip[i][8]
        data_dict[keys[i]]['DCT Map'] = data_zip[i][9]
        data_dict[keys[i]]['DCT Curve'] = data_zip[i][10]
        data_dict[keys[i]]['PD Maps'] = data_zip[i][11]
        data_dict[keys[i]]['FSIM'] = data_zip[i][12]
        data_dict[keys[i]]['GMS'] = data_zip[i][13]
        data_dict[keys[i]]['GMS Maps'] = data_zip[i][14]

        i = i + 1

# In[6]:

to_dict_w_hists( snow_dict, snow_names, snow_zip )
snow_df = pd.DataFrame.from_dict(snow_dict)
snow_df = snow_df.transpose()

# In[8]:

# Snow Scores and Ranks

snow_scores = snow_df.copy()
snow_scores['Pattern'] = snow_df['name']
snow_scores = snow_scores[['Pattern', 'MSE', 'SSIM', 'Procrustres Disparity',
                           'GMS','FSIM', 'CW SSIM']]
snow_scores = snow_scores.sort_values( 'CW SSIM', ascending = False )

ranks = snow_scores.copy()
ranks['Pattern'] = snow_df['name']
ranks['MSE Rank'] = np.round(snow_scores['MSE'].rank(ascending=True))
ranks['SSIM Rank'] = snow_scores['SSIM'].rank(ascending=False)
ranks['CW-SSIM Rank'] = snow_scores['CW SSIM'].rank(ascending=False)
ranks['Disparity Rank'] = snow_scores['Procrustres Disparity'].rank()
ranks['GMS Rank'] = snow_scores['GMS'].rank()
ranks['FSIM Rank'] = snow_scores['FSIM'].rank(ascending = False)

del ranks['MSE']
del ranks['SSIM']
del ranks['CW SSIM']
del ranks ['Procrustres Disparity']
del ranks['GMS']
del ranks['FSIM']
ranks = ranks.sort_values('CW-SSIM Rank')

# In[24]:

render_mpl_table(ranks)
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/snow_test_ranks.png',
            bbox_inches = 'tight', dpi = 300)
plt.close()

render_mpl_table(snow_scores)
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/snow_test_scores.png',
            bbox_inches = 'tight', dpi = 300)
plt.close()

plot_snow( snow_names, snow_data )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_snow_warps.png',
            bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()

# names, test_vals, test_name, data, rows, cols, cmin, cmax

plot_tests( snow_names, gms_vals, " GMS: ", gms_maps, 2, 5, 0, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_gms_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()

plot_tests( snow_names, mse_vals, " MSE: ", mse_maps, 2, 5, 0, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_mse_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()

plot_tests( snow_names, ssim_vals, " SSIM: ", ssim_maps, 2, 5, -1, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_ssim_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()

plot_tests( snow_names, cw_ssim_vals, " CW SSIM: ", cw_ssim_maps, 2, 5, -1, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_cw_ssim_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()


# In[12]:
#DCT Plots
    
def plot_transforms(names, data, title, rows, cols, cmin, cmax):
    
    fig, axes = plt.subplots( nrows = rows, ncols = cols )
    fig.suptitle( title )
    
    for p, dat, ax in zip( names, data, axes.flat ):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest', vmin = cmin, vmax = cmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title( p, fontsize = 8 )
        
    # Make an axis for the colorbar on the bottom
    
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ( [cmin, cmax] ) )
    cax.tick_params(labelsize = 8)


# In[23]:

### transform plots

plot_transforms( snow_names, dct_maps, "DCT Maps of Snow Patterns", 2, 5, -5, 5)
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_dct_maps.png', bbox_inches = 'tight',
             dpi = 300)
plt.close()


fig, axes = plt.subplots( nrows = 2, ncols = 5 )

fig.suptitle( 'X Transect Mean of DCT of Snow Patterns' )

for p, dat, ax in zip( snow_names, dct_curves, axes.flat ):
    
    ymin = np.round(dat.min(),1)
    ymax = np.round(dat.max(),1)
    
    f = ax.plot( dat, lw = 1, color = '#236192' )
    ax.plot( dct_curves[0], lw = 1, ls = 'dashed', color = '#FFCD00', alpha = 0.67 )
    
    ax.set_yticks([ymin,ymax])
    ax.set_xticks([0,512])
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_yticklabels([ymin,ymax], size = 6)
    ax.set_xticklabels([0,512], size = 6)
    
    ax.set_xlim(0,512)
    ax.set_title( p, size = 7 )

plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_dct_lines.png', bbox_inches = 'tight',
            dpi = 300)
plt.close()
