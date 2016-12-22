#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Sep 20 12:26:13 2016

@author: cparr

This script uses a variety of image quality assessment (IQA) metrics to
determine the similiarity or lack thereof between two images.

"""
import glob
import phasepack
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from scipy import signal
from scipy import fftpack
from matplotlib import six

# Globals go here
master = dict()
compare_years_l1 = dict()
         
def input_data(path_to_snow_data, y1,y2,x1,ix2):
    
    """
    Return list of .tif files from the specified directory.
    Subset each .tif into ROI and store in dicts.
    User determines coordinates.
    """
    
    file_list = glob.glob(str(path_to_snow_data) + '*.tif')
    
    for f in file_list:
        
        src = rasterio.open(f)
        src_img = src.read(1)
        src_img = np.ma.masked_values( src_img, src.nodata )[0:5700]
        
        roi_img = src_img[y1:y2,x1:ix2]
        
        #plt.figure()
        #plt.imshow(roi_img)    

        name = f.split('/')[-1]
        name = filter(lambda x: x.isdigit(), name)
        date =  name[:4] + '_' + name[4:]
        
        master[date] = dict()
        master[date][date + '_source'] = src_img
        master[date][date + '_roi'] = roi_img
        
        print str(date) +  ' is subset and stored in memory'
    
            
    print "Source shape: " + str(src_img.shape)
    print "ROI shape: " + str(roi_img.shape)

def create_pairs():
    '''
    This finds all unique combinations of years. The indexing chooses which
    image extent to use, i.e. source or roi for the comparisons.
    The pairs are stored in a comparison dictionary. Each key is a pair of
    observations from different years over the same location.
    '''

    for pair in combinations(master.iterkeys(), 2):
        
        years = pair[0][0:4] + " vs. " + pair[1][0:4]
        
        compare_years_l1[years] = {}
        compare_years_l1[years][pair[0]] = master[pair[0]][pair[0] + '_roi']
        compare_years_l1[years][pair[1]] = master[pair[1]][pair[1] + '_roi']
        
    print "Creating " + str(len(compare_years_l1.keys())) + " Unique Map Pairs"
        
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
        
def discostrans(reference, target):
    
    reference_transform = fftpack.dct( reference )
    target_transform = fftpack.dct( target )
    reference_curve = reference_transform.mean(axis = 0) #
    target_curve = target_transform.mean(axis = 0)

    return reference_transform, target_transform, reference_curve, target_curve


def gmsd(reference, target):
    
    """
    Return a map of Gradient Magnitude Similarity and a global index measure.
    
    Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014). 
    Gradient magnitude similarity deviation: A highly efficient perceptual
    image quality index. IEEE Transactions on Image Processing, 23(2), 668–695.
    http://doi.org/10.1109/TIP.2013.2293423
    """
        
    # Prewitt kernels given in paper
    h_x = [0.33, 0, -0.33,0.33, 0, -0.33, 0.33, 0, -0.33]
    h_x = np.array(h_x).reshape(3,3)                  
    h_y = np.flipud( np.rot90(h_x) )
    
    # Create gradient magnitude images with Prewitt kernels
    ref_conv_hx = convolve(reference, h_x)
    ref_conv_hy = convolve(reference, h_y)
    ref_grad_mag = np.sqrt( ( ref_conv_hx**2 ) + ( ref_conv_hy**2 ) )
    
    dst_conv_hx = convolve(target, h_x)
    dst_conv_hy = convolve(target, h_y)
    dst_grad_mag = np.sqrt( ( dst_conv_hx**2 ) + ( dst_conv_hy**2 ) )
        
    c = 0.0026  # Constant provided by the authors
    
    gms_map = ( 2 * ref_grad_mag * dst_grad_mag + c ) / ( ref_grad_mag**2 + dst_grad_mag**2 + c )
    
    gms_index = round(( np.sum(( gms_map-gms_map.mean() )**2 ) / gms_map.size )**0.5, 3 )    
    
    return gms_index, gms_map

def ft_sim(reference, target):
    
    """
    Return the Feature Similarity Index (FSIM).
    Can also return FSIMc for color images
    
    Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). 
    FSIM: A feature similarity index for image quality assessment. 
    IEEE Transactions on Image Processing, 20(8), 2378–2386. 
    http://doi.org/10.1109/TIP.2011.2109730
    """

    # Constants provided by the authors
    
    t1 = 0.85
    t2 = 160
    
    # Phase congruency (PC) images. "PC...a dimensionless measure for the
    # significance of local structure.
    
    pc1 = phasepack.phasecong(reference, nscale = 4, norient = 4, 
                              minWaveLength = 6, mult = 2, sigmaOnf=0.55)
                              
    pc2 = phasepack.phasecong(target, nscale = 4, norient = 4,
                              minWaveLength = 6, mult = 2, sigmaOnf=0.55)
                              
    pc1 = pc1[0]  # Reference PC map
    pc2 = pc2[0]  # Distorted PC map
    
    # Similarity of PC components
    s_PC = ( 2*pc1*pc2 + t1 )  / ( pc1**2 + pc2**2 + t1 )
    
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    refgradX = cv2.Sobel(reference, cv2.CV_64F, dx = 1, dy = 0, ksize = -1)
    refgradY = cv2.Sobel(reference, cv2.CV_64F, dx = 0, dy = 1, ksize = -1)
    
    targradX = cv2.Sobel(target, cv2.CV_64F, dx = 1, dy = 0, ksize = -1)
    targradY = cv2.Sobel(target, cv2.CV_64F, dx = 0, dy = 1, ksize = -1)
    
    refgradient = np.maximum(refgradX, refgradY)    
    targradient = np.maximum(targradX, targradY)   
    
    #refgradient = np.sqrt(( refgradX**2 ) + ( refgradY**2 ))
    
    #targradient = np.sqrt(( targradX**2 ) + ( targradY**2 ))

    # The gradient magnitude similarity

    s_G = (2*refgradient*targradient + t2) / (refgradient**2 + targradient**2 + t2)
    
    s_L = s_PC * s_G  # luma similarity
    
    pcM = np.maximum(pc1,pc2)
        
    fsim = round( np.nansum( s_L * pcM) / np.nansum(pcM), 3)
    
    return fsim

def cw_ssim(reference, target, width):
    
        """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
        image to the target image.
        Args:
          reference and target images
          width: width for the wavelet convolution (default: 30)
        Returns:
          Computed CW-SSIM float value and map.
        """

        # Define a width for the wavelet convolution
        widths = np.arange(1, width+1)
        
        # Use the image data as arrays
        sig1 = np.ravel(reference)
        sig2 = np.ravel(target)

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
        cw_ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)
        cw_ssim_map = cw_ssim_map.reshape(reference.shape[0],
                                          reference.shape[1])

        # Average the per pixel results
        cw_ssim_index = round( np.average(cw_ssim_map), 3)
        
        return cw_ssim_index, cw_ssim_map

def do_all_metrics():
    
    pairs = [k for k in compare_years_l1.iterkeys()]
    
    for p in pairs:
        
        t = [k for k in combinations(compare_years_l1[p].iterkeys(), 2)]
        yr1 = t[0][0]
        yr2 = t[0][1]
        
        compare_years_l1[p]['MSE'] = round(mse(compare_years_l1[p][yr1],
                                            compare_years_l1[p][yr2]),3)
        compare_years_l1[p]['MSE Map'] = (compare_years_l1[p][yr1] - 
        compare_years_l1[p][yr2])**2
        
        ssim_results = ssim(compare_years_l1[p][yr1],compare_years_l1[p][yr2],
                            full = True)
        compare_years_l1[p]['SSIM'] = round(ssim_results[0],3)
        compare_years_l1[p]['SSIM Map'] = ssim_results[1]
        
        cw_ssim_results = cw_ssim(compare_years_l1[p][yr1],
                                  compare_years_l1[p][yr2],20) #width of filter
        compare_years_l1[p]['CW-SSIM'] = round(cw_ssim_results[0],3)
        compare_years_l1[p]['CW-SSIM Map'] = cw_ssim_results[1]
        
        gmsd_results = gmsd(compare_years_l1[p][yr1], compare_years_l1[p][yr2])
        compare_years_l1[p]['GMSD'] = round(gmsd_results[0],3)
        compare_years_l1[p]['GMSD Map'] = gmsd_results[1]
        
        tr_res = discostrans(compare_years_l1[p][yr1],compare_years_l1[p][yr2])
        compare_years_l1[p]['DCT Map '+yr1] = tr_res[0]
        compare_years_l1[p]['DCT Map '+yr2] = tr_res[1]
        compare_years_l1[p]['DCT Graph '+yr1] = tr_res[2]
        compare_years_l1[p]['DCT Graph '+yr2] = tr_res[3]

        compare_years_l1[p]['FSIM'] = ft_sim(compare_years_l1[p][yr1],compare_years_l1[p][yr2])


input_data('/home/cparr/surfaces/depth_ddems/hv/',4400,4800,475,555)
create_pairs()
do_all_metrics()
input_data('/home/cparr/surfaces/level_1_surfaces/hv/bare_earth/',4400,4800,475,555)
bare = master['2012_158']
del master['2012_158']

def render_mpl_table(data, roi_name, col_width=3.0, row_height=0.5, font_size=14,
                     header_color='#236192', row_colors=['#C7C9C7', 'w'],
                     edge_color='w',bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText = data.values, rowLabels = data.index,
                         bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='#FFCD00')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    plt.savefig('/home/cparr/Snow_Patterns/figures/'+roi_name+'_table.png',
        dpi = 300)
    return ax

def make_roi_map(roi_name, cmin, cmax):
    
    fig, axs = plt.subplots(1,4, figsize=(8, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(top = 0.85, bottom = 0.14, wspace = 0.1)
      
    axs = axs.ravel()
    yrs = [k for k in master.iterkeys()]
    yrs.sort()  
    i = 0
    for y in yrs:
        
        im = axs[i].imshow(master[y][y+'_roi'], cmap = 'viridis', vmin=cmin, vmax=cmax)
        axs[i].axhline(y=250, color ='r') # to plot a transect comment this in or out
        axs[i].set_title(str(y), fontsize = 12)
        
        if i == 0:
            axs[i].set_yticks( [0,100,200,300] )
            axs[i].set_xticks( [0, 25,50,75] )
            axs[i].set_xticklabels(['0','50','100','150'], size = 7)
            axs[i].set_yticklabels(['0','200','400','600'],size = 7)
            axs[i].set_ylabel('m',size = 7)
            axs[i].set_xlabel('m',size = 7)
            i+=1
        else:
            axs[i].set_yticklabels( [] )
            axs[i].set_xticklabels( [] )
            i+=1

    plt.suptitle('Snow Depth [m]',size=14)
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')

    plt.savefig('/home/cparr/Snow_Patterns/figures/'+roi_name+'_roi_map.png',
            dpi = 300)
    
make_roi_map('lake_cutbank',0,2.5)

def make_slice(y_transect):
    yrs = [k for k in master.iterkeys()]
    for y in yrs:
        
        master[y][y+'_roi_transect'] = master[y][y+'_roi'][y_transect]
    bare['transect']=bare['2012_158_roi'][y_transect]
                
make_slice(250)

# plot surfaces
plt.plot(master['2013_103']['2013_103_roi_transect']+bare['transect'],label='Winter 2013')
plt.plot(master['2016_096']['2016_096_roi_transect']+bare['transect'],label='Winter 2016')
ax = plt.plot(bare['transect'],color = 'k',lw=2.5,label='Bare Earth')
plt.xlabel('meters')
plt.ylabel('elevation [m]')
plt.legend()
ticks = range(0,80,10)
labels = [str(n*2) for n in ticks]
plt.xticks(ticks,labels)
plt.tight_layout()
plt.savefig('/home/cparr/Snow_Patterns/figures/transect_2013_2016.png',dpi = 300)
plt.close()
# plot surfaces
plt.plot(master['2012_107']['2012_107_roi_transect']+bare['transect'],label='Winter 2012')
plt.plot(master['2015_096']['2015_096_roi_transect']+bare['transect'],label='Winter 2015')
ax = plt.plot(bare['transect'],color = 'k',lw=2.5,label='Bare Earth')
plt.xlabel('meters')
plt.ylabel('elevation [m]')
plt.legend()
ticks = range(0,80,10)
labels = [str(n*2) for n in ticks]
plt.xticks(ticks,labels)
plt.tight_layout()
plt.savefig('/home/cparr/Snow_Patterns/figures/transect_2012_2015.png',dpi = 300)

#plot all profiles
plt.plot(master['2012_107']['2012_107_roi_transect']+bare['transect'],label='Winter 2012')
plt.plot(master['2013_103']['2013_103_roi_transect']+bare['transect'],label='Winter 2013')
plt.plot(master['2015_096']['2015_096_roi_transect']+bare['transect'],label='Winter 2015')
plt.plot(master['2016_096']['2016_096_roi_transect']+bare['transect'],label='Winter 2016')
ax = plt.plot(bare['transect'],color = 'k',lw=2.5,label='Bare Earth')
plt.xlabel('meters')
plt.ylabel('elevation [m]')
plt.legend()
ticks = range(0,80,10)
labels = [str(n*2) for n in ticks]
plt.xticks(ticks,labels)
plt.tight_layout()
plt.savefig('/home/cparr/Snow_Patterns/figures/all_transects.png',dpi = 300)


def make_score_map_figure(roi_name, score, cmin, cmax):
    
    fig, axs = plt.subplots(1,6, figsize=(8, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(top = 0.85, bottom = 0.14)
    
    axs = axs.ravel()
    pairs = [k for k in compare_years_l1.iterkeys()]  
    i = 0
    for p in pairs:
        
        im = axs[i].imshow(compare_years_l1[p][score], cmap = 'viridis', vmin=cmin, vmax=cmax)
        axs[i].set_title(p + '\n'+score[:-3] + '= ' + str(compare_years_l1[p][score[:-4]]),
            fontsize = 7)
        if i == 0:
            axs[i].set_yticks( [0,100,200,300] )
            axs[i].set_xticks( [0, 25,50,75] )
            axs[i].set_xticklabels(['0','50','100','150'], size = 7)
            axs[i].set_yticklabels(['0','200','400','600'],size = 7)
            axs[i].set_ylabel('m',size = 7)
            axs[i].set_xlabel('m',size = 7)
            i+=1
        else:
            axs[i].set_yticklabels( [] )
            axs[i].set_xticklabels( [] )
            i+=1


    plt.suptitle(score + ' Comparisons',size=12)
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')

    plt.savefig('/home/cparr/Snow_Patterns/figures/'+roi_name+'_'+score+'.png',
            dpi = 300)

make_score_map_figure('lake_cutbank', 'MSE Map', 0,1)
make_score_map_figure('lake_cutbank', 'SSIM Map',-1,1)
make_score_map_figure('lake_cutbank', 'CW-SSIM Map',0,1)
make_score_map_figure('lake_cutbank', 'GMSD Map',0,1)


def make_dct_graph(roi_name):
    
    dct_2016 = compare_years_l1['2015 vs. 2016']['DCT Graph 2016_096']
    dct_2015 = compare_years_l1['2015 vs. 2016']['DCT Graph 2015_096']
    dct_2013 = compare_years_l1['2013 vs. 2012']['DCT Graph 2013_103']
    dct_2012 = compare_years_l1['2013 vs. 2012']['DCT Graph 2012_107']
    
    plt.plot(dct_2012, lw = 2, label = '2012')
    plt.plot(dct_2013, lw = 2, label = '2013')
    plt.plot(dct_2015, lw = 2, label = '2015')
    plt.plot(dct_2016, lw = 2, label = '2016')
    plt.legend()
    
    plt.savefig('/home/cparr/Snow_Patterns/figures/'+roi_name+'_dct.png',
        dpi = 300)
    
make_dct_graph('lake_cut_bank')


# make df to render table
df = pd.DataFrame.from_dict(compare_years_l1)
df = df.T
df = df[['MSE','SSIM','CW-SSIM','FSIM','GMSD']]
df['MSE Rank'] = df['MSE'].rank(ascending = True)
df['SSIM Rank'] = df['SSIM'].rank(ascending = False)
df['CW-SSIM Rank'] = df['CW-SSIM'].rank(ascending = False)
df['FSIM Rank'] = df['FSIM'].rank(ascending = False)
df['GMSD Rank'] = df['GMSD'].rank(ascending = True)
df = df.sort_values(['MSE'])
ranks = df[['MSE Rank', 'SSIM Rank', 'CW-SSIM Rank', 'FSIM Rank', 'GMSD Rank']]
score_df = df[['MSE','SSIM','CW-SSIM','FSIM','GMSD']]
render_mpl_table(score_df, 'lake_cutbank_scores')
render_mpl_table(ranks, 'lake_cutbank_ranks')
#



