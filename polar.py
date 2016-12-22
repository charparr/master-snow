#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:40:52 2016

@author: cparr
"""

import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate

# Read and clean 2011 speed, direction, and RH

fr_wind11 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2011m.csv', header = 25)
rng = pd.date_range('1/1/2011', periods=8760, freq='H')
fr_wind11 = fr_wind11.set_index(rng)

fr_wind11['10 m Wind Speed'] = fr_wind11['10 meter wind speed (m/s)']
fr_wind11['10 m Wind Direction'] = fr_wind11['Wind direction TN']
fr_wind11['10 m RH %'] = fr_wind11[' 10 meter relative humidity (%)']
fr_wind11 = fr_wind11[['10 m Wind Direction','10 m Wind Speed','10 m RH %']]
fr_wind11 = fr_wind11.replace(7777, 'nan')
fr_wind11_daily = pd.DataFrame()
fr_wind11_daily = fr_wind11.resample('D').mean()

# Read and clean 2012 speed, direction, and RH

fr_wind12 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2012m.csv', header = 7)
fr_wind12 = fr_wind12[2::]
rng = pd.date_range('1/1/2012', periods=8784, freq='H')
fr_wind12 = fr_wind12.set_index(rng)

fr_wind12['10 m Wind Speed'] = fr_wind12['WS_10m_WVc(1)']
fr_wind12['10 m Wind Direction'] = fr_wind12['WS_10m_WVc(2)']
fr_wind12['10 m RH %'] = fr_wind12['RH_10m_Avg']
fr_wind12 = fr_wind12[['10 m Wind Direction','10 m Wind Speed','10 m RH %']]
fr_wind12['10 m Wind Direction'] = fr_wind12['10 m Wind Direction'].astype(float)
fr_wind12['10 m Wind Speed'] = fr_wind12['10 m Wind Speed'].astype(float)
fr_wind12['10 m RH %'] = fr_wind12['10 m RH %'].astype(float)
fr_wind12_daily = pd.DataFrame()
fr_wind12_daily = fr_wind12.resample('D').mean()

# Read and clean 2013 speed, direction, and RH

fr_wind13 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2013m.csv', header = 5)
fr_wind13 = fr_wind13[2::]
rng = pd.date_range('1/1/2013', periods=8760, freq='H')
fr_wind13 = fr_wind13.set_index(rng)

fr_wind13['10 m Wind Speed'] = fr_wind13['WS_10m_WVc(1)']
fr_wind13['10 m Wind Direction'] = fr_wind13['WS_10m_WVc(2)']
fr_wind13['10 m RH %'] = fr_wind13['RH_10m_Avg']
fr_wind13 = fr_wind13[['10 m Wind Direction','10 m Wind Speed','10 m RH %']]
fr_wind13['10 m Wind Direction'] = fr_wind13['10 m Wind Direction'].astype(float)
fr_wind13['10 m Wind Speed'] = fr_wind13['10 m Wind Speed'].astype(float)
fr_wind13['10 m RH %'] = fr_wind13['10 m RH %'].astype(float)
fr_wind13_daily = pd.DataFrame()
fr_wind13_daily = fr_wind13.resample('D').mean()

# Read and clean 2014 speed, direction, and RH

fr_wind14 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2014m.csv', header = 7)
fr_wind14 = fr_wind14[2::]
rng = pd.date_range('1/1/2014', periods=8760, freq='H')
fr_wind14 = fr_wind14.set_index(rng)

fr_wind14['10 m Wind Speed'] = fr_wind14['WS_10m_WVc(1)']
fr_wind14['10 m Wind Direction'] = fr_wind14['WS_10m_WVc(2)']
fr_wind14['10 m RH %'] = fr_wind14['RH_10m_Avg']
fr_wind14 = fr_wind14[['10 m Wind Direction','10 m Wind Speed','10 m RH %']]
fr_wind14['10 m Wind Direction'] = fr_wind14['10 m Wind Direction'].astype(float)
fr_wind14['10 m Wind Speed'] = fr_wind14['10 m Wind Speed'].astype(float)
fr_wind14['10 m RH %'] = fr_wind14['10 m RH %'].astype(float)
fr_wind14_daily = pd.DataFrame()
fr_wind14_daily = fr_wind14.resample('D').mean()

# Read and clean 2015 speed, direction, and RH

fr_wind15 = pd.read_csv('/home/cparr/Snow_Patterns/wind_data/fr2015m.csv', header = 19)
rng = pd.date_range('1/1/2015', periods=14126, freq='H')
fr_wind15 = fr_wind15.set_index(rng)

fr_wind15['10 m Wind Speed'] = fr_wind15['windSpeed_10m']
fr_wind15['10 m Wind Direction'] = fr_wind15['winddir_10m']
fr_wind15['10 m RH %'] = fr_wind15['rh_3m'] # no 10 m rh
fr_wind15 = fr_wind15[['10 m Wind Direction','10 m Wind Speed','10 m RH %']]
fr_wind15['10 m Wind Direction'] = fr_wind15['10 m Wind Direction'].astype(float)
fr_wind15['10 m Wind Speed'] = fr_wind15['10 m Wind Speed'].astype(float)
fr_wind15['10 m RH %'] = fr_wind15['10 m RH %'].astype(float)
fr_wind15_daily = pd.DataFrame()
fr_wind15_daily = fr_wind15.resample('D').mean()

# Hourly and daily winter seasons

winter_11_12d = fr_wind11_daily.ix['9/1/2011'::].append(fr_wind12_daily.ix['1/1/2012':'4/15/2012'])
winter_12_13d = fr_wind12_daily.ix['9/1/2012'::].append(fr_wind13_daily.ix['1/1/2013':'4/15/2013'])
winter_13_14d = fr_wind13_daily.ix['9/1/2013'::].append(fr_wind14_daily.ix['1/1/2014':'4/15/2014'])
winter_14_15d = fr_wind14_daily.ix['9/1/2014'::].append(fr_wind15_daily.ix['1/1/2015':'4/15/2015'])
winter_15_16d = fr_wind15_daily.ix['9/1/2015':'4/15/2016']


winter_11_12h = fr_wind11.ix['9/1/2011'::].append(fr_wind12.ix['1/1/2012':'4/15/2012'])
winter_12_13h = fr_wind12.ix['9/1/2012'::].append(fr_wind13.ix['1/1/2013':'4/15/2013'])
winter_13_14h = fr_wind13.ix['9/1/2013'::].append(fr_wind14.ix['1/1/2014':'4/15/2014'])
winter_14_15h = fr_wind14.ix['9/1/2014'::].append(fr_wind15.ix['1/1/2015':'4/15/2015'])
winter_15_16h = fr_wind15.ix['9/1/2015':'4/15/2016']
# mask out days where mean ws is less than 5 m/s

w1112_g5ms_d = winter_11_12d[winter_11_12d['10 m Wind Speed'] >= 5]
w1213_g5ms_d = winter_12_13d[winter_12_13d['10 m Wind Speed'] >= 5]
w1314_g5ms_d = winter_13_14d[winter_13_14d['10 m Wind Speed'] >= 5]
w1415_g5ms_d = winter_14_15d[winter_14_15d['10 m Wind Speed'] >= 5]
w1516_g5ms_d = winter_15_16d[winter_15_16d['10 m Wind Speed'] >= 5]

w1112_g5ms_h = winter_11_12h[winter_11_12h['10 m Wind Speed'] >= 5]
w1213_g5ms_h = winter_12_13h[winter_12_13h['10 m Wind Speed'] >= 5]
w1314_g5ms_h = winter_13_14h[winter_13_14h['10 m Wind Speed'] >= 5]
w1415_g5ms_h = winter_14_15h[winter_14_15h['10 m Wind Speed'] >= 5]
w1516_g5ms_h = winter_15_16h[winter_15_16h['10 m Wind Speed'] >= 5]

#integrating hourly wind data

def integrate_method(self, how='trapz', unit='s'):
    '''Numerically integrate the time series.

    @param how: the method to use (trapz by default)
    @return 

    Available methods:
     * trapz - trapezoidal
     * cumtrapz - cumulative trapezoidal
     * simps - Simpson's rule
     * romb - Romberger's rule

    See http://docs.scipy.org/doc/scipy/reference/integrate.html for the method details.
    or the source code
    https://github.com/scipy/scipy/blob/master/scipy/integrate/quadrature.py
    '''
    available_rules = set(['trapz', 'cumtrapz', 'simps', 'romb'])
    if how in available_rules:
        rule = integrate.__getattribute__(how)
    else:
        print('Unsupported integration rule: %s' % (how))
        print('Expecting one of these sample-based integration rules: %s' % (str(list(available_rules))))
        raise AttributeError

    result = rule(self.values, self.index.astype(np.int64) / 10**9)
    #result = rule(self.values)
    return result

pd.TimeSeries.integrate = integrate_method

hourly_ws_ts_2011_2012 = pd.TimeSeries(w1112_g5ms_h['10 m Wind Speed'].copy())
hourly_ws_ts_2012_2013 = pd.TimeSeries(w1213_g5ms_h['10 m Wind Speed'].copy())
hourly_ws_ts_2013_2014 = pd.TimeSeries(w1314_g5ms_h['10 m Wind Speed'].copy())
hourly_ws_ts_2014_2015 = pd.TimeSeries(w1415_g5ms_h['10 m Wind Speed'].copy())
hourly_ws_ts_2015_2016 = pd.TimeSeries(w1516_g5ms_h['10 m Wind Speed'].copy())
tseries = [hourly_ws_ts_2011_2012, hourly_ws_ts_2012_2013, hourly_ws_ts_2013_2014, hourly_ws_ts_2014_2015, hourly_ws_ts_2015_2016]

integral_wind = []
yrs = [range(2012,2017)]

def integrate_wind():
    for t in tseries:
        integral_wind.append(t.integrate(how='simps', unit='h'))

integrate_wind()    
winddf = pd.DataFrame()

winddf['Seasonal Wind Integration']= integral_wind
winddf.set_index(yrs,inplace = True)
y = pd.Series(range(2012,2017))
winddf['Year'] = y.values

ax = sns.barplot(x="Year", y="Seasonal Wind Integration", data=winddf)
ax.set_yscale('log')

def compare_wind(yr1data, yr2data, stryr1, stryr2):

    wind_ticks = [5,10,15,20]
    az_ticks = [0,90,180,270,360]
    
    ax1 = plt.subplot(2,1,1)
    plt.plot(yr1data['10 m Wind Speed'])
    plt.title(stryr1, fontsize = 12)
    plt.ylim(5,20)
    plt.ylabel('m/s')
    plt.yticks(wind_ticks)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    for tx in ax1.get_xticklabels():
        tx.set_fontsize(7)
    
    ax2 = ax1.twinx()
    ax2.plot(yr1data['10 m Wind Direction'], color ='r',alpha = 0.66, linestyle = '--')
    plt.ylim(0,360)
    plt.yticks(az_ticks)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    
    ax3 = plt.subplot(2,1,2)
    plt.plot(yr2data['10 m Wind Speed'])
    plt.title(stryr2, fontsize = 12)
    plt.ylim(5,20)
    plt.ylabel('m/s')
    plt.yticks(wind_ticks)
    for tl in ax3.get_yticklabels():
        tl.set_color('b')
    for tx in ax3.get_xticklabels():
        tx.set_fontsize(7)
    
    ax4 = ax3.twinx()
    ax4.plot(yr2data['10 m Wind Direction'], color ='r',alpha = 0.66, linestyle = '--')
    plt.ylim(0,360)
    plt.yticks(az_ticks)
    for tl in ax4.get_yticklabels():
        tl.set_color('r')

    plt.subplots_adjust(top = 0.85, hspace = 0.5)
    plt.suptitle('Franklin Bluffs Mean Daily Wind Speed & Direction')
    plt.savefig('/home/cparr/Snow_Patterns/figures/fb_wind_'+
                stryr1+'_'+stryr2+'.png', dpi = 300, bbox_inches = 'tight')
    
compare_wind(winter_11_12d, winter_14_15d, '2012', '2015' )
compare_wind(winter_12_13d, winter_15_16d, '2013', '2016' )

def compare_rh(yr1data, yr2data, stryr1, stryr2):
    
    wind_ticks = [5,10,15,20]
    h_ticks = [50,60,70,80,90,100]
    
    ax1 = plt.subplot(2,1,1)
    plt.plot(yr1data['10 m Wind Speed'])
    plt.title(stryr1, fontsize = 12)
    plt.ylim(5,20)
    plt.ylabel('m/s')
    plt.yticks(wind_ticks)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    for tx in ax1.get_xticklabels():
        tx.set_fontsize(7)
    
    ax2 = ax1.twinx()
    ax2.plot(yr1data['10 m RH %'], color ='g',alpha = 0.66, linestyle = '--')
    plt.ylim(50,100)
    plt.yticks(h_ticks)
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    
    ax3 = plt.subplot(2,1,2)
    plt.plot(yr2data['10 m Wind Speed'])
    plt.title(stryr2, fontsize = 12)
    plt.ylim(5,20)
    plt.ylabel('m/s')
    plt.yticks(wind_ticks)
    for tl in ax3.get_yticklabels():
        tl.set_color('b')
    for tx in ax3.get_xticklabels():
        tx.set_fontsize(7)
    
    ax4 = ax3.twinx()
    ax4.plot(yr2data['10 m RH %'], color ='g',alpha = 0.66, linestyle = '--')
    plt.ylim(50,100)
    plt.yticks(h_ticks)
    for tl in ax4.get_yticklabels():
        tl.set_color('g')

    plt.subplots_adjust(top = 0.85, hspace = 0.5)
    plt.suptitle('Franklin Bluffs Mean Daily Wind Speed & Direction')
    plt.savefig('/home/cparr/Snow_Patterns/figures/fb_wind_rh_'+
                stryr1+'_'+stryr2+'.png', dpi = 300, bbox_inches = 'tight')

compare_rh(winter_11_12d, winter_14_15d, '2012', '2015' )    



#plt.subplots_adjust(hspace = 0.3)
#plt.suptitle('Franklin Bluffs Daily Average Wind Speed & Direction')
#plt.savefig('/home/cparr/Snow_Patterns/figures/fb_wind_2012vs2015.png',dpi = 300)
#############
#fig = plt.figure()
#plt.title('Franklin Bluffs Winter 2011/2012 Daily Average Wind')
#ax1 = fig.add_subplot(111)
#ax1.plot(winter_11_12['Wind Direction'], 'r', lw = 2, alpha = 0.5)
#plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
#ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
#ax1.set_ylim([0,360])
#ax1.set_yticks([0,90,180,270,360])
#
#ax2 = ax1.twinx()
#ax2.set_xlabel([])
#ax2.plot(winter_11_12['3m Wind Speed'], lw = 2)
#ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
#ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
#ax2.set_ylim([5,13])
#
#plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_11_12.png',dpi = 300)
#
#############
#
#fig = plt.figure()
#plt.title('Franklin Bluffs Winter 2012/2013 Daily Average Wind')
#ax1 = fig.add_subplot(111)
#ax1.plot(winter_12_13['Wind Direction'], 'r', lw = 2, alpha = 0.5)
#plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
#ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
#ax1.set_ylim([0,360])
#ax1.set_yticks([0,90,180,270,360])
#
#ax2 = ax1.twinx()
#ax2.set_xlabel([])
#ax2.plot(winter_12_13['3m Wind Speed'], lw = 2)
#ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
#ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
#ax2.set_ylim([5,19])
#plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_12_13.png',dpi = 300)
#
#############
#
#fig = plt.figure()
#plt.title('Franklin Bluffs Winter 2013/2014 Daily Average Wind')
#ax1 = fig.add_subplot(111)
#ax1.plot(winter_13_14['Wind Direction'], 'r', lw = 2, alpha = 0.5)
#plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
#ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
#ax1.set_ylim([0,360])
#ax1.set_yticks([0,90,180,270,360])
#
#ax2 = ax1.twinx()
#ax2.set_xlabel([])
#ax2.plot(winter_13_14['3m Wind Speed'], lw = 2)
#ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
#ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
#ax2.set_ylim([5,19])
#plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_13_14.png',dpi = 300)
#
#      
#############
#
#fig = plt.figure()
#plt.title('Franklin Bluffs Winter 2014/2015 Daily Average Wind')
#ax1 = fig.add_subplot(111)
#ax1.plot(winter_14_15['Wind Direction'], 'r', lw = 2, alpha = 0.3)
#plt.axhline(y=180, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=270, color='k', linestyle='--', alpha = 0.5)
#plt.axhline(y=90, color='k', linestyle='--', alpha = 0.5)
#ax1.set_ylabel('Wind Direction (deg. N.)', color='r')
#ax1.set_ylim([0,360])
#ax1.set_yticks([0,90,180,270,360])
#
#ax2 = ax1.twinx()
#ax2.set_xlabel([])
#ax2.plot(winter_14_15['3m Wind Speed'], lw = 2)
#ax2.set_xticklabels(['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'])
#ax2.set_ylabel('3m Wind Speed (m/s)', color='b')
#ax2.set_ylim([5,18])
#plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_14.png',dpi = 300)
#
#
#def main():
#    azi = wd_11_12
#    z = ws_11_12
#
#    plt.figure(figsize=(5,6))
#    plt.subplot(111, projection='polar')
#    coll = rose(azi, z=z, bidirectional=True)
#    plt.xticks(np.radians(range(0, 360, 45)), 
#               ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
#    plt.colorbar(coll, orientation='horizontal')
#    plt.xlabel('2011 - 2012 3m Wind rose colored by mean wind speed')
#    plt.rgrids(range(5, 20, 5), angle=290)
#
#    plt.savefig('/home/cparr/Snow_Patterns/figures/wind/winter_11_12.png',dpi = 300)
#
#    plt.show()
#    
#def rose(azimuths, z=None, ax=None, bins=36, bidirectional=False, 
#         color_by=np.mean, **kwargs):
#    """Create a "rose" diagram (a.k.a. circular histogram).  
#
#    Parameters:
#    -----------
#        azimuths: sequence of numbers
#            The observed azimuths in degrees.
#        z: sequence of numbers (optional)
#            A second, co-located variable to color the plotted rectangles by.
#        ax: a matplotlib Axes (optional)
#            The axes to plot on. Defaults to the current axes.
#        bins: int or sequence of numbers (optional)
#            The number of bins or a sequence of bin edges to use.
#        bidirectional: boolean (optional)
#            Whether or not to treat the observed azimuths as bi-directional
#            measurements (i.e. if True, 0 and 180 are identical).
#        color_by: function or string (optional)
#            A function to reduce the binned z values with. Alternately, if the
#            string "count" is passed in, the displayed bars will be colored by
#            their y-value (the number of azimuths measurements in that bin).
#        Additional keyword arguments are passed on to PatchCollection.
#
#    Returns:
#    --------
#        A matplotlib PatchCollection
#    """
#    azimuths = np.asanyarray(azimuths)
#    if color_by == 'count':
#        z = np.ones_like(azimuths)
#        color_by = np.sum
#    if ax is None:
#        ax = plt.gca()
#    ax.set_theta_direction(-1)
#    ax.set_theta_offset(np.radians(90))
#    if bidirectional:
#        other = azimuths + 180
#        azimuths = np.concatenate([azimuths, other])
#        if z is not None:
#            z = np.concatenate([z, z])
#    # Convert to 0-360, in case negative or >360 azimuths are passed in.
#    azimuths[azimuths > 360] -= 360
#    azimuths[azimuths < 0] += 360
#    counts, edges = np.histogram(azimuths, range=[0, 360], bins=bins)
#    if z is not None:
#        idx = np.digitize(azimuths, edges)
#        z = np.array([color_by(z[idx == i]) for i in range(1, idx.max() + 1)])
#        z = np.ma.masked_invalid(z)
#    edges = np.radians(edges)
#    coll = colored_bar(edges[:-1], counts, z=z, width=np.diff(edges), 
#                       ax=ax, **kwargs)
#    return coll
#
#def colored_bar(left, height, z=None, width=0.8, bottom=0, ax=None, **kwargs):
#    """A bar plot colored by a scalar sequence."""
#    if ax is None:
#        ax = plt.gca()
#    width = itertools.cycle(np.atleast_1d(width))
#    bottom = itertools.cycle(np.atleast_1d(bottom))
#    rects = []
#    for x, y, h, w in zip(left, bottom, height, width):
#        rects.append(Rectangle((x,y), w, h))
#    coll = PatchCollection(rects, array=z, **kwargs)
#    ax.add_collection(coll)
#    ax.autoscale()
#    return coll
#
#if __name__ == '__main__':
#    main()
    
