import numpy as np
from numpy import log as ln
import matplotlib
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
#import scipy.spatial as sp
#import math
#import os
#import scipy.signal as ss
#import sys

isq2pi = 1/(2*np.pi)**(0.5)


def histogram(on_data,off_data,bin_master):
    fig = plt.figure(figsize=(20,6))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    hy,hx,_=plt.hist(on_data,bins=bin_master,alpha=.3,label='on-pulse data',color='green')
    hy,hx,_=plt.hist(off_data,bins=bin_master,alpha=.3,label='off-pulse data',color='darkorange')

    fig.text(0.07, 0.5, 'Count', ha='center', va='center', rotation='vertical', size=22)
    fig.text(0.504, 0.02, 'I/<I>', ha='center', va='center', rotation='horizontal', size=22)
    plt.legend(prop={'size': 20},loc='upper left')
    plt.grid()

    fig.suptitle('Histogram of Flux Density in On- and Off-Windows', fontsize=30)

    return fig

def histogram_neg(on_data,off_data,bin_master):
    fig = plt.figure(figsize=(20,6))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    hy_on,hx_on,_=plt.hist(on_data,bins=bin_master,alpha=.3,label='on-pulse data',color='green')
    hy_off,hx_off,_=plt.hist(off_data,bins=bin_master,alpha=.3,label='off-pulse data',color='darkorange')

    fig.text(0.07, 0.5, 'Count', ha='center', va='center', rotation='vertical', size=22)
    fig.text(0.504, 0.02, 'I/<I>', ha='center', va='center', rotation='horizontal', size=22)
    plt.legend(prop={'size': 20},loc='upper left')
    plt.xlim(right=0)
    plt.grid()

    fig.suptitle('Histogram with Negative Values of Flux Density Only', fontsize=30)    

    return fig, hy_on, hy_off, 
    
def scaled_hist(x, y_off, y_on, best_scaling):
    fig, ax = plt.subplots(figsize =(20, 6))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.fill_between(x, y_off, step="post", alpha=0.3, color='darkorange',label='off-pulse data')
    plt.fill_between(x, y_on, step="post", alpha=0.3, color='green',label='on-pulse data')
    plt.grid()
    fig.text(0.07, 0.5, 'Count', ha='center', va='center', rotation='vertical', size=22)
    fig.text(0.504, 0.02, 'Bins', ha='center', va='center', rotation='horizontal', size=22)
    plt.legend(prop={'size': 20},loc='upper left')

    plt.ylim(bottom=0)

    fig.suptitle('Histogram After On-Window Data Has Been Scaled By {:.3f}'.format(best_scaling), fontsize=30)    

def bayes_hist(hist_data, x_func, tot_func, on_func, null_func):
    fig = plt.figure(figsize=(20,6))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    hy,hx,_=plt.hist(hist_data,50,alpha=.3,label='data',color='gray')
    peak_hist = np.max(hy)
    peak_function = np.max(tot_func)
    peak_ratio = peak_function/peak_hist
    plt.plot(x_func, tot_func/peak_ratio,'k--',linewidth=3)
    plt.plot(x_func, null_func/peak_ratio,'b-',linewidth=1)
    plt.plot(x_func, on_func/peak_ratio,'r-',linewidth=1)
    hx=(hx[1:]+hx[:-1])/2 # for len(x)==len(y)
    
    plt.grid()
    fig.text(0.07, 0.5, 'Count', ha='center', va='center', rotation='vertical', size=22)
    fig.text(0.504, 0.02, 'I/<I>', ha='center', va='center', rotation='horizontal', size=22)
    fig.suptitle('Histogram of On-Window Flux Density with Recovered Bayesian Parameters', fontsize=30)    

    return fig

def waterfall_and_average(data,mean,windows):
    fig = plt.figure(figsize=(20,8))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    ax1 = plt.subplot2grid((3,1),(0,0),rowspan = 2)                                                                                                                                                          
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False, labeltop=False)                                                                                                                                              
    ax1.imshow(data[:,:], aspect='auto',cmap='binary',origin='lower')
    fig.text(0.07, 0.645, 'Pulse Number', ha='center', va='center', rotation='vertical', size=18)
    fig.text(0.07, 0.25, 'Normalized\nFlux Density', ha='center', va='center', rotation='vertical', size=18)
    fig.text(0.504, 0.03, 'Phase Bin', ha='center', va='center', rotation='horizontal', size=18)
    ax2 = plt.subplot2grid((3,1),(2,0))
    ax2.plot(mean,color='k')
    ax2.set_xlim(0,data.shape[1])
    ax2.set_ylim(-0.1,1.1)
    ax2.grid()
    plt.vlines(windows[0],-0.1,1.1,linestyles='dashed')
    plt.vlines(windows[1],-0.1,1.1,linestyles='dashed')
    ax2.add_patch(Rectangle((windows[0],-0.1),(windows[1]-windows[0]),1.2,facecolor='green',alpha = 0.1, zorder = 39))
    plt.vlines(windows[2],-0.1,1.1,linestyles='dashed')                                                                                                                                                  
    plt.vlines(windows[3],-0.1,1.1,linestyles='dashed')                                                                                                                                                    
    ax2.add_patch(Rectangle((windows[2],-0.1),(windows[3]-windows[2]),1.2,facecolor='darkorange',alpha = 0.1, zorder = 39))                                                                    
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.0)

    fig.suptitle('All Profiles and Average Profile', fontsize=30)
    return fig;
    


def logL(params, data):
    print('called from here')
    """likelihood"""
    Noff, u, sig = params
    
    N = len(data)
    
    norm = isq2pi/N
    
    norm_on = (N-Noff)/sig
    arg_on = -0.5 * ((data-u)/sig)**2
    norm_off = Noff/std_of_nulls
    arg_off = -0.5 * ((data-mean_of_nulls)/std_of_nulls)**2
    
    logLs = np.log(norm * (norm_on*np.exp(arg_on) + norm_off*np.exp(arg_off)))
    
    return np.sum(logLs)

def logL_lognormal(params, data):
    """likelihood"""
    Noff, u, sig = params

    N = len(data)
    
    pos_data = []
    
    # take positive numbers only because it's lognormal - replace -ve numbers with huge number which the will have a low likelihood of coming from the log_normal.
    for each in data:
        if each > 0:
            pos_data.append(each)
        else:
            pos_data.append(1e50)
    
    pos_data = np.array(pos_data)
    
    norm = isq2pi/N
    
    #print('N, Noff, N-Noff',N, Noff, N-Noff)
    
    norm_on = (1/pos_data) * (N-Noff)/sig
    arg_on = -0.5 * ((ln(pos_data)-u)/sig)**2
    norm_off = Noff/std_of_nulls
    arg_off = -0.5 * ((data-mean_of_nulls)/std_of_nulls)**2
    
    arg_for_log = norm * (norm_on*np.exp(arg_on) + norm_off*np.exp(arg_off))
       
    #where_are_NaNs = np.isnan(arg_for_log)
    #arg_for_log[where_are_NaNs] = 1.0
    
    logLs = np.log(arg_for_log)
    
    return np.sum(logLs)

def logL_lognormal_conv(params, data, mean_off_window_nulls, std_off_window_nulls):
    
    """likelihood"""
    
    # do the convolution
    
    total_function_x, total_function_y, _, _ = convolve(params, data, mean_off_window_nulls, std_off_window_nulls)

    
    N = len(data)    
    
    # now put the data values into the function
    
    likelihood = []
    
    if params[0] > N or params[1] < 0 or params[2] < 0:
    
        sum_out = -1e50
    
    else:
        
        for each in data:
            difference_array = np.absolute(total_function_x-each)
            index = difference_array.argmin()
            likelihood.append(total_function_y[index])
    
        logLs = np.log(likelihood)
    
        sum_out = np.sum(logLs)

    return sum_out

def convolve(params, data,  mean_off_window_nulls, std_off_window_nulls):
    """
    Convolves the lognormal function with the noise in the data. 
    """
        
    Noff, u, sig = params
    
    N = len(data)    

    # the resolution of the of the functions
    
    function_resolution = 0.01
    
    # the range of the functions
    
    if abs(np.min(data)) > abs(np.max(data)):
        set_range = abs(np.min(data))
    else:
        set_range = abs(np.max(data))

    
    # the off on and conv x-range
    
    off_func_x = np.arange(-set_range,set_range,function_resolution)
    null_func_x = np.arange(-set_range,set_range,function_resolution)
    on_func_x = np.arange(function_resolution,set_range,function_resolution)

    conv_func_x = np.arange(-set_range,(2.0*set_range)-function_resolution,function_resolution)
    
    # the y values after the function has acted
    
    
    func_arg_off = -0.5 * ((off_func_x-mean_off_window_nulls)/std_off_window_nulls)**2
    off_func_y_for_conv = 1*np.exp(func_arg_off)
    #off_func_y_for_conv = np.ones((func_arg_off.shape[0]))
    
    func_norm_null = Noff/std_off_window_nulls
    func_arg_null = -0.5 * ((null_func_x-mean_off_window_nulls)/std_off_window_nulls)**2
    #func_arg_null = -0.5 * ((null_func_x-uoff)/std_off_window_nulls)**2
    func_norm_on = (1/on_func_x) * (N-Noff)/sig
    func_arg_on = -0.5 * ((ln(on_func_x)-u)/sig)**2
    
    
    null_func_y = func_norm_null*np.exp(func_arg_null)
    on_func_y_for_conv = func_norm_on*np.exp(func_arg_on)
    
    # do the convolution

    conv_func_y = np.convolve(on_func_y_for_conv,off_func_y_for_conv,mode='full')/(np.sum(on_func_y_for_conv))
    
    area_on = np.sum(on_func_y_for_conv)
    area_conv = np.sum(conv_func_y)
    
    conv_func_y *= (area_on/area_conv)
    
    # pad the on function now so that it can be plotted nicely with off function
    
    on_func_y_expanded = np.zeros((off_func_y_for_conv.shape))
    
    if on_func_y_expanded.shape[0] - (2*on_func_y_for_conv.shape[0]) == 2: # to stop it sometimes not having same length as other arrays
        on_func_y_expanded[on_func_y_for_conv.shape[0]+2:] = on_func_y_for_conv
    elif on_func_y_expanded.shape[0] - (2*on_func_y_for_conv.shape[0]) == 1:
        on_func_y_expanded[on_func_y_for_conv.shape[0]+1:] = on_func_y_for_conv
    else:
        sys.exit()
    
    # now null function must be expaned to be added to the conv function
    
    null_func_y_expanded = np.zeros((conv_func_y.shape))
    null_func_y_expanded[:null_func_y.shape[0]] = null_func_y
    
    # add the conv and the off:
    
    total_function_y = null_func_y_expanded + conv_func_y
    total_function_x = conv_func_x

    return total_function_x, total_function_y, null_func_y_expanded, conv_func_y

def logP(params):
    """prior"""
    p_mean = np.array([Nexp, mean_of_on, std_of_on])
    p_std = np.array([1.0e50, 0.5, 0.05]) # effective uniform prior on Noff
    
    norm = isq2pi/p_std
    arg = -0.5 * ((params-p_mean)/p_std)**2
    return np.sum(np.log(norm)+arg, axis=-1)

def logPost(params, data):
    return logL(params, data) + logP(params)

###DEFINE THE GAUSSIAN AND LOGNORMAL FUNCTION###                                                                   

def gauss(x,mu,sigma,A):
    return A*plt.exp(-(x-mu)**2/2./sigma**2)

def lognormal(x,mu,sigma,A):
    return (A*np.exp(-(ln(x)-mu)**2/2./sigma**2)) * (1/x)

### DO THE SCALING ###

def wang(hist_list_on,hist_list_off):
    scaling = 1.0
    if len(hist_list_on) == len(hist_list_off):
        pass
    else:
        print('Different number of bins!')
    smallest_diff = 1e9 # just to make the while loop work
    while np.mean(hist_list_on)*scaling/np.mean(hist_list_off) < 1.5 :
        diff = 0.0
        for i in range(len(hist_list_on)):
            diff += hist_list_off[i]-hist_list_on[i]*scaling
        if abs(diff) < abs(smallest_diff):
            smallest_diff = diff
            best_scaling = scaling
        scaling += 0.001
    return best_scaling
