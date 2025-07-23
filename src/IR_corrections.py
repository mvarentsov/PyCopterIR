import sys, os, pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import  ndimage

from IR_processing_utils import *
import cv2
import lowess


def apply_corr2diff_matrix (corr_matrix, corr):
    img_N = len (corr)
    corr_matrix_new = corr_matrix.copy()
    for i in range (0, img_N):
        for j in range (0, img_N):
            corr_matrix_new[i,j] = corr_matrix[i,j] + corr[i] - corr[j]
    return corr_matrix_new

def circular_filter(image_data, radius):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel = kernel / kernel.sum()
    filtered_image = ndimage.convolve (image_data, kernel)
    return filtered_image     

def run_L0_corr (img_array, img_df, sm_radius, bt_sample_size = 200, bt_repeat_n = 50, pics_dir = None, fig_name = 'L0_corr', reload_pkl = False):
    img_N = img_array.shape[2]
    img_array_new = img_array.copy()
    
    unique_dirs = img_df['folder'].unique()

    for unique_dir in unique_dirs:
        dir_idx = np.where(img_df['folder'] == unique_dir)[0]

        pkl_path = unique_dir + 't0_bt_corrs.pkl'
        
        if not os.path.isfile (pkl_path):
            reload_pkl = True
        if reload_pkl:
            bt_corrs = np.array([])
            for i in tqdm (range (0, bt_repeat_n)):
                sample_idx = np.random.choice(dir_idx.shape[0], size=bt_sample_size, replace=True)
                sample_idx = dir_idx[sample_idx]
                img_array_bt = img_array[:,:,sample_idx]
                bt_corr = img_array_bt.mean(axis=2) - img_array_bt.mean()
                if i == 0:
                    bt_corrs = bt_corr
                else:
                    bt_corrs = np.concatenate((np.atleast_3d(bt_corrs), np.atleast_3d(bt_corr)), axis=2)
            
            try_dump_pkl (bt_corrs, pkl_path)
        else:
            bt_corrs = try_load_pkl (pkl_path)

        
        # mean_corr = img_array_bt[:,:,dir_idx].mean(axis=2) - img_array_bt[:,:,dir_idx].mean()
        
        mean_corr = bt_corrs.mean(axis=2)
        mean_corr_sm = circular_filter (mean_corr, sm_radius)
        
        for i in dir_idx: #range (0, img_N):
            img_array_new[:,:,i]  = img_array_new[:,:,i]  - mean_corr_sm

        f, ax = plt.subplots (1,3, sharex=True, sharey=True, figsize=(10,4))
        plt.subplots_adjust(left = 0.01, bottom = 0.01,top = 0.98, right = 0.98) #, hspace=0.01,wspace=0.01)

        if pics_dir is not None:
            
            plt.axes(ax[0])
            plt.pcolormesh(np.mean(bt_corrs, axis=2), vmin=-0.75, vmax=0.75, cmap = 'seismic')
            plt.gca().set_aspect(1)
            plt.title ('Bootstrap mean')
            
            plt.axes(ax[1])
            plt.pcolormesh(np.std(bt_corrs, axis=2), vmin=-0.75, vmax=0.75, cmap = 'seismic')
            plt.gca().set_aspect(1)
            plt.title ('Bootstrap std')
            
            plt.axes(ax[2])
            plt.pcolormesh(mean_corr_sm, vmin=-0.75, vmax=0.75, cmap = 'seismic')
            plt.gca().set_aspect(1)
            plt.title ('Correction')

            plt.suptitle(unique_dir)
            plt.savefig(pics_dir + fig_name + '_new.png')
        
    return img_array_new

def run_L1_corr (img_array, diff_matrix, use_detrend, pics_dir = None, fig_name = 'L1_corr', opts_str = ''):

    img_N = img_array.shape[2]

    corr_tend = np.array([diff_matrix[i, i-1] if i > 0 else 0 for i in range (0, img_N)])

    L1_corr, L1_diff2prev, outlier_ind = sigma_tend_corr_multi (corr_tend)

    img_array_new = apply_corr2array (img_array, L1_corr)
    diff_matrix_new = apply_corr2diff_matrix (diff_matrix, L1_corr)

    if use_detrend:
        L1_corr_dt      = detrend_corr (img_array_new)
        img_array_new   = apply_corr2array (img_array_new, L1_corr_dt)
        diff_matrix_new = apply_corr2diff_matrix (diff_matrix_new, L1_corr_dt)

    mean_t0 = np.mean(np.mean(img_array,     axis=0), axis=0)
    mean_t1 = np.mean(np.mean(img_array_new, axis=0), axis=0)
    t0_diff_mean = np.diff (mean_t0)
    t1_diff_mean = np.diff (mean_t1)

    corr_tend_new = np.array([diff_matrix_new[i, i-1] if i > 0 else 0 for i in range (0, img_N)])


    if pics_dir is not None:
        fig, ax = plt.subplots(4,1, sharex = True)
        ax[0].plot(mean_t0, label = 'T before L1')
        ax[0].plot(outlier_ind, mean_t0[outlier_ind], 'ok')
        ax[0].plot(mean_t1, label = 'T after L1')
        ax[0].legend()

        ax[1].plot(L1_corr, label = 'L1 corr')
        ax[1].legend()


        ax[2].plot (corr_tend, label = 'corr_tend (t0)')
        ax[2].plot (outlier_ind, corr_tend[outlier_ind], 'ok')
        ax[2].plot(t0_diff_mean, label = 'diff2prev (t0)')
        ax[2].plot(t1_diff_mean, label = 'diff2prev (t1)')
        ax[2].legend()

        ax[3].plot (corr_tend, label = 'corr_tend (t0)')
        ax[3].plot (outlier_ind, corr_tend[outlier_ind], 'ok')
        ax[3].plot (corr_tend_new, label = 'corr_tend (t1)')
        ax[3].legend()


        plt.savefig(pics_dir + fig_name + ', ' + opts_str + '.png')
    
    return img_array_new, diff_matrix_new

def run_L2_corr (img_array, diff_matrix, use_detrend, n_steps, wnd_size, corr_rate = 1, pics_dir = None, fig_name = 'L2_corr', opts_str = ''):
    
    img_N = img_array.shape[2]

    diff_matrix_new = diff_matrix.copy()
    img_array_new = img_array.copy()

    fig = plt.figure()
    
    for step in range (0, n_steps):

        corr_tend_c = np.array([diff_matrix_new[i+1, i-1] if (i > 0 and i < img_N - 1) else 0 for i in range (0, img_N)])
        corr_tend_l = np.array([diff_matrix_new[i+1, i]   if  i < img_N - 1 else 0 for i in range (0, img_N)])
        corr_tend_r = np.array([diff_matrix_new[i, i-1]   if  i > 0 else 0 for i in range (0, img_N)])

        corr_tend = (corr_tend_c / 2 + corr_tend_l + corr_tend_r) / 3
        corr_tend [np.isnan(corr_tend)] = 0

        corr_tend_sm = pd.Series(corr_tend).rolling(wnd_size, center = True, min_periods = 1).mean()

        L2_corr = -np.cumsum(corr_tend_sm*corr_rate)
        if use_detrend:
            L2_corr = scipy.signal.detrend (L2_corr)

        img_array_new = apply_corr2array (img_array_new, L2_corr)
        diff_matrix_new = apply_corr2diff_matrix (diff_matrix_new, L2_corr)

        
        if pics_dir is not None:

            mean_t1 = np.mean(np.mean(img_array, axis=0), axis=0)
            mean_t2 = np.mean(np.mean(img_array_new, axis=0), axis=0)
            
            t1_diff_mean = np.diff (mean_t1)
            t2_diff_mean = np.diff (mean_t2)

            if step == 0:
                mean_t2_step0 = mean_t2
                L2_corr_step0 = L2_corr

            plt.clf()
            ax = fig.subplots(3,1, sharex = True)

            ax[0].plot(mean_t1, '-k', label = 'T before L2')
            ax[0].plot(mean_t2_step0, label = 'T after L2 (step = 0)')
            ax[0].plot(mean_t2,       label = 'T after L2 (step = %d)'%step)
            ax[0].grid()
            ax[0].legend()

            ax[1].plot(corr_tend,    label = 'corr_tend')
            ax[1].plot(corr_tend_sm, label = 'corr_tend_sm')
            ax[1].plot(t1_diff_mean, label = 'diff2prev (t1)', linewidth = 0.5)
            ax[1].plot(t2_diff_mean, label = 'diff2prev (t2)', linewidth = 0.5)
            ax[1].grid()
            ax[1].legend()

            ax[2].plot(L2_corr_step0, label = 'L2 corr (step = 0)')
            ax[2].plot(L2_corr,       label = 'L2 corr (step = %d)'%step)
            ax[2].grid()
            ax[2].legend()

            ax[2].set_xlim([0, img_N])
            
            plt.savefig(pics_dir + '%s, step = %d, %s'%(fig_name, step, opts_str) + '.png')
    return img_array_new, diff_matrix_new

def run_L3_corr (img_array, img_df, diff_matrix, diff_weights, n_steps, wnd_size, pics_dir = None, fig_name = 'L3_corr', opts_str = ''):
    img_N = img_array.shape[2]

    crd = np.arange(0, img_N)

    mean_t2 = np.mean(np.mean(img_array, axis=0), axis=0)


    diff_matrix_new = diff_matrix.copy()
    img_array_new = img_array.copy()

    fig = plt.figure()
    
    for step in range (0, n_steps):

        L3_corr = np.zeros_like(img_df['gps_lon'])

        for i in tqdm (range (0, img_N)):

            diff_line = diff_matrix_new[i, :]
            weight_line = diff_weights[i, :]

            ind2sel = np.where(~np.isnan(diff_line))[0]

            gps_az_diff = img_df['gps_azimuth'] - img_df['gps_azimuth'][i]
            gps_az_diff = np.mod (np.abs(gps_az_diff), 360)

            # display(img_df['gps_azimuth'][idx2test])
            # display(img_df['gps_azimuth'][idx2test-1])
            # display(gps_az_diff[idx2test-1])
            # display(np.abs(np.mod (gps_az_diff[idx2test-1], 360)))

            ind2sel_az =  np.where(~np.isnan(diff_line) & (gps_az_diff > 45))[0]

            #L3_corr[i] = -np.mean(diff_line[ind2sel_az])/2
            L3_corr[i] = -0.5 * np.sum(diff_line[ind2sel_az] * weight_line[ind2sel_az]) / np.sum(weight_line[ind2sel_az])

            if np.isnan(L3_corr[i]):
                idx2test = i

                display (mean_t2[idx2test])
                display (mean_t2[ind2sel_az])
                display (corr_line[ind2sel_az])

                ind2draw = slice (np.min(ind2sel), np.max(ind2sel))
                # plt.figure()
                # plt.scatter(img_df['gps_lon'], img_df['gps_lat'], 25, mean_t2, edgecolor = 'white')
                # plt.scatter(img_df['gps_lon'][ind2draw], img_df['gps_lat'][ind2draw], 25, mean_t2[ind2draw], edgecolor = 'gray')
                # plt.scatter(img_df['gps_lon'][ind2sel], img_df['gps_lat'][ind2sel], 25, mean_t2[ind2sel], edgecolor = 'black')
                # plt.scatter(img_df['gps_lon'][ind2sel_az], img_df['gps_lat'][ind2sel_az], 25, mean_t2[ind2sel_az], 's', edgecolor = 'black')

                # plt.scatter(img_df['gps_lon'][idx2test], img_df['gps_lat'][idx2test], 25, mean_t2[idx2test], edgecolor = 'red')


                # plt.figure()
                # plt.pcolormesh(crd[ind2draw], crd[ind2draw], corr_matrix[ind2draw, ind2draw], cmap='seismic')


                # plt.plot(crd[idx2test], crd[idx2test], 'sr', markerfacecolor="None")
                # for i in ind2sel_az:
                #     plt.plot(crd[idx2test], crd[i], 'sk', markerfacecolor="None", markersize = 2)
                #     plt.plot(crd[i], crd[idx2test], 'sk', markerfacecolor="None", markersize = 2)

                # plt.figure()
                # plt.plot(crd[ind2sel], test_line[ind2sel], 'o')
                # break

        #L3_corr_sm = np.convolve(L3_corr, np.ones(wnd_size)/wnd_size, mode='same')
        L3_corr_sm = pd.Series(L3_corr).rolling(wnd_size, center = True, min_periods = 1).mean()   

        img_array_new = apply_corr2array(img_array_new, L3_corr_sm)
        diff_matrix_new =  apply_corr2diff_matrix(diff_matrix_new, L3_corr_sm)

        

        if pics_dir is not None:

            mean_t3 = np.mean(np.mean(img_array_new, axis=0), axis=0)

            if step == 0:
                mean_t3_step0 = mean_t3
                L3_corr_step0 = L3_corr

            
            #fig, ax = plt.subplots(2,1, sharex = True)
            plt.clf()
            ax = fig.subplots(2,1, sharex = True)
            ax[0].plot(mean_t2, '-k', label = 'T before L3')
            ax[0].plot(mean_t3_step0, label = 'T after L3, step = 0')
            ax[0].plot(mean_t3, label = 'T ater L3, step = %d'%step)
            ax[0].legend()

            ax[1].plot(L3_corr_step0, label = 'L3 corr, step = 0')
            ax[1].plot(L3_corr, label = 'L3 corr, step = %d'%step)
            ax[1].plot(L3_corr_sm, label = 'L3 corr sm, step = %d'%step)
            ax[1].legend()

            plt.savefig(pics_dir + '%s, step = %d, %s'%(fig_name, step, opts_str) + '.png')
            
    return img_array_new, diff_matrix_new    


def run_L4_corr (img_array, img_df, diff_matrix, use_detrend, lowess_width = 0.5, pics_dir = None, fig_name = 'L4_corr', opts_str = '', wnd_size = None):

    img_df = img_df.copy()

    img_df['mean_t'] = np.mean(np.mean(img_array, axis=0), axis=0)
    
    swath_pos = img_df['swath_pos']
    idx_neg = np.where(swath_pos < 0)[0]
    idx_pos = np.where(swath_pos > 0)[0]

    ids_neg = img_df['swath_id'][idx_neg].unique()
    ids_pos = img_df['swath_id'][idx_pos].unique()


    y_sm = np.zeros_like(img_df['mean_t'])
    y_sm[idx_neg] = lowess.lowess(swath_pos [idx_neg], img_df['mean_t'][idx_neg], bandwidth=0.5)
    y_sm[idx_pos] = lowess.lowess(swath_pos [idx_pos], img_df['mean_t'][idx_pos], bandwidth=0.5)

    corr = - (y_sm - np.mean(y_sm))

    if wnd_size is not None:
        #corr_sm = np.convolve(corr, np.ones(wnd_size)/wnd_size, mode='same')
        corr_sm = pd.Series(corr).rolling(wnd_size, center = True, min_periods = 1).mean()
    else:
        corr_sm = corr 

    img_array_new   = apply_corr2array       (img_array, corr_sm)
    diff_matrix_new = apply_corr2diff_matrix (diff_matrix, corr_sm)

    if use_detrend:
        corr_dt      = detrend_corr (img_array_new)
        img_array_new   = apply_corr2array (img_array_new, corr_dt)
        diff_matrix_new = apply_corr2diff_matrix (diff_matrix_new, corr_dt)

    if pics_dir is not None:
        
        img_df['mean_t_new'] = np.mean(np.mean(img_array_new, axis=0), axis=0)

        fig, ax = plt.subplots(3,1) #, sharex = True)
        #ax[0].plot(swath_pos, img_df['mean_t'], 'ok')

        unique_ids = np.unique(img_df['swath_id'])
        for id in unique_ids:
            ax[0].plot (swath_pos[img_df['swath_id'] == id], img_df['mean_t'][img_df['swath_id'] == id])
        #ax[0].plot(swath_pos[idx_neg], img_df['mean_t'][idx_neg], 'ob')
        #ax[0].plot(swath_pos[idx_pos], img_df['mean_t'][idx_pos], 'or')

        ax[0].plot(swath_pos[idx_neg], y_sm[idx_neg], 'ob')
        ax[0].plot(swath_pos[idx_pos], y_sm[idx_pos], 'or')

        ax[0].set_xlabel ('Position in swath (number of image)')
        ax[0].set_xlabel ('Temperature, °C')

        ax[1].plot(corr, label = 'L4 corr')
        ax[1].plot(corr_sm, label = 'L4 corr smoothed')
        ax[1].legend()

        ax[2].plot(img_df['mean_t'], '-k', label = 'T before L4')
        ax[2].plot(y_sm, label = 'T appriximated')
        ax[2].plot(img_df['mean_t_new'], label = 'T after L4')

        ax[2].legend()

        plt.savefig(pics_dir + fig_name + ', ' + opts_str + '.png')

    return img_array_new, diff_matrix_new


def run_L5_corr (img_array, img_df, diff_matrix, diff_weights = None,
                 n_iter = 20, corr_rate = 0.5, corr_smooth_wnd = None, pics_dir = None, fig_name = 'L5_corr', opts_str = ''):
    if diff_weights is not None:
        diff_weights_na = diff_weights.copy()
        diff_weights_na[diff_weights == 0] = np.nan

    img_array_new = img_array.copy()
    diff_new = diff_matrix.copy()

    mean_abs_diff = []

    for i in tqdm(range (0, n_iter)):
        if diff_weights is not None:
            corr = corr_rate * np.nanmean(diff_new * diff_weights_na, axis=0) / np.nanmean(diff_weights_na, axis=0)
        else:
            corr = corr_rate * np.nanmean(diff_new, axis=0) 

        if corr_smooth_wnd is not None:
            corr_sm = pd.Series(corr).rolling(corr_smooth_wnd, center = True, min_periods = 1).mean()
        else:
            corr_sm = corr 

        if i == 0:
            first_corr = corr_sm

        img_array_new = apply_corr2array (img_array_new, corr_sm)
        diff_new = apply_corr2diff_matrix (diff_new, corr_sm)

        mean_abs_diff.append(np.nanmean(np.abs(diff_new)))

    if pics_dir is not None:

        img_df = img_df.copy()
        img_df['mean_t'] = np.mean(np.mean(img_array, axis=0), axis=0)
        img_df['mean_t_new'] = np.mean(np.mean(img_array_new, axis=0), axis=0)

        fig, ax = plt.subplots(3,1) 
        
        ax[0].plot (mean_abs_diff)
        ax[0].set_xlabel ('iteration')
        ax[0].set_ylabel ('mean_abs_diff')

        ax[1].plot (first_corr, label = 'corr, step = 0')
        ax[1].plot (corr_sm, label = f'corr, step={i}')
        ax[1].legend()
        
        ax[2].plot(img_df['mean_t'], '-k', label = 'T before L5')
        ax[2].plot(img_df['mean_t_new'], label = 'T after L5')
        ax[2].legend()

        plt.savefig(pics_dir + fig_name + ', ' + opts_str + '.png')

    return img_array_new, diff_new



