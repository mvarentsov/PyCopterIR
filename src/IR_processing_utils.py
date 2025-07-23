import os, sys, io, math, time 
import glob
import subprocess
import pickle

from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.signal

from matplotlib import pyplot as plt
from matplotlib.path import Path
from tqdm import tqdm
import tifffile as tiff

from PIL import Image, ExifTags, TiffTags
import exifread

from shapely import Polygon
from geopy import Point
from geopy.distance import distance, geodesic 

import rasterio
from pyproj import Transformer

import cv2


def get_subarray_coords_rio (arr_shape, src_crs, trs, dest_crs = 4326):
    height = arr_shape[0]
    width = arr_shape[1]

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(trs, rows, cols)

    xs = np.array(xs)
    ys = np.array(ys)

    trs_wgs = Transformer.from_crs (src_crs, dest_crs)
    lons, lats = trs_wgs.transform(xs,ys)
    return lons, lats

def decimal_coords(coords, ref):
    
    try:
        coords = coords.values
    except:
        pass
    
    decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float (coords[2]) / 3600
    if ref == "S" or ref =='W' :
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def calc_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def get_exif_tags (path):
    img = Image.open (path)
        
    # Open image file for reading (must be in binary mode)
    with open(path, 'rb') as f:
        exif_tags = exifread.process_file(f, details=False)
    #display (tags)


    # if path.endswith('.tiff'):
    #     # for key in img.tag.keys():
    #         # display(key)
    #         # display(img.tag[key])
    #         # display(TiffTags.TAGS.get(key))
    #     exif_tags = {TiffTags.TAGS.get(key): img.tag[key] for key in img.tag.keys()} 
    #     #exif_tags2 = {TiffTags.TAGS.get(key): img.tag[key] for key in img.tag_v2.keys()} 
    #     display(exif_tags)
    #     display(exif_tags)
    #     #display(exif_tags2)
    # elif path.endswith('.jpg'):
    #     exif = img._getexif()
    #     exif_tags = {}
    #     for tag, value in exif.items():
    #         exif_tags[ExifTags.TAGS.get(tag, tag)] = value

        
    # for key in exif_tags['GPSInfo'].keys():
    #      decode = ExifTags.GPSTAGS.get(key,key)
    #      exif_tags[decode] = exif_tags['GPSInfo'][key]
    return exif_tags


def create_mesh (min_lon, max_lon, min_lat, max_lat, shape):
    x_grid = np.arange(min_lon, max_lon, (max_lon - min_lon)/shape[1])
    y_grid = np.arange(min_lat, max_lat, (max_lat - min_lat)/shape[0])
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    xy_mesh = np.dstack((x_mesh, y_mesh))
    xy_mesh_flat = xy_mesh.reshape((-1, 2))
    return x_mesh, y_mesh, xy_mesh_flat

def img_diff_poly (img_array, img_df, i1, i2, draw_figure = False):
    pol1 = img_df['Polygon'][i1]
    pol2 = img_df['Polygon'][i2]

    pol_intersect = pol1.intersection(pol2)

    if i1 != i2 and pol_intersect.area > pol1.area * 0.25:

        pol1_x, pol1_y, pol1_xy_flat = create_mesh (img_df['min_lon'][i1], img_df['max_lon'][i1], img_df['min_lat'][i1], img_df['max_lat'][i1], img_array.shape)
        pol2_x, pol2_y, pol2_xy_flat = create_mesh (img_df['min_lon'][i2], img_df['max_lon'][i2], img_df['min_lat'][i2], img_df['max_lat'][i2], img_array.shape)

        mpath = Path(np.transpose(pol_intersect.exterior.xy)) # the vertices of the polygon
        pol1_mask = mpath.contains_points(pol1_xy_flat).reshape(pol1_x.shape) 
        pol2_mask = mpath.contains_points(pol2_xy_flat).reshape(pol2_x.shape) 

        pol1_t = img_array[:,:,i1]
        pol2_t = img_array[:,:,i2]

        all_t = np.concatenate ((np.atleast_3d(pol1_t), np.atleast_3d(pol2_t)), axis=2).flatten()

        pol1_t_mean = pol1_t[pol1_mask].mean()
        pol2_t_mean = pol2_t[pol2_mask].mean()

        delta_t  = pol1_t_mean - pol2_t_mean
        weight = 0.5 * (np.sum(pol1_mask.flatten()) + np.sum(pol2_mask.flatten())) / (img_array.shape[0]*img_array.shape[1])

        
        if draw_figure:
            min_x = np.min([pol1_x.min(), pol2_x.min()])
            max_x = np.max([pol1_x.max(), pol2_x.max()])
            min_y = np.min([pol1_y.min(), pol2_y.min()])
            max_y = np.max([pol1_y.max(), pol2_y.max()])
            min_t = np.min([pol1_t.min(), pol2_t.min()])
            max_t = np.max([pol1_t.max(), pol2_t.max()])


            plt.figure()
            plt.pcolormesh (pol1_x, pol1_y,pol1_t, vmin = min_t, vmax = max_t) 
            plt.plot(*pol1.exterior.xy, '-k')
            plt.plot(*pol2.exterior.xy, '-k')
            plt.xlim([min_x, max_x])
            plt.ylim([min_y, max_y])
            plt.colorbar()
            plt.savefig('1.png')
            
            

            plt.figure()
            plt.pcolormesh (pol2_x, pol2_y,pol2_t, vmin = min_t, vmax = max_t) 
            plt.plot(*pol1.exterior.xy, '-k')
            plt.plot(*pol2.exterior.xy, '-k')
            plt.xlim([min_x, max_x])
            plt.ylim([min_y, max_y])
            plt.colorbar()
            
            plt.savefig('2.png')
            

        return delta_t, weight
    else:
        return np.nan, 0

def img_diff_SIFT (img_array, img_df, i1, i2, draw_figure = False):
    
    pol1 = img_df['Polygon'][i1]
    pol2 = img_df['Polygon'][i2]

    pol_intersect = pol1.intersection(pol2)

    if i1 != i2 and pol_intersect.area > pol1.area * 0.25:

        # if img_df['file'][i1] == 'DJI_20240812105408_0255_T.tiff':
        #     print ('hi file')
        # else:
        #     return  np.nan, 0

        img1 = img_array[:,:,i1]
        img2 = img_array[:,:,i2]
        
        img1_cv = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img2_cv = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img1_enhanced = clahe.apply(img1_cv)
        img2_enhanced = clahe.apply(img2_cv)

        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1_enhanced, None)
        kp2, des2 = sift.detectAndCompute(img2_enhanced, None)

        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2) # k=2 for ratio test

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance: # Lowe's ratio test
                good_matches.append(m)

        if len (good_matches) < 10:
            return np.nan, 0

        img1_pts = []
        img2_pts = []

        for match in good_matches:
            # Get the keypoints for the match
            img1_idx = match.queryIdx  # Index in first image
            img2_idx = match.trainIdx  # Index in second image
            
            # Get coordinates from keypoints
            img1_pt = kp1[img1_idx].pt  # (x, y) in first image
            img2_pt = kp2[img2_idx].pt  # (x, y) in second image
            
            img1_pts.append(img1_pt)
            img2_pts.append(img2_pt)

        # Convert to numpy arrays for easier manipulation
        img1_pts = np.array(img1_pts)
        img2_pts = np.array(img2_pts)

        img1_coords = np.round(img1_pts).astype(int)
        img2_coords = np.round(img2_pts).astype(int)

        img1_tiff_values = img1[img1_coords[:, 1], img1_coords[:, 0]]
        img2_tiff_values = img2[img2_coords[:, 1], img2_coords[:, 0]]
        
        delta_t = np.mean(img1_tiff_values - img2_tiff_values)
        weight = img1_tiff_values.shape[0]
        return delta_t, weight
    else:
        return np.nan, 0

def calc_diff_matrix (img_array, img_df, diff_func = img_diff_poly):
    corr_matrix = np.zeros((img_df.shape[0], img_df.shape[0]))*np.nan
    corr_weights = np.zeros((img_df.shape[0], img_df.shape[0]))

    for i1, pol1 in enumerate(tqdm (img_df['Polygon'])):
        for i2, pol2 in enumerate (img_df['Polygon']):
            if np.isnan (corr_matrix[i2,i1]):
                corr_matrix[i1,i2], corr_weights[i1,i2] = diff_func (img_array, img_df, i1, i2)
            else:
                corr_matrix[i1,i2] = -corr_matrix[i2,i1]
                corr_weights[i1,i2] = corr_weights[i2,i1]
    return corr_matrix, corr_weights

def apply_corr2diff_matrix (corr_matrix, corr):
    img_N = len (corr)
    corr_matrix_new = corr_matrix.copy()
    for i in range (0, img_N):
        for j in range (0, img_N):
            corr_matrix_new[i,j] = corr_matrix[i,j] + corr[i] - corr[j]
    return corr_matrix_new

def apply_corr2array (img_array, corr):
    img_array_corr = img_array.copy()
    for i, c in enumerate (corr):
        img_array_corr[:,:,i] += c

    mean_diff = img_array_corr.mean() - img_array.mean()

    img_array_corr -= mean_diff

    return img_array_corr

def write_IR_image (img_data, dest_path, exif_src_path = None, exiftool_path = 'exiftool.exe', update_files = True):
        
    if not os.path.isfile (dest_path) or update_files:
        tiff.imwrite(dest_path, np.flipud(img_data), photometric='minisblack')

        if exif_src_path is not None:

            cmd = '%s -tagsfromfile "%s" "%s"'%(exiftool_path, exif_src_path, dest_path)
            subprocess.run(cmd, check=True, shell=True)
            os.remove(dest_path + '_original')

    return True


def write_IR_image_dict (d:dict):
    return write_IR_image (d['img_data'], d['dest_path'], d['exif_src_path'], d['exiftool_path'], d['update_files'])

def write_IR_images (img_array:np.ndarray, img_df:pd.DataFrame, out_dir:str,  n_jobs = 1, exiftool_path = 'exiftool.exe', update_files = True):
    
    d = [{'img_data': img_array[:,:,i], 
          'dest_path':     out_dir + '\\' + img_df['file'][idx], 
          'exif_src_path': img_df['folder'][idx] + '\\' + img_df['file'][idx], 
          'exiftool_path': exiftool_path,
          'update_files': update_files} for i, idx in enumerate(img_df.index)]

    with Pool(n_jobs) as p:
        res = list(p.imap(write_IR_image_dict, tqdm(d, total=len(d))))


def read_IR_image (file):
    
    cur_array = np.flipud (np.array(tiff.imread(file)))

    exif_tags = get_exif_tags (file) #.replace('.tiff', '.jpg'))
    
    exif_df = pd.DataFrame()
    exif_df['file'] = [os.path.basename(str(file))]
    
    
    
    try:
        exif_df['gps_lat'] = decimal_coords(exif_tags['GPS GPSLatitude'], exif_tags['GPS GPSLatitudeRef'])
        exif_df['gps_lon'] = decimal_coords(exif_tags['GPS GPSLongitude'], exif_tags['GPS GPSLongitudeRef'])
    except Exception as err:
        print ('hi exception')
    
    for key in exif_tags.keys():
        try:
            exif_df[key] = float (exif_tags[key].values[0])
        except:
            exif_df[key] = str (exif_tags[key])

    return {'img_data': cur_array, 'img_info': exif_df}


def try_dump_pkl (obj, path, max_attempts = 10):
    dump_ok = False
    n_attempts = 0
    while not dump_ok:
        try:
            with open(path, 'wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            dump_ok = True
        except Exception as err:
            if n_attempts < max_attempts:
                time.sleep(1)
                pass  
            else:
                raise err          

        n_attempts += 1

def try_load_pkl (path, max_attempts = 10):
    dump_ok = False
    n_attempts = 0
    while not dump_ok:
        try:
            with open(path, 'rb') as handle:
                res = pickle.load(handle)
            dump_ok = True
            return res
        except Exception as err:
            if n_attempts < max_attempts:
                time.sleep(1)
                pass  
            else:
                raise err          
        n_attempts += 1


def read_IR_images (data_dir, reload = False, n_jobs = 1, N_files = None):
    pkl_path = data_dir + '/img_data.pkl'

    if not os.path.isfile (pkl_path):
        reload = True

    if reload:
        img_df = pd.DataFrame()
        img_array = {}

        print ('read_IR_images(): start reading files')

        files = sorted(glob.glob (data_dir + '/*.tiff'))

        assert len(files) > 0, 'no tiff files found in %s' % (data_dir)

        if N_files is not None:
            files = files[0:N_files]  

        if n_jobs == 1:
            res = list (tqdm (map(read_IR_image, files)))
        else:
            with Pool(n_jobs) as p:
                res = list(tqdm(p.imap(read_IR_image, files), total=len(files)))

        for i, r in enumerate(res):        
            cur_array = np.atleast_3d (r['img_data'])

            if i == 0:
                img_array = cur_array
                img_df = r['img_info']
            else:
                img_array =  np.concatenate((img_array, cur_array), 2)   
                img_df = pd.concat((img_df, r['img_info']), ignore_index=True)
        
        with open(pkl_path, 'wb') as handle:
            pickle.dump((img_array, img_df), handle, protocol=pickle.HIGHEST_PROTOCOL) 
    else:
        with open(pkl_path, 'rb') as handle:
            img_array, img_df = pickle.load(handle)

    img_df['folder'] = data_dir

    return img_array, img_df
    

def init_polygons4df (img_df, sensor_size, flight_height):
    img_polygons = []

    for i in img_df.index:
        #print(img_df['file'][i])
        focal_lengh = img_df['EXIF FocalLength'][i]

        img_h= flight_height * sensor_size [0] / (focal_lengh) 
        img_w= flight_height * sensor_size [1] / (focal_lengh)
        
        lat_c = img_df['gps_lat'][i]
        lon_c = img_df['gps_lon'][i]
        
        dist = np.sqrt((img_h/2)**2 + (img_w/2)**2)

        alpha = 90 - np.rad2deg (np.arctan (img_h/img_w)) #90 

        try:
            p1 = geodesic (meters=dist).destination((lat_c, lon_c), alpha)
            p2 = geodesic (meters=dist).destination((lat_c, lon_c), 180-alpha) 
            p3 = geodesic (meters=dist).destination((lat_c, lon_c), -(180-alpha))
            p4 = geodesic (meters=dist).destination((lat_c, lon_c), -alpha)
        except Exception as err:
            print ('hello exception')

        pol = Polygon (((p1[1], p1[0]), (p2[1], p2[0]), (p3[1], p3[0]), (p4[1], p4[0])))

        img_polygons.append(pol)


    img_df['min_lon'] = [np.min(pol.exterior.xy[0]) for pol in img_polygons]
    img_df['max_lon'] = [np.max(pol.exterior.xy[0]) for pol in img_polygons]
    img_df['min_lat'] = [np.min(pol.exterior.xy[1]) for pol in img_polygons]
    img_df['max_lat'] = [np.max(pol.exterior.xy[1]) for pol in img_polygons]
    img_df['delta_lon'] = img_df['max_lon'] - img_df['min_lon']
    img_df['delta_lat'] = img_df['max_lat'] - img_df['min_lat']

    img_df['Polygon'] = img_polygons

    return img_df

def calc_azimuth4df (img_df):
    gps_az = np.zeros_like (img_df.gps_lon)
    for i in range (0, img_df.shape[0]-1):
        try:
            gps_az[i] = calc_compass_bearing ((img_df.gps_lat[i],img_df.gps_lon[i]), (img_df.gps_lat[i+1],img_df.gps_lon[i+1]))
        except Exception as err:
            print ('hi error')
    gps_az[-1] = gps_az[-2]
    res_df = img_df.copy()
    res_df['gps_azimuth'] = gps_az
    return res_df

def identify_swaths4df (img_df, min_segment_len = 5, crit_delta_az = 170):   

    img_N = img_df.shape[0]

    def calc_delta_az (az1, az2):
        delta = np.abs(az1-az2)
        if delta > 180:
            delta = 360-delta
        return delta

    delta_az = np.abs(np.diff(img_df['gps_azimuth'], prepend=0))
    delta_az[delta_az > 180] = 360 - delta_az[delta_az > 180]

    swath_id = np.zeros_like(img_df['gps_azimuth'])
    swath_pos = np.zeros_like(img_df['gps_azimuth'])

    cur_swath_start = 0
    cur_swath_id = 0
    cur_swath_pos = 1
    sign = 1
    swath_az = np.nan # img_df['gps_azimuth'][0] #to do: calculate azimuth based on data for all swaths

    for i in range (0, img_N):
        swath_az_all = img_df['gps_azimuth'][cur_swath_start:i+1]

        n_avg = min (len(swath_az_all), int ((len(swath_az_all))/2))
        u = np.mean(np.sin(np.radians(swath_az_all[:n_avg])))
        v = np.mean(np.cos(np.radians(swath_az_all[:n_avg])))

        avg_az = np.degrees(np.arctan2(u, v))
        if avg_az < 0:
            avg_az += 360
    
        swath_id[i] = cur_swath_id
        swath_pos[i] = cur_swath_pos
        cur_swath_pos += 1 * sign
        cur_delta_az = []
        for j in range (i, min(img_N, i+min_segment_len)):
            cur_delta_az.append (calc_delta_az (avg_az, img_df['gps_azimuth'][j]))
        
        if np.abs (cur_swath_pos) > min_segment_len and cur_delta_az[0] > crit_delta_az and np.min(cur_delta_az) > crit_delta_az:
            print ('identify_swaths4df(): segment %d reached %d with mean_az %d\n'%(cur_swath_id, cur_swath_pos, int(avg_az)))
            cur_swath_start = i
            cur_swath_id += 1
            sign *= -1
            cur_swath_pos = 1 * sign
            #swath_az = img_df['gps_azimuth'][i]

    res_df = img_df.copy()
    res_df['delta_az']  = delta_az
    res_df['swath_id']  = swath_id
    res_df['swath_pos'] = swath_pos
    return res_df


def sigma_tend_corr_single (diff2prev, n_sigma=3):
    
    diff2prev_c = diff2prev.copy()

    diff_std = np.nanstd (diff2prev)
    corr = np.zeros_like (diff2prev)

    outlier_ind = []
    
    for i, diff in enumerate (diff2prev):
        if np.abs(diff) > n_sigma * diff_std:
            corr[i:] = corr[i:] - diff
            outlier_ind.append(i)
            diff2prev_c[i] = 0

    return corr, diff2prev_c, outlier_ind

def sigma_tend_corr_multi (diff2prev, n_sigma=3, max_iter = 10):

    corr = np.zeros_like (diff2prev)
    diff2prev_c = diff2prev.copy()
    outlier_ind = []

    for i in range (0, max_iter):
        cur_corr, diff2prev_c, cur_ind = sigma_tend_corr_single (diff2prev_c)
        if len (cur_ind) == 0:
            break
        corr += cur_corr
        display('n_outliers = %d, max_diff = %f'%(len(cur_ind), np.max (np.abs(diff2prev[cur_ind]))))
        outlier_ind += cur_ind

    return corr, diff2prev_c, outlier_ind

def detrend_corr (img_array):
    mean_vals  = np.mean(np.mean(img_array, axis=0), axis=0)
    mean_vals_dt = scipy.signal.detrend (mean_vals)
    corr = mean_vals_dt - mean_vals
    return corr

def preview_photos (img_data, img_df, idx2preview, diff_matrix = None, save_dir = None):

    img_shape = img_data.shape

    if save_dir is not None and not os.path.isdir (save_dir):
        os.mkdir (save_dir)

    plt.figure()

    min_val = np.percentile (img_data.flatten(), 1)
    max_val = np.percentile (img_data.flatten(), 99)

    for idx in idx2preview:
        
        plt.clf()
        for pol in img_df['Polygon']:
            plt.plot(*pol.exterior.xy, '-k', color = 'gray')

        plt_x_lim = plt.xlim()
        plt_y_lim = plt.ylim()

        pol = img_df['Polygon'][idx]

        plt.plot(*pol.exterior.xy, '-k', linewidth = 3)

        x_mesh, y_mesh, _ = create_mesh (img_df['min_lon'][idx], img_df['max_lon'][idx],
                                         img_df['min_lat'][idx], img_df['max_lat'][idx], img_shape)

        
        plt.pcolormesh (x_mesh, y_mesh, img_data[:,:,idx], vmin = min_val, vmax = max_val, zorder = 100) #, alpha=0.5)

        plt.xlim(plt_x_lim)
        plt.ylim(plt_y_lim)
        #cx.add_basemap(plt.gca(), source = cx.providers.Esri.WorldImagery, crs = 4326)            
        plt.gca().set_aspect(1.0/np.cos(np.array(plt.ylim()).mean()*np.pi/180))
        
        if save_dir is not None:
            plt.savefig(save_dir + str(idx)+'.png')


