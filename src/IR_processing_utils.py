import os, sys, io, math
import glob
import subprocess
import pickle

from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.signal

from matplotlib.path import Path
from tqdm import tqdm
import tifffile as tiff

from PIL import Image, ExifTags, TiffTags
import exifread

from shapely import Polygon
from geopy import Point
from geopy.distance import distance, geodesic 


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

def compare_polygons (img_array, img_df, i1, i2):
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

        return delta_t, weight
    else:
        return np.nan, 0


def calc_diff_matrix (img_array, img_df):
    corr_matrix = np.zeros((img_df.shape[0], img_df.shape[0]))*np.nan
    corr_weights = np.zeros((img_df.shape[0], img_df.shape[0]))

    for i1, pol1 in enumerate(tqdm (img_df['Polygon'])):
        for i2, pol2 in enumerate (img_df['Polygon']):
            if np.isnan (corr_matrix[i2,i1]):
                corr_matrix[i1,i2], corr_weights[i1,i2] = compare_polygons (img_array, img_df, i1, i2)
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




def write_IR_image (img_data, dest_path, exif_src_path = None, exiftool_path = 'exiftool.exe'):
        
    tiff.imwrite(dest_path, np.flipud(img_data), photometric='minisblack')

    if exif_src_path is not None:

        cmd = '%s -tagsfromfile "%s" "%s"'%(exiftool_path, exif_src_path, dest_path)
        subprocess.run(cmd, check=True, shell=True)
        os.remove(dest_path + '_original')

    return True


def write_IR_image_dict (d:dict):
    return write_IR_image (d['img_data'], d['dest_path'], d['exif_src_path'], d['exiftool_path'])

        # tiff.imwrite(out_path, np.flipud(cur_data), photometric='minisblack')
        # cmd = 'exiftool.exe -tagsfromfile "%s" "%s"'%(org_path, out_path)
        # subprocess.run(cmd, check=True, shell=True)
        # os.remove(out_path + '_original')

def write_IR_images (img_array:np.ndarray, img_df:pd.DataFrame, out_dir:str, raw_dir:str, n_jobs = 1, exiftool_path = 'exiftool.exe'):
    
    d = [{'img_data': img_array[:,:,i], 
          'dest_path':     out_dir + '\\' + img_df['file'][idx], 
          'exif_src_path': raw_dir + '\\' + img_df['file'][idx], 
          'exiftool_path': exiftool_path} for i, idx in enumerate(img_df.index)]

    with Pool(n_jobs) as p:
        res = list(p.imap(write_IR_image_dict, tqdm(d, total=len(d))))


    




def read_IR_image (file):
    
    cur_array = np.flipud (np.array(tiff.imread(file)))

    exif_tags = get_exif_tags (file) #.replace('.tiff', '.jpg'))
    
    exif_df = pd.DataFrame()
    exif_df['file'] = [os.path.basename(str(file))]
    
    exif_df['gps_lat'] = decimal_coords(exif_tags['GPS GPSLatitude'], exif_tags['GPS GPSLatitudeRef'])
    exif_df['gps_lon'] = decimal_coords(exif_tags['GPS GPSLongitude'], exif_tags['GPS GPSLongitudeRef'])
    
    for key in exif_tags.keys():
        try:
            exif_df[key] = float (exif_tags[key].values[0])
        except:
            exif_df[key] = str (exif_tags[key])

    return {'img_data': cur_array, 'img_info': exif_df}


def read_IR_images (data_dir, reload = False, n_jobs = 1, N_files = None):
    pkl_path = data_dir + '/img_data.pkl'

    if not os.path.isfile (pkl_path):
        reload = True

    if reload:
        img_df = pd.DataFrame()
        img_array = {}

        print ('hello files')

        files = glob.glob (data_dir + '/*.tiff')
        
        if N_files is not None:
            files = files[0:N_files]  

        if n_jobs == 1:
            res = list (map(read_IR_image, files))
        else:
            with Pool(n_jobs) as p:
                res = list(p.map(read_IR_image, files))

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

    return img_array, img_df
    

def init_polygons (img_df, sensor_size, flight_height):
    img_polygons = []

    for i in img_df.index:
        focal_lengh = img_df['EXIF FocalLength'][i]

        img_h= flight_height * sensor_size [0] / (focal_lengh) 
        img_w= flight_height * sensor_size [1] / (focal_lengh)
        
        lat_c = img_df['gps_lat'][i]
        lon_c = img_df['gps_lon'][i]
        
        dist = np.sqrt((img_h/2)**2 + (img_w/2)**2)

        alpha = 90 - np.rad2deg (np.arctan (img_h/img_w)) #90 

        p1 = geodesic (meters=dist).destination((lat_c, lon_c), alpha)
        p2 = geodesic (meters=dist).destination((lat_c, lon_c), 180-alpha) 
        p3 = geodesic (meters=dist).destination((lat_c, lon_c), -(180-alpha))
        p4 = geodesic (meters=dist).destination((lat_c, lon_c), -alpha)

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

def apply_corr2array (img_array, corr):
    img_array_corr = img_array.copy()
    for i, c in enumerate (corr):
        img_array_corr[:,:,i] += c

    mean_diff = img_array_corr.mean() - img_array.mean()

    img_array_corr -= mean_diff

    return img_array_corr

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
    