# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:48:13 2021

@author: Jin-Zhu Yu
"""

import numpy as np
import pandas as pd
import os
os.chdir('/mnt/c/code/Scale_Mobility/src/')
# from my_utils import *
# import pickle
# import glob

# import models    # in case where infostop cannot be installed

import infostop
# from datetime import datetime
from time import time
from joblib import Parallel, delayed
# from geopy.geocoders import Nominatim    #https://geopy.readthedocs.io/en/stable/#nominatim


# # class MobilityData:
#     '''  
#     Our objective:
#         combine data in a day
#         merge consercutive identify stop points
#         Extract home locations
#         Relabel the locations
#         Store the data for say 10000 individual in one file (preferably a pickle file for faster readin)
#     Data:
#         Data are stored in the path: All data/ folder for each day/thousands of csv files
#         each csv file contains the mobility trajectories of thousands of invididuals: id_str, loc id, time and coords of locations
#         'id_str', 'label', 'start', 'end', 'lat_start', 'lon_start', 'lat_end', 'lon_end'
#         Note that the location id for different dates are independent. That's why we also need to relabel the locations.      
#     '''
# def __init__(dir_raw_data, dir_processed_data,
#          night_start=18, night_end=8,
#          min_dist_betwn_point=30,
#          is_save=False):
# dir_raw_data = dir_raw_data
# dir_processed_data = dir_processed_data
# # r2 = r2    # min distances between two different stop points
# night_start = night_start
# night_end = night_end
# min_dist_betwn_point = min_dist_betwn_point
# is_save = is_save

night_start, night_end = 18, 8

def merge_consec_stopoint(df_temp):
    '''
    Merge consecutive stop points that are actually the same point.
        For example, consecutive entries 13 to 16 all have location label 2, thse entries should be merged into one entry.
        The start time (use min) and end time (use max) as well as start coords (use mean) and end coords (use mean) are updated as well

    Params
    ----------
    df_temp: a df that includes the trajectories of one csv file (about 2000 individuals) (or use all?) in a single day
    '''
    # assign the same group id if the values of id str and label are the same
    df_is_value_same = (df_temp[['id_str','label']] != df_temp.shift().fillna(method='bfill')[['id_str', 'label']])
    # increase group id number only if both values are true
    temp_group =(df_is_value_same.sum(axis=1)>0).cumsum().rename('temp_group')
    # group by id stri, label and id of temp group
    df_temp_merged = df_temp.groupby(['id_str','label',temp_group]).agg({'start': ['min'],'end': ['max'],'latitude': ['mean'],'longitude': ['mean']})
    # rename and select 
    df_temp_merged.columns = ['start', 'end', 'latitude','longitude']
    df_temp_merged = df_temp_merged.reset_index()      
    return df_temp_merged      


def remove_single_record_user(df, n_min_record=2):
    '''
    Select users that have at least two different stop points per day.
    This can be done immediately after merging the consecutive identical stop points
    '''        
    df = df[df.groupby('id_str')['id_str'].transform('size') >= n_min_record]
    return df


# relabel stop points of a single user
def relabel_by_group(group):
    ''' relabel the stop points of each individual's df
        find home location of each individual
    '''
    coord_arr = np.array(group[['latitude','longitude']].values)
    print(coord_arr)

    # # get labels of locations of single individual
    # model_infostop = infostop.SpatialInfomap(r2=min_dist_betwn_point,
    #                                          label_singleton=True,
    #                                          min_spacial_resolution=0.0001,
    #                                          distance_metric='haversine',            
    #                                          verbose=False) ###only true for testing
    # label_list = model_infostop.fit_predict(coord_arr)
    # TODO: relabel all users instead of by group 
        #If the input type is a list of arrays, each array is assumed
        #to be the trace of a single user, in which case the obtained stop locations are shared by all users in
        #the population.
    # coord_nest_list = df.groupby('id_str')[['latitude','longitude']].apply(pd.Series.tolist).tolist()[:2]
    # coord_arr = [np.asarray(item) for item in coord_nest_list][:2]
    # coord_arr

    min_dist_betwn_point = 30
    model_infostop = infostop.SpatialInfomap(r2=min_dist_betwn_point,
                                            label_singleton=True,
                                            min_spacial_resolution=0.0001,
                                            distance_metric='haversine',            
                                            verbose=False) ###only true for testing
    label_list = model_infostop.fit_predict(coord_arr)        

    return label_list         


# TODO: label whether or not a location is home

def relabel_all_group(df, min_dist_betwn_point=30):
    # change to float
    df[['start','end', 'latitude', 'longitude']] = df[['start', 'end', 'latitude', 'longitude']].astype(float)         
    # relabel stop points
    # df_group = df.groupby('id_str')
    label_list = df.groupby('id_str').apply(relabel_by_group)
    df['label'] = label_list
    return df


def find_home_loc(df_temp):    
    df_temp_night = df_temp[(pd.to_datetime(df_temp['start'], unit='s').dt.hour>=night_start)&\
                            (pd.to_datetime(df_temp['end'], unit='s').dt.hour<=night_end)]
    if len(df_temp_night)==0:
        lat, lon = None, None
        # lat, lon, county, state = None, None, None, None
    else:
        (lat, lon) = df_temp_night.groupby(['latitude', 'longitude']).size().idxmax()
        # geolocator = Nominatim(user_agent="geoapiExercises")
        # location = geolocator.reverse(str(lat) + "," + str(lon))
        # address = location.raw['address']
        # county= address.get('county', '')           
        # state = address.get('state', '')
      # city = address.get('city', '')
      # zipcode = address.get('postcode') 
    return lat, lon  #, county, state


def find_home_by_group(group):
    # home_lat,home_lon,county,state = find_home_loc(group)
    home_lat,home_lon = find_home_loc(group)
    user_id = group['id_str'].iloc[0]
    return [user_id, home_lat, home_lon]  #, county, state]


def find_home_all_group(df):
    # get each user's home location
    user_home_list = df.groupby('id_str').apply(find_home_by_group)
    df_user_home = pd.DataFrame(user_home_list, columns=['id_str','home_lat','home_lon'])  #,'county','stte'])    
    return df_user_home


def get_movement_feature(df_temp):
    '''Get travel time, stay time, travel distance, travel bearing
    '''
    n_row = len(df_temp)
    df_temp['stay_time'] = df_temp['end'] - df_temp['start']    
    # travel time
    df_temp_shift = df_temp.shift().fillna(method='bfill')
    df_temp['travel_time'] = df_temp['start'] - df_temp_shift['end']
    df_temp.loc[n_row, 'travel_time'] = np.nan   # no travel time for the last row
    # # dist and angle
    # df_temp['travel_dist'], df_temp['travel_angle'] = haversine(df_temp['latitude'], df_temp['longitude'],
    #                                                             df_temp_shift['latitude'], df_temp_shift['longitude'])
    # df_temp.loc[n_row, 'travel_dist'] = np.nan   # no travel dist for the last row
    # df_temp.loc[n_row, 'travel_angle'] = np.nan   # no travel angle for the last row     
    return df_temp

# def process_all_date():
def loop_over_date(this_date):
    t0 = time()
    file_list= list(os.listdir(dir_raw_data+this_date))
    # load data file one at a time and merge consecutive identical stop points
    # TODO: another option is loading all csv files into one giant csv and 
    def loop_over_file(fname):
        df_temp = pd.read_csv(dir_raw_data+this_date+'/'+fname)
        # drop the useless column
        if 'Unnamed: 0' in df_temp.columns:
            df_temp = df_temp.drop(['Unnamed: 0'], axis=1)
        # get the center of stop points
        df_temp['latitude'] = (df_temp['lat_start']+df_temp['lat_end'])/2
        df_temp['longitude'] = (df_temp['lon_start']+df_temp['lon_end'])/2    
        df_temp_merged = merge_consec_stopoint(df_temp)
        return df_temp_merged
    df_list = Parallel(n_jobs=4,verbose=0)(delayed(loop_over_file)(fname) for fname in file_list)
    df = pd.concat(df_list, ignore_index=True)
    df = df[['id_str', 'label', 'start', 'end','latitude', 'longitude']]
    t1 = time()
    print('===== Time for merging:{}'.format(t1-t0))

    # remove users with just one record
    t0 = time()
    df = remove_single_record_user(df)
    t1 = time()
    print('===== Time for removing users with a single record:{}'.format(t1-t0))            

    # find user home location
    # TODO: label home location in the mobility data
    t0 = time()
    df_user_home = find_home_all_group(df)
    t1 = time()
    print('===== Time for finding home location:{}'.format(t1-t0))     

    # # relabel
    # t0 = time()
    # df = relabel_all_group(df)
    # t1 = time()
    # print('===== Time for relabeling:{}'.format(t1-t0))

    # TODO: labels are all NaNs;
    # TODO: groupby is inefficient. Use transform that takes a series as input
    # https://towardsdatascience.com/how-to-make-your-pandas-loop-71-803-times-faster-805030df4f06 
    # TODO: check if the code can be converted to cython code                                
        
    # get other movement feature: travel time and dist, stay time, travel angle.
    t0 = time()
    df = get_movement_feature(df)
    t1 = time()
    print('===== Time for extracting other movement features:{}'.format(t1-t0))             

    # save the processed data
    t0 = time()
    if is_save:
        # saving and loading pickle files are significantly faster than csv files
        df.to_pickle(dir_processed_data +'{}.pkl'.format(this_date))
        df_user_home.to_pickle(dir_processed_data+'user_home_location.pkl')
        del df, df_user_home
    t1 = time()
    print('===== Time for saving:{}'.format(t1-t0))

is_save = False
dir_raw_data = '../data/mobility_data_6_month/'
dir_processed_data = '../data_processed/stop_points/'
folder_list = list(os.listdir(dir_raw_data))[:1]    #[-3:]
# this_date = folder_list[1]
# print(folder_list)
[loop_over_date(folder) for folder in folder_list]   # returns list of 'None'. A bit faster than 'for loop' and 'any'


# #####
# def main():
#     dir_raw_data = '../data/mobility_data_6_month/'
#     dir_processed_data = '../data_processed/stop_points/'
      # is_save = False
    
#     mobility_data = MobilityData(dir_raw_data, dir_processed_data, is_save=is_save)
#     t0 = time()
#     mobility_data.process_all_date()
#     t1 = time()
#     t_total = t1 - t0
#     print('\n===== Completed all. Time elapsed: {}'.format(t_total))
    

######################
# if __name__ == 'main':
# main()
    

