# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:48:13 2021

@author: Jin-Zhu Yu
"""
'''  
Our objective:
    Extract the mobility data for each individual
    Relabel the locations
    Store the data for say 10000 individual in one file (preferably a pickle file for faster readin)
Data:
    Data are stored in the path: All data/ folder for each day/thousands of csv files
    each csv file contains the mobility trajectories of thousands of invididuals: id_str, loc id, time and coords of locations
    'id_str', 'label', 'start', 'end', 'lat_start', 'lon_start', 'lat_end', 'lon_end'
    Note that the location id for different dates are independent. That's why we also need to relabel the locations.      
'''

import numpy as np
import pandas as pd
import os
os.chdir("c:/code/Scale_Mobility/src")
from my_utils import *
import pickle
import glob
# from input_library import *

# import infostop
# from functools import reduce
from datetime import datetime
from time import time
from joblib import Parallel, delayed
from geopy.geocoders import Nominatim    #https://geopy.readthedocs.io/en/stable/#nominatim
# Section 1. Relabelling
# 
# 1, relbel_all
# 2, filter out
# 3, home locations
# 4, re_scale

#####input 

class MobilityData:
        
    def __init__(self, dir_raw_data, dir_processed_data, is_save=False):
        self.dir_raw_data = dir_raw_data
        self.dir_processed_data = dir_processed_data
        # self.r2 = r2    # min distances between two different stop points
        self.is_save = is_save
    
    def merge_consec_stopoints(self, df_temp):
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

    def get_movement_feature(self, df_temp):
        '''Get travel time, stay time, travel distance, travel bearing
        
        '''
        n_row = len(df_temp)
        df_temp['stay_time'] = df_temp['end'] - df_temp['start']    
        # travel time
        df_temp_shift = df_temp.shift().fillna(method='bfill')
        df_temp['travel_time'] = df_temp['start'] - df_temp_shift['end']
        df_temp.loc[n_row, 'travel_time'] = np.nan   # no travel time for the last row
        # dist and angle
        df_temp['travel_dist'], df_temp['travel_angle'] = haversine(df_temp['latitude'], df_temp['longitude'],
                                                                    df_temp_shift['latitude'], df_temp_shift['longitude'])
        df_temp.loc[n_row, 'travel_dist'] = np.nan   # no travel dist for the last row
        df_temp.loc[n_row, 'travel_angle'] = np.nan   # no travel angle for the last row     
        return df_temp

    def find_home_city(lat, lon):
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.reverse(str(lat) + "," + str(lon))
        address = location.raw['address']
        county= address.get('county', '')
        # city = address.get('city', '')
        state = address.get('state', '')
        # country = address.get('country', '')
        # zipcode = address.get('postcode')
        #print(county,",",city,",",state,",",country,",",zipcode)
        return county, state
    
    def find_home_loc(df_temp):
        night_start, night_end = 18, 8    # night time
        # df_temp['start_h'] = df_temp['start_h'].astype(float)
        # df_temp['end_h'] = df_temp['end_h'].astype(float)         
        # TODO: get hour value from unix time              
        df_temp_x = df_temp[(pd.to_datetime(df_temp['start'], unit='s').dt.hour>=night_start)&\
                            (pd.to_datetime(df_temp['end'], unit='s').dt.hour<=night_end)]
        if len(df_temp_x)==0:
            lat, lon, county, state = None, None, None, None
        else:
            (lat, lon) = df_temp_x.groupby(['latitude', 'longitude']).size().idxmax()
            county, state = find_home_city(lat, lon)
        return lat, lon, county, state
        
    def merge_data_single_date(self):
        def loop_over_date(this_date):
            t0 = time()
            file_list= list(os.listdir(self.dir_raw_data+this_date))
            # load data file one at a time and merge consecutive identical stop points
            # TODO: another option is loading all csv files into one giant csv and 
            def loop_over_file(fname):
                df_temp = pd.read_csv(self.dir_raw_data+this_date+'/'+fname)
                # drop the useless column
                if 'Unnamed: 0' in df_temp.columns:
                    df_temp = df_temp.drop(['Unnamed: 0'], axis=1)
                # get the center of stop points
                df_temp['latitude'] = (df_temp['lat_start']+df_temp['lat_end'])/2
                df_temp['longitude'] = (df_temp['lon_start']+df_temp['lon_end'])/2    
                df_temp_merged = self.merge_consec_stopoints(df_temp)
                return df_temp_merged
            df_list = Parallel(n_jobs=8,verbose=0)(delayed(loop_over_file)(fname) for fname in file_list)
            df = pd.concat(df_list, ignore_index=True)
            df = df[['id_str', 'label', 'start', 'end','latitude', 'longitude']]
            # get other movement feature: travel time and dist, stay time, travel angle.
            df = self.get_movement_feature(df)
            # save the processed data
            t1 = time()
            if self.is_save:
                # saving and loading pickle files are significantly faster than csv files
                df.to_pickle(self.dir_processed_data +'{}.pkl'.format(this_date))
            del df
            t2 = time()
            print('\n===== Time for merging:{}'.format(t1-t0))
            print('===== Time for saving:{}'.format(t2-t1))
        folder_list = list(os.listdir(self.dir_raw_data))    #[-3:]   
        [loop_over_date(folder) for folder in folder_list]   # returns list of 'None'. A bit faster than 'for loop' and 'any'
        print('\n===== Complete loading data and merging consecutive stop points')
   


#####
# from mpl_toolkits.basemap import Basemap as Basemap
# #####Here we use pyspark
# from pyspark.sql import SparkSession
# import pyspark
# from pyspark.sql import *
# from pyspark.sql.types import *
# from pyspark.sql.functions import col, length,monotonically_increasing_id,udf,row_number
# from pyspark.sql import types as T
# from pyspark.sql.window import Window

# from pyspark.serializers import MarshalSerializer
# # #set an application name
# # #start spark cluster; if already started then get it else create it
# sc = SparkSession.builder.appName('data_preprocess').master("local[*]").getOrCreate()
# # initialize SQLContext from spark cluster
# sqlContext = SQLContext(sparkContext = sc.sparkContext, sparkSession = sc)


def load_data(path_data, pkg_for_df='spark'):
    '''import the raw data.
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
        package - the package used for loading the csv.gz file, including spark and dd (dask)
    '''
    # load raw data
    path_datafile = glob.glob(path_data + "/*.csv")[0:2]
    # print(path_datafile)
    if pkg_for_df == 'spark':
       # df = sqlContext.read.option("delimiter", "\t").csv(path_datafile)
        df = sqlContext.read.csv(path_datafile, header = False)  
    else:
        df_from_each_file = (pd.read_csv(f) for f in path_datafile)
        df = pd.concat(df_from_each_file, ignore_index = True)     
    # rename columns
    col_names_old = df.columns
    col_names_new = ['index','id_str', 'label', 'start', 'end', 'latitude',
                     'longitude', 'date', 'start_h', 'end_h', 'stay_t_mins',
                     'travel_t_mins', 'travel_dist_m', 'travel_bearing']
    if pkg_for_df == 'spark':
        for i in range(len(col_names_old)):
            df = df.withColumnRenamed(col_names_old[i], col_names_new[i])
    return df

'''
Select users that have at least two different stop points per day.
This can be done immediately after merging the consecutive identical stop points
'''
# def unique_users(dir_processed_data, df, readin=True):
#     if readin:
#         df_user = pd.read_csv(dir_processed_data+'users/user_list.csv')
#     else:
#         threshold_record = 2
#         df_user = df.groupBy("id_str").count().toPandas()
#         df_user = df_user[df_user['count']>=threshold_record]
#         df_user.to_pickle(dir_processed_data+'users/user_list.pkl')
#     print('user_cnt:', len(df_user))
#     return df_user

def remove_single_record_user(self, df, n_min_record=2):
    row_group_sizes = df['id_str'].groupby('id_str').transform('size')
    df_new = df[row_group_sizes>=n_min_record]
    return df_new
   
  
def relabel_stop_points(self, df, max_dist_betwn_point=30):
    # remove users with fewer than two entries in a day
    df = self.remove_single_record_user(df, n_min_record=2)
    # change to float
    df[['start','end', 'latitude', 'longitude']] = df[['start', 'end', 'latitude', 'longitude']].astype(float)    
    # # select a few to test
    # n_select = 1000
    # user_list = df_user['id_str'][:n_select]
    
    # relabel stop points of a single user
    def relabel_by_group(group):
        ''' relabel the stop points of each individual's df
            find home location of each individual
        '''
        coord_arr = np.array(group[['latitude','longitude']].values)
        # get labels of locations of single individual
        model_infostop = infostop.SpatialInfomap(r2=max_dist_betwn_point,
                                                 label_singleton=True,
                                                 min_spacial_resolution=0.0001,
                                                 distance_metric='haversine',            
                                                 verbose=False) ###only true for testing
        label_list = model_infostop.fit_predict(coord_arr)        
        return label_list 

    def find_home_by_group(group):
        home_lat,home_lon,county,state = find_home_loc(group)
        user_id = group['id_str'].value[0]
        return [user_id, home_lat,home_lon,county,state]
    # relabel stop points
    label_list = df.groupby('id_str').apply(relabel_by_group)
    df['label'] = label_list
    # get each user's home location
    user_home_list = df.groupby('id_str').apply(relabel_by_group)
    df_user_home = pd.DataFrame(user_home_list, columns=['id_str','home_lat','home_lon','county','stte'])  
    return df, df_user_home

####here we could relabel separately for each user
def main():

    dir_raw_data = '../data/mobility_data_6_month/'
    dir_processed_data = '../data_processed/stop_points/'
    
    mobility_data = MobilityData(dir_raw_data, dir_processed_data, is_save=True)
    mobility_data.merge_data_single_date()    
        
    pkg_for_df = 'spark'
    df = load_data(dir_processed_data, pkg_for_df)
    
    ####read user_list
    df_user = unique_users(dir_processed_data,df,read = True)
    
    ###begin relabel
    from time import time
    t0 = time()
    df, df_indiv_loc = stop_points_relabel(df, df_user)
    df.to_pickle(dir_processed_data+'after_relabel/All_Data_stoppoints.pkl')
    df_indiv_loc.to_pickle(dir_processed_data+'after_relabel/All_Data_indiv_loc.pkl')
    t1 = time()
    t_diff = t1 - t0
    print('\n===== Completed relabeling. Time elapsed: {}'.format(t_diff))



######################
if __name__ == 'main':
    main()
    

