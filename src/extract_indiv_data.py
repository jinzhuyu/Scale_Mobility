# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:48:13 2021

@author: Jin-Zhu Yu
"""

import numpy as np
import pandas as pd
import os
os.chdir("c:/code/Scale_Mobility/src")
from my_utils import *
# import pickle
import glob

# import infostop
# from datetime import datetime
from time import time
from joblib import Parallel, delayed
from geopy.geocoders import Nominatim    #https://geopy.readthedocs.io/en/stable/#nominatim




class MobilityData:
    '''  
    Our objective:
        combine data in a day
        merge consercutive identify stop points
        Extract home locations
        Relabel the locations
        Store the data for say 10000 individual in one file (preferably a pickle file for faster readin)
    Data:
        Data are stored in the path: All data/ folder for each day/thousands of csv files
        each csv file contains the mobility trajectories of thousands of invididuals: id_str, loc id, time and coords of locations
        'id_str', 'label', 'start', 'end', 'lat_start', 'lon_start', 'lat_end', 'lon_end'
        Note that the location id for different dates are independent. That's why we also need to relabel the locations.      
    '''
    def __init__(self, dir_raw_data, dir_processed_data,
                 night_start=18, night_end=8,
                 max_dist_betwn_point=30,
                 is_save=False):
        self.dir_raw_data = dir_raw_data
        self.dir_processed_data = dir_processed_data
        # self.r2 = r2    # min distances between two different stop points
        self.night_start = night_start
        self.night_end = night_end
        self.max_dist_betwn_point = max_dist_betwn_point
        self.is_save = is_save
    
    def merge_consec_stopoint(self, df_temp):
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
   
    def remove_single_record_user(self, df, n_min_record=2):
        '''
        Select users that have at least two different stop points per day.
        This can be done immediately after merging the consecutive identical stop points
        '''        
        row_group_sizes = df['id_str'].groupby('id_str').transform('size')
        df_new = df[row_group_sizes>=n_min_record]
        return df_new

    # relabel stop points of a single user
    def relabel_by_group(self, group):
        ''' relabel the stop points of each individual's df
            find home location of each individual
        '''
        coord_arr = np.array(group[['latitude','longitude']].values)
        # get labels of locations of single individual
        model_infostop = infostop.SpatialInfomap(r2=self.max_dist_betwn_point,
                                                 label_singleton=True,
                                                 min_spacial_resolution=0.0001,
                                                 distance_metric='haversine',            
                                                 verbose=False) ###only true for testing
        label_list = model_infostop.fit_predict(coord_arr)        
        return label_list         
    
    def relabel_all_group(self, df, max_dist_betwn_point=30):
        # remove users with fewer than two entries in a day
        df = self.remove_single_record_user(df, n_min_record=2)
        # change to float
        df[['start','end', 'latitude', 'longitude']] = df[['start', 'end', 'latitude', 'longitude']].astype(float)         
        # relabel stop points
        label_list = df.groupby('id_str').apply(self.relabel_by_group)
        df['label'] = label_list
        return df
    
    def find_home_loc(self, df_temp):    
        df_temp_night = df_temp[(pd.to_datetime(df_temp['start'], unit='s').dt.hour>=self.night_start)&\
                            (pd.to_datetime(df_temp['end'], unit='s').dt.hour<=self.night_end)]
        if len(df_temp_night)==0:
            lat, lon, county, state = None, None, None, None
        else:
            (lat, lon) = df_temp_night.groupby(['latitude', 'longitude']).size().idxmax()
            geolocator = Nominatim(user_agent="geoapiExercises")
            location = geolocator.reverse(str(lat) + "," + str(lon))
            address = location.raw['address']
            county= address.get('county', '')           
            state = address.get('state', '')
          # city = address.get('city', '')
          # zipcode = address.get('postcode') 
        return lat, lon, county, state
        
    def find_home_by_group(self, group):
        home_lat,home_lon,county,state = self.find_home_loc(group)
        user_id = group['id_str'].value[0]
        return [user_id, home_lat, home_lon, county, state]
      
    def find_home_all_group(self, df):
        # get each user's home location
        user_home_list = df.groupby('id_str').apply(self.find_home_by_group)
        df_user_home = pd.DataFrame(user_home_list, columns=['id_str','home_lat','home_lon','county','stte'])    
        return df_user_home
    
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
        
    def process_all_date(self):
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
                df_temp_merged = self.merge_consec_stopoint(df_temp)
                return df_temp_merged
            df_list = Parallel(n_jobs=8,verbose=0)(delayed(loop_over_file)(fname) for fname in file_list)
            df = pd.concat(df_list, ignore_index=True)
            df = df[['id_str', 'label', 'start', 'end','latitude', 'longitude']]
            t1 = time()
            print('\n===== Time for merging:{}'.format(t1-t0))
            
            # remove users with just one record
            t0 = time()
            df = self.remove_single_record_user(df)
            t1 = time()
            print('\n===== Time for remove users with a single record:{}'.format(t1-t0))            
            
            # relabel
            t0 = time()
            df = self.relabel_all_group(df)
            t1 = time()
            print('\n===== Time for relabeling:{}'.format(t1-t0))              
            
            # find user home location
            t0 = time()
            df_user_home = self.find_home_all_group(df)
            t1 = time()
            print('\n===== Time for finding home location:{}'.format(t1-t0))            
            
            # get other movement feature: travel time and dist, stay time, travel angle.
            t0 = time()
            df = self.get_movement_feature(df)
            t1 = time()
            print('\n===== Time for extract other movement features:{}'.format(t1-t0))             
            
            # save the processed data
            t0 = time()
            if self.is_save:
                # saving and loading pickle files are significantly faster than csv files
                df.to_pickle(self.dir_processed_data +'{}.pkl'.format(this_date))
                df_user_home.to_pickle(self.dir_processed_data+'user_home_location.pkl')
                del df, df_user_home
            t1 = time()
            print('===== Time for saving:{}'.format(t1-t0))
        
        folder_list = list(os.listdir(self.dir_raw_data))    #[-3:]   
        [loop_over_date(folder) for folder in folder_list]   # returns list of 'None'. A bit faster than 'for loop' and 'any'
        

#####
def main():
    dir_raw_data = '../data/mobility_data_6_month/'
    dir_processed_data = '../data_processed/stop_points/'
    
    mobility_data = MobilityData(dir_raw_data, dir_processed_data, is_save=True)
    t0 = time()
    mobility_data.process_all_date()
    t_total = t1 - t0
    print('\n===== Completed all. Time elapsed: {}'.format(t_total))
    

######################
if __name__ == 'main':
    main()
    

