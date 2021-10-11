# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:48:13 2021

@author: Jin-Zhu Yu
"""
import numpy as np
import pandas as pd
import os
os.chdir('c:/code/Scale_Mobility/src/')
from glob import glob
from time import time
from h3 import geo_to_h3

class MobilityData:
    '''  
    objective:
        combine data in a day
        merge consercutive identify stop points
        Extract home locations
        Find location label using h3
        Store the data for each day in one file (preferably a pickle file for faster readin)
    Data:
        Data are stored in the path: All data/ folder for each day/thousands of csv files
        each csv file contains the mobility trajectories of thousands of invididuals: id_str, loc id, time and coords of locations
        'id_str', 'label', 'start', 'end', 'lat_start', 'lon_start', 'lat_end', 'lon_end'
        Note that the location id for different dates are independent. That's why we also need to relabel the locations.      
    '''
    def __init__(self, dir_raw_data, dir_processed_data, dir_user_record,
                 resol=12, n_min_record_daily=2, is_save=False, verbose=False):
        self.dir_raw_data = dir_raw_data
        self.dir_processed_data = dir_processed_data
        self.dir_user_record = dir_user_record
        self.resol = resol  # resolution of GeoToH3 that determines the size of a hexegon/location
        self.n_min_record_daily = n_min_record_daily
        self.is_save = is_save
        self.verbose = verbose
        
        self.user_str_df = self.import_user_record()

        
    def import_user_record(self):
        user_str_df = pd.read_pickle(self.dir_user_record)
        for col in ['Unnamed: 0', 'Unnamed: 0.1']:
            if col in user_str_df.columns:
                user_str_df = user_str_df.drop([col], axis=1)
        user_str_df['id_int'] = range(len(user_str_df))
        return user_str_df
        
        
    def id_str2int(self, df):
        df['id_str'] = df['id_str'].map(self.user_str_df.set_index('id_str')['id_int'])
        return df
    
    
    def merge_consec_stopoint(self, df):
        '''
        Merge consecutive stop points that are actually the same point.
            For example, consecutive entries 13 to 16 all have location label 2, thse entries should be merged into one entry.
            The start time (use min) and end time (use max) as well as start coords (use mean) and end coords (use mean) are updated as well
        Params
        ----------
        df: a df that includes the trajectories of all csv files (several million individuals for about 6 months)
        '''
        # assign the same group id if the values of id str and label are the same
        df_is_value_same = (df[['id_str','label']] == df.shift().fillna(method='bfill')[['id_str', 'label']])
        # use the same group id number only if both values are true
        temp_group = (df_is_value_same.sum(axis=1)<2).cumsum().rename('temp_group')
        # group by id stri, label and id of temp group
        df_merged = df.groupby(['id_str','label',temp_group], sort=False).agg({'start': ['min'],
                                                                               'end': ['max'],
                                                                               'latitude': ['mean'],
                                                                               'longitude': ['mean']})
        # rename and select 
        df_merged.columns = ['start', 'end', 'latitude','longitude']
        df_merged = df_merged.reset_index()      
        return df_merged      
    
        
    def remove_single_record_user(self, df):
        '''
        Select users that have at least two different stop points per day.
        This can be done immediately after merging the consecutive identical stop points
        '''        
        df = df[df.groupby('id_str')['id_str'].transform('size') >= self.n_min_record_daily]
        return df
    
    
    # relabel stop points of a single user
    def relabel_loc(self, df):
        ''' relabel the stop points of each individual's df
            find home location of each individual
        '''
        # to_numpy for pandas > 0.24.0
        # lats = df['latitude'].to_numpy(dtype=np.float64)
        # lngs = df['longitude'].to_numpy(dtype=np.float64)
        lats = df['latitude'].values
        lons = df['longitude'].values
        resolution = 12  # in [0,15]; finest resolution is 15
        label_list = [geo_to_h3(x[0], x[1], resolution) for x in zip(lats, lons) ]
        df['label'] = label_list
        return df         
    
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the earth distance and bearing in degree between two points 
        on the earth (specified in decimal degrees)
        """
        #check input type
        if (not isinstance(lon1, list)) and (not isinstance(lon1, pd.Series)):
            raise TypeError("Only list or pandas series are supported as input coordinates")
        #Convert decimal degrees to Radians:
        lon1, lat1 = np.radians(lon1), np.radians(lat1)
        lon2, lat2 = np.radians(lon2), np.radians(lat2)
        #Implementing Haversine Formula: 
        dlon, dlat = np.subtract(lon2, lon1), np.subtract(lat2, lat1)
    
        a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                   np.multiply(np.cos(lat1), 
                               np.multiply(np.cos(lat2), 
                                           np.power(np.sin(np.divide(dlon, 2)), 2))))
        c = np.multiply(2, np.arcsin(np.sqrt(a)))
        r = 6371*1e3  # global average radius of earth in m. Use 3956 for miles.
        dist = c*r
        
        bearing = np.arctan2(np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon),
                             np.sin(dlon)*np.cos(lat2)) 
        bearing = np.degrees(bearing)
        bearing_deg = (bearing + 360) % 360
        return np.round(dist, 4), np.round(bearing_deg, 4)
    
    
    def get_other_feature(self, df):
        '''Get travel time, stay time, travel distance, travel bearing
        '''
        # stay time
        df['stay_time'] = df['end'] - df['start'] 
        # travel time 
        df_shift = df.shift().fillna(method='bfill')    
        df['travel_time'] = df['start'] - df_shift['end']
        # dist and angle
        df['travel_dist'], df['travel_angle'] = self.haversine(df['latitude'], df['longitude'],
                                                               df_shift['latitude'], df_shift['longitude'])
        # set to nan when the consercutive rows are not of the same user  
        # and first row
        df = df.reset_index(drop=True)
        df.loc[0, ['travel_time', 'travel_dist', 'travel_angle']] = np.nan
        
        df_shift = df.shift().fillna(method='bfill')
        df_is_user_same = (df['id_str'] == df_shift['id_str'])
        cols = ['travel_time', 'travel_dist', 'travel_angle']
        df[cols] = df[cols].mask(df_is_user_same==False, np.nan)
        return df
       
    
    def loop_over_date(self, this_date):
        t1 = time()
        csv_list = glob(self.dir_raw_data + this_date + '/*.csv')   
        df = pd.concat([pd.read_csv(file) for file in csv_list], ignore_index=True)
        t2 = time()
    
        # drop the useless column
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1)
        # get the center of stop points
        df['latitude'] = (df['lat_start']+df['lat_end'])/2
        df['longitude'] = (df['lon_start']+df['lon_end'])/2
        
        # replace id str with integer
        df = self.id_str2int(df)
        
        df_merge = self.merge_consec_stopoint(df)
        df_merge = df_merge[['id_str', 'label', 'start', 'end','latitude', 'longitude']]    
    
        # remove users with just one record
        t3 = time()
        df = self.remove_single_record_user(df)
        t4 = time() 
        
        # get other mobility feature: travel time and dist, stay time, travel angle.
        df = self.get_other_feature(df) 
        
        # relabel
        t5 = time() 
        df = self.relabel_loc(df)
        t6 = time() 
        # save the processed data
        if self.is_save:
            # saving and loading pickle files are significantly faster than csv files
            df.to_pickle(self.dir_processed_data +'{}.pkl'.format(this_date))
            df.to_csv(self.dir_processed_data +'{}.csv'.format(this_date))
            # del df
        t7 = time()
    
        if self.verbose:    
            print('\n===== Time for importing all csv files:  {} seconds'.format(round(t2-t1, 1)) )
            print('===== Time for merging:  {} seconds'.format(round(t3-t2, 1)) )    
            print('===== Time for removing single-record users:  {} seconds'.format(round(t4-t3, 1)) )     
            print('===== Time for relabeling:  {} seconds'.format(round(t6-t5, 1)) )
            print('===== Time for getting other features:  {} seconds'.format(round(t5-t4, 1)) )  
            print('===== Time for saving as csv and pkl:  {} seconds'.format(round(t7-t6, 1)) )
            print('===== Total time:  {} seconds'.format(round(t7-t1,1)) )  

    def process_all_date(self):
        date_list = list(os.listdir(self.dir_raw_data))    #  [:2]   # [-3:]
        [self.loop_over_date(date) for date in date_list]     # returns list of 'None'. A bit faster than 'for loop' and 'any'

#####
def main():
    
    dir_raw_data = '../data/mobility_data_6_month/'
    dir_processed_data = '../data_processed/stop_points/'
    dir_user_record = '../data/user_record.pkl'
    is_save = True
    
    mobility_data = MobilityData(dir_raw_data, dir_processed_data, dir_user_record,
                                 resol=12, n_min_record_daily=2, is_save=is_save, verbose=False)
    t0 = time()
    mobility_data.process_all_date()
    t1 = time()
    t_total = t1 - t0
    print('\n===== Completed all. Time elapsed: {}'.format(t_total))
    

if __name__ == '__main__':
    main()
    

