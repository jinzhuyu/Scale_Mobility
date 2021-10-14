import pickle
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import infostop
import sys
sys.stdout.flush()
import multiprocessing as mp
import process_utils
from functools import partial
import time
from multiprocessing import Pool
import glob
from datetime import datetime
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance and bearing bewteen (lat1,lon1) and (lat2,lon2)
    """

    dL = lon2 - lon1
    X = math.cos(lat2) * math.sin(dL)
    Y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dL)
    bearing = np.arctan2(X, Y)
    bearing = np.degrees(bearing)

    R = 6373.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c
    return d, bearing  # in kilometers



def relabel_process(df_individual):
    df_individual['end'] = df_temp['end'].astype(float)
    df_individual['start'] = df_temp['start'].astype(float)
    df_individual['latitude'] = df_temp['latitude'].astype(float)
    df_individual['longitude'] = df_temp['longitude'].astype(float)
    array_list=df_temp[['latitude','longitude']].values

    r2=30
    model_infostop = infostop.SpatialInfomap(r2 =r2,
                            label_singleton=True,
                            min_spacial_resolution = 0.0001,
                            distance_metric ='haversine',
                            verbose = False) ###only true for testing

    labels_list = model_infostop.fit_predict(array_list)
    df_individual['label']=labels_list
    return df_individual
    
    
def interval_process(df_individual):
    interval_mat=[]
    for index, row in df_individual.iterrows():
        stay_t=(row['end']-row['start'])/3600
        start_h=datetime.fromtimestamp(row['start']).hour
        end_h=datetime.fromtimestamp(row['end']).hour
        month=datetime.fromtimestamp(row['start']).month
        date=datetime.fromtimestamp(row['start']).strftime('%Y%m%d')
        travel_t=-1;travel_d=-1;travel_a=-1;
        if index<len(df_individual)-1:
            row2=df_individual.iloc[index+1]
            if row2['id']==row['id']:
                travel_t=(row2['start']-row['end'])/3600
                travel_d,travel_a=haversine(row['latitude'],row['longitude'],row2['latitude'],row2['longitude'])
        interval_mat.append([stay_t,start_h,end_h,travel_t,travel_d*1000,travel_a,date,month])
    interval_mat=np.array(interval_mat)
    df_individual['stay_t(h)']=interval_mat[:,0]
    df_individual['start_h']=interval_mat[:,1]
    df_individual['end_h']=interval_mat[:,2]
    df_individual['travel_t(h)']=interval_mat[:,3]
    df_individual['travel_d(km)']=interval_mat[:,4]
    df_individual['travel_bearing']=interval_mat[:,5]
    df_individual['date']=interval_mat[:,6]
    df_individual['month']=interval_mat[:,7]
    
    return df_individual




def consecutive_merge(df_individual):
    df_individual['temp_cluster']=np.arange(len(df_individual))
    df_individual=df_individual.reset_index()
    for index,row in df_individual.iterrows():
        if index>0:
            if row['id']==df_individual['id'][index-1] and row['label']==df_individual['label'][index-1]:
                df_individual['temp_cluster'][index]=df_individual['temp_cluster'][index-1]
    
    df_individual = df_individual.groupby(['id','label','temp_cluster']).agg({'start': ['min'],'end': ['max'],'latitude': ['mean'],'longitude': ['mean']})
    
    df_individual.columns = ['start', 'end', 'latitude','longitude']
    df_individual = df_individual.reset_index()
    return  df_individual[['id','label','start', 'end', 'latitude','longitude']]
    
    
def finding_home_locations(df_individual):
    start=18;end=8; #####night time
    monthly_home=[]
    idx=df_individual['id'].values[0]
    for month in range(1,7):
        df_temp=df_individual[df_individual['month']==month]
        df_temp=df_temp[(df_temp['start_h']>=start)&(df_temp['end_h']<=end)]
        lat=None;
        lon=None;
        if len(df_temp)>5:
            print(len(df_temp))
            (lat,lon)=df_temp.groupby(['latitude', 'longitude']).size().idxmax()
        monthly_home.append([idx,month,lat,lon])
    return monthly_home
    
    
def finding_home_city(Latitude,Longitude):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(str(Latitude) + "," + str(Longitude))
    address = location.raw['address']
    county= address.get('county', '')
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    zipcode = address.get('postcode')
    #print(county,",",city,",",state,",",country,",",zipcode)
    return county,state
    
if __name__ == "__main__":

    start_time = time.time()
    ####pareameters from .sh file
    begin_id = int(sys.argv[1])
    end_id= int(sys.argv[2])
    print(begin_id,end_id)
    
    
    ######read input data
    path_stoppoints = '/gpfs/u/scratch/COVP/shared/stoppoints/'
    path_merge_stoppoints='/gpfs/u/scratch/COVP/shared/stoppoints_merge/'
    
    #####output data
    path_individual_stoppoints='/gpfs/u/scratch/COVP/shared/stoppoints_individual/'
    
    with open(path_merge_stoppoints+'greater30days_ids_record_dict.pickle', 'rb') as handle:
        dict_id_record= pickle.load(handle)
        
    #####subset of dictionary
    dict_id_record={i: dict_id_record[i] for i in list(dict_id_record.keys())[begin_id:end_id]}
    
    #####change the dictionary
    date_id_record=defaultdict()
    for id,value in dict_id_record.items():
        for date, record in value.items():
            if date not in date_id_record.keys():
                date_id_record[date]=dict()
            date_id_record[date][id]=record
    #print(date_id_record)
    
    ###extract##
    df_individual=pd.DataFrame()
    for date, value in date_id_record.items():
        df_date=pd.read_csv(path_stoppoints+date+"_merge.csv")
        record_index_list=[]
        for id, record in value.items():
            #print(date,id,record)
            record_index=list(map(int, record.strip('][').split(',')))
            record_index_list=record_index_list+record_index
        df_temp=df_date.loc[record_index_list]
        df_temp=df_temp[['id','label','start', 'end', 'latitude','longitude']]
        df_individual=pd.concat([df_individual,df_temp])
        del [df_date]
        break
    df_individual=df_individual.reset_index()
    df_individual=df_individual.sort_values(by='id')
    df_individual=df_individual.sort_values(by='start')
    df_individual.to_csv(path_individual_stoppoints+str(begin_id)+'_'+str(end_id)+'_stoppoints.csv')
    
    print('all individuals',len(df_individual))
    end_time = time.time()
    print('minutes need',(end_time-start_time)/60)
    ########compute each id's data##
    df_all=pd.DataFrame()
    df_all_home=pd.DataFrame()
    
    for idx, df_temp in df_individual.groupby('id'):
        #print('id',idx)
        df_temp=df_temp.sort_values(by='start')
        ####relabel
        df_temp=relabel_process(df_temp)

        ####merge same
        #df_temp=consecutive_merge(df_temp)

        ####interval computes
        df_temp=interval_process(df_temp)
        
        ####montly home_locations
        monthly_home=finding_home_locations(df_temp)
        
        ####save
        df_all=pd.concat([df_all,df_temp])
        df_home=pd.DataFrame(np.array(monthly_home),columns=['id','month','home_lat','home_lon'])
        df_all_home=pd.concat([df_all_home,df_home])

        print(len(df_all),len(df_all_home))
        
    df_all=df_all.reset_index()
    df_all_home=df_all_home.reset_index()
    df_all.to_csv(path_individual_stoppoints+str(begin_id)+'_'+str(end_id)+'_stoppoints.csv')
    df_all_home.to_csv(path_individual_stoppoints+str(begin_id)+'_'+str(end_id)+'_home.csv')
    
    print('done')
    end_time = time.time()
    print('minutes need',(end_time-start_time)/60)
    
    


      
            
