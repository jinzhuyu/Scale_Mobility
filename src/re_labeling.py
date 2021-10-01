#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('###########import packages')

import sys
sys.path.append('/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/codes/')
import os
os.chdir("/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/")
import glob
from input_library import *
import utils
print('done')


# In[2]:


import infostop


# # Section 1. Relabelling
# 
# 
# First, relbel_all
# Secod, filter out
# Third, home locations
# Fourth, re_scale

# In[15]:


#####input 
from datetime import datetime

path_stoppoints='/Volumes/SeagateDrive/stoppoints/'
path_summary='/Volumes/SeagateDrive/stoppoints_summary/'


def consecutive_merge(df_temp):
    df_temp['temp_cluster']=np.arange(len(df_temp))
    for index,row in df_temp.iterrows():
        if index>0:
            if row['id_str']==df_temp['id_str'][index-1] and row['label']==df_temp['label'][index-1]:
                df_temp['temp_cluster'][index]=df_temp['temp_cluster'][index-1]
    
    df_temp = df_temp.groupby(['id_str','label','temp_cluster']).agg({'start': ['min'],'end': ['max'],'latitude': ['mean'],'longitude': ['mean']})
    
    df_temp.columns = ['start', 'end', 'latitude','longitude']
    df_temp = df_temp.reset_index()
    return  df_temp[['id_str','label','start', 'end', 'latitude','longitude']]

#####all stoppoints
directories= list(os.listdir(path_stoppoints))
directories.remove('.DS_Store')
df_list=[]
for directory in directories[1:2]:
    print(directory)
    file_list= list(os.listdir(path_stoppoints+directory))
    for file in file_list:
        print(file)
        df_temp=pd.read_csv(path_stoppoints+directory+'/'+file)
        df_temp['latitude']=(df_temp['lat_start']+df_temp['lat_end'])/2
        df_temp['longitude']=(df_temp['lon_start']+df_temp['lon_end'])/2
        df_temp=consecutive_merge(df_temp)
        df_list.append(df_temp)
    df=pd.concat(df_list)
    df=df[['id_str', 'label', 'start', 'end','latitude', 'longitude']]
    df.to_csv(path_summary+directory+'.csv')
print('done')


# In[78]:


print(df.head(1))


# In[9]:


from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap as Basemap

def finding_home_locations(df_temp):
    start=18;end=8; #####night time
    df_temp['start_h'] = df_temp['start_h'].astype(float)
    df_temp['end_h'] = df_temp['end_h'].astype(float)
    df_temp_x=df_temp[(df_temp['start_h']>=start)&(df_temp['end_h']<=end)]
    if len(df_temp_x)==0:
        lat=None
        lon=None
        county=None
        state=None
    else:
        (lat,lon)=df_temp_x.groupby(['latitude', 'longitude']).size().idxmax()
        county,state=finding_home_city(lat,lon)
    return lat,lon,county,state
    
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


# In[12]:




import infostop
from functools import reduce
from datetime import datetime
#####Here we use pyspark
from pyspark.sql import SparkSession
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, length,monotonically_increasing_id,udf,row_number
from pyspark.sql import types as T
from pyspark.sql.window import Window

from pyspark.serializers import MarshalSerializer
# #set an application name
# #start spark cluster; if already started then get it else create it
sc = SparkSession.builder.appName('data_preprocess').master("local[*]").getOrCreate()
# initialize SQLContext from spark cluster
sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)

path_summary='/Volumes/SeagateDrive/stoppoints_summary/'


def consecutive_merge(df_temp):
    df_temp['temp_cluster']=np.arange(len(df_temp))
    for index,row in df_temp.iterrows():
        if index>0:
            if row['id_str']==df_temp['id_str'][index-1] and row['label']==df_temp['label'][index-1]:
                df_temp['temp_cluster'][index]=df_temp['temp_cluster'][index-1]
    
    df_temp = df_temp.groupby(['id_str','label','temp_cluster']).agg({'start': ['min'],'end': ['max'],'latitude': ['mean'],'longitude': ['mean']})
    
    df_temp.columns = ['start', 'end', 'latitude','longitude']
    df_temp = df_temp.reset_index()
    return  df_temp[['id_str','label','start', 'end', 'latitude','longitude']]


def stopoints_interval_process(df_temp):
    
    interval_mat=[]
    for index, row in df_temp.iterrows():
        stay_t=(datetime.fromtimestamp(row['end'])-datetime.fromtimestamp(row['start'])).total_seconds()/60
        start_h=datetime.fromtimestamp(row['start']).hour
        end_h=datetime.fromtimestamp(row['end']).hour
        date=datetime.fromtimestamp(row['start']).strftime('%Y%m%d')
        travel_t=-1;travel_d=-1;travel_a=-1;
        if index<len(df_temp)-1:
            row2=df_temp.iloc[index+1]
            if row2['id_str']==row['id_str']: 
                travel_t=(datetime.fromtimestamp(row2['start'])-datetime.fromtimestamp(row['end'])).total_seconds()/60 
                travel_d,travel_a=utils.haversine(row['latitude'],row['longitude'],row2['latitude'],row2['longitude'])
        interval_mat.append([stay_t,start_h,end_h,travel_t,travel_d*1000,travel_a,date])
    interval_mat=np.array(interval_mat)
    df_temp['stay_t(min)']=interval_mat[:,0]
    df_temp['start_h']=interval_mat[:,1]
    df_temp['end_h']=interval_mat[:,2]
    df_temp['travel_t(min)']=interval_mat[:,3]
    df_temp['travel_d(m)']=interval_mat[:,4]
    df_temp['travel_bearing']=interval_mat[:,5]
    df_temp['date']=interval_mat[:,6]
    return df_temp


def load_data(path_data='..',package_for_df='spark'):
    '''import the raw data.
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
        package - the package used for loading the csv.gz file, including spark and dd (dask)
    '''
    # load raw data
    path_datafile = glob.glob(path_data + "/*.csv")[0:2]
    print(path_datafile)
    if package_for_df == 'spark':
       # df=sqlContext.read.option("delimiter", "\t").csv(path_datafile)
        df = sqlContext.read.csv(path_datafile,header=False)
        
    else:
        df_from_each_file = (pd.read_csv(f) for f in path_datafile)
        df  = pd.concat(df_from_each_file, ignore_index=True)
        
        
        
    ####rename columns
    col_names_old = df.columns
    col_names_new = ['index','id_str', 'label', 'start', 'end', 'latitude',
       'longitude', 'date', 'start_h', 'end_h', 'stay_t(min)', 'travel_t(min)',
       'travel_d(m)', 'travel_bearing']
    if package_for_df == 'spark':
        for i in range(len(col_names_old)):
            df = df.withColumnRenamed(col_names_old[i], col_names_new[i])
    return df



def unique_users(path_summary,df,read=True):
    if read==False:
        threshold_record=1
        df_user=df.groupBy("id_str").count().toPandas()
        df_user=df_user[df_user['count']>threshold_record]
        df_user.to_csv(path_summary+'users/user_list.csv')
    if read==True:
        df_user=pd.read_csv(path_summary+'users/user_list.csv')
    print('user_cnt:',len(df_user))
    return df_user

    


def stop_points_relabel(df,df_user):
    user_list=df_user['id_str'][0:10000]
    
    dfArray = [df.where(df.id_str == x) for x in user_list ]
    mat_demographic=[]
    for i in range(len(user_list)):
        print(i)
        df_temp=dfArray[i]
        df_temp=df_temp.toPandas()
        
        #######begin relable 
        df_temp['end'] = df_temp['end'].astype(float)
        df_temp['start'] = df_temp['start'].astype(float)
        df_temp['latitude'] = df_temp['latitude'].astype(float)
        df_temp['longitude'] = df_temp['longitude'].astype(float)
        array_list=np.array(df_temp[['latitude','longitude']].values)
        print(array_list.shape)
        r2=30
        model_infostop = infostop.SpatialInfomap(r2 =r2,
                                label_singleton=True,
                                min_spacial_resolution = 0.0001,
                                distance_metric ='haversine',            
                                verbose = False) ###only true for testing

        labels_list = model_infostop.fit_predict(array_list)
        print(labels_list)
        df_temp['label']=labels_list
        #######begin compute intervals
        df_temp=consecutive_merge(df_temp)
        df_temp=stopoints_interval_process(df_temp)
        dfArray[i]=df_temp
        home_lat,home_lon,county,state=finding_home_locations(df_temp)
        print(home_lat,home_lon,county,state)
        mat_demographic.append([df_user['id_str'][i],home_lat,home_lon,county,state])
        
        #print(array_list.shape,len(labels_list))
    
    df = pd.concat(dfArray)
    df_demographic=pd.DataFrame(np.array(mat_demographic),columns=['id_str','home_lat','home_lon','county','stte'])
    return df,df_demographic


####here we could relabel separately for each user

package_for_df = 'spark'
df = load_data(path_summary,package_for_df)

####read user_list
df_user=unique_users(path_summary,df,read=True)

###begin relabel
df,df_demographic=stop_points_relabel(df,df_user)

df.to_csv(path_summary+'after_relabel/All_Data_stoppoints.csv')
df_demographic.to_csv(path_summary+'after_relabel/All_Data_demographic.csv')
print('done')


# In[ ]:




