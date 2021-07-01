# -*- coding: utf-8 -*-
"""
@author: Jinzhu Yu
"""
###
import numpy as np
import pandas as pd
from datetime import datetime
from functools import reduce
import os
from pathlib import Path

import infostop
import utils

# # import and set up pyspark configuration
# import findspark
# findspark.init()

# os.environ["SPARK_HOME"] = "/home/spark-1.6.0"

# from pyspark.sql import SparkSession
from pyspark.sql import*
from pyspark.sql.types import*
from pyspark.sql.functions import col, length  #, lit

# #set an application name 
# #start spark cluster; if already started then get it else create it 
sc = SparkSession.builder.appName('data_preprocess').master("local[*]").getOrCreate()
#initialize SQLContext from spark cluster 
sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)


###################
# class TrajData:
    
# traj_data = TrajData()
# traj_data.main()
    
def load_data(path_data='../data', location='Albany', date='20200207'):
    '''import the raw data.
    
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
    '''
        
    #load raw data
    os.chdir(path_data)
    # path_datafile = os.path.join(os.getcwd(), '{}/{}00.csv.gz'.format(location, date))
    path_datafile = os.path.join(os.getcwd(), '{}/*.csv.gz'.format(location)) #load all csv.gz files at once
    
    df = sqlContext.read.csv(path_datafile, header=False)

    #change back to the default trajectory
    os.chdir('../src')

    return df

# df_of_indiv = sqlContext.read.csv('/home/jayz/Documents/GitHub/Scale_Mobility/data/data_indiv/indiv_1.csv', header=True)

def rename_col(df):
    '''rename the columns. The columns in the resultant df from loading data with spark are all of string type.
       note: 'time' and 'time_original' are in unix timestamp format (integer value)
       ## change column type
           # # check out data type of each column
           # df.printSchema()  # all string, column names are '_c0' to '_c9'
           # # show top 5 rows:
               # df.show(5)
    '''
          
    # rename the columns
    col_names_old = df.columns
    col_names_new = ['time_original','id_str','device_type','latitude','longitude','accuracy','timezone','class','transform','time']

    for i in range(len(col_names_old)):
        df = df.withColumnRenamed(col_names_old[i], col_names_new[i])

        
    # change column type
    schema_new = [IntegerType(), StringType(), IntegerType(), FloatType(), FloatType(),
                  FloatType(), IntegerType(), StringType(), StringType(), IntegerType()]
    for i in range(len(col_names_new)):
        df = df.withColumn(col_names_new[i], df[col_names_new[i]].cast(schema_new[i]))
              
    return df 
   

def select_col(df):
    '''select columns: id_str, time, latidude, and longitude, accuracy
    '''
    
    col_select = ['time', 'latitude', 'longitude', 'id_str']
    df_select = df.select(*col_select)
    
    return df_select

    
def remove_error_entry(df):
    
    '''remove entries with erreneous value of coordinates and 'id_str'.
       There can be errors in the latitude or longitude. E.g., the min of latitude is -14400
    '''
      
    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180
    id_str_len_min = 15

    df = df.filter((df.latitude<=lat_max) & (df.latitude>=lat_min) &
                   (df.longitude<=lon_max) & (df.longitude>=lon_min))

    df = df.filter(length(col('id_str')) > id_str_len_min)
    
    return df
    

def save_df_traj(df, location='Albany', date='20200316'):
    '''save the processed df to the folder for the current location
         within 'data_processed' folder, which is on the same level of the 'data' folder'
    '''

    # create a new folder if the 'data_processed' folder or the folder for the
        # current location doesn't exist 
    path_parent = str(Path(os.getcwd()).parent)  # get parent path of src
    path_data_processed =  '../data_processed'
    path_location =  '../data_processed/{}'.format(location)  # path for the data for current location/place/city
    
    if not os.path.isdir(path_data_processed):
        path_data_processed_abs = path_parent + path_data_processed.replace('.', '')
        os.makedirs(path_data_processed_abs)

    if not os.path.isdir(path_location):
        path_location_abs = path_parent + path_location.replace('.', '')
        os.makedirs(path_location_abs)
        
    csv_path_and_name = path_location + '/' + date + '00.csv'    
    df.write.csv(csv_path_and_name)
    
    return None


def save_df_indiv(df, id_indiv):
    '''save the df to the folder for the current individual
         within 'data_indiv' folder, which is on the same level of the 'data' folder'
         
    Parameters
    ----------
    df : spark df containing the trajectory data for all individuals that have data for at least 30 days
         
    '''

    # create a new folder if it does not exist 
    path_parent = str(Path(os.getcwd()).parent)  # get parent path of src
    path_data_indiv =  '../data_of_indiv'
    
    if not os.path.isdir(path_data_indiv):
        path_data_indiv_abs = path_parent + path_data_indiv.replace('.', '')
        os.makedirs(path_data_indiv_abs)
        
    csv_path_and_name = path_data_indiv + '/' + '.csv'    
    df.write.csv(csv_path_and_name)
    
    return None


def retrieve_data_indiv(id_indiv, i, days_need_min=30, is_save=False):
     
    df_of_indiv = df.filter(df.id_str == id_indiv)
    
    # only keep the df for individual with data for every single day in at least 30 days
    # retrieve time data and convert to datetime
    time_timestamp  = [row.time for row in df_of_indiv.collect()]
    # convert unix timestamp to string time
    time_str= list( map(lambda num: datetime.fromtimestamp(num).strftime('%Y-%m-%d'),
                        time_timestamp) )
    
    date_uniq = list(set(time_str))  # unique dates
    # is_indiv_save = 0
    if len(date_uniq) >= days_need_min:
        if is_save:
            save_df_indiv(df=df_of_indiv, id_indiv = id_indiv)               
            
    return df_of_indiv


def create_empty_spark_df():
    # create empty pyspark df for stop points
    field_temp = [StructField("id_indiv", StringType(), True),
                  StructField("label", IntegerType(), True),
                  StructField("start", IntegerType(), True),
                  StructField("end", IntegerType(), True),
                  StructField("latitude", FloatType(), True),
                  StructField("longitude", FloatType(), True)]
    schema_temp = StructType(field_temp)
    df = sqlContext.createDataFrame(sc.sparkContext.emptyRDD(), schema_temp)
    
    return df 


def infer_indiv_stoppoint(df_of_indiv, id_indiv):
    '''
    infer the stop points of each individual given their trajectory data

    Parameters
    ----------
    df_of_indiv : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
       
    # sort by time to be used in infostop model
    df_of_indiv = df_of_indiv.sort(df_of_indiv.time.asc())
    
    # id_indiv = df_of_indiv.select(col('id_str')).first()[0]
    
    # convert to array
    traj_all =  np.array(df_of_indiv.select('latitude','longitude','time').collect())
    time_all = traj_all[:, 2]
    coord_all = traj_all[:, :2]
    
    # why the some values in the time array are the same? Precision is not high enough?
    time_all, uniq_indices = np.unique(time_all, return_index=True)
    traj_all =  traj_all[uniq_indices, :]
    coord_all = coord_all[uniq_indices, :]
        
    # define model
    # max distance between time-consecutive points to label them as stationary, and max distance between stationary points to form an edge.
    r1 , r2 = 30, 30 
    min_staying_time, max_time_between = 600, 86400  # in seconds
    model_infostop = infostop.Infostop(r1 =r1, r2 =r2,
                                       label_singleton=False,
                                       min_staying_time = min_staying_time,
                                       max_time_between = max_time_between,
                                       min_size = 2)

    # infer stops
    try:
        # labels are for stops: transition -1; positive integer indicates stop id, such as 1, 2, 3,
        labels = model_infostop.fit_predict(traj_all)
        is_stop_found = True
    except:
        print("\n ===== Oops! Failed to find stop point for individual: {} ===== ".format(id_indiv))
        is_stop_found = False       

    # get the coordinates of the stop points
    if is_stop_found and ( np.max(labels)>=2 ):
        # get stop id, t_start, t_end
        traj_at_stop = infostop.postprocess.compute_intervals(labels, time_all)

        # get coordinates for stop points only;
        # those for transition stops will be filtered out
        traj_at_stop = np.array(traj_at_stop)
        
        time_at_stop_start = traj_at_stop[:, 1]  #[:, None]
        idx_at_stop = np.where( np.in1d(time_all, time_at_stop_start) )[0]  #.any(axis=-1)
        coord_at_stop = coord_all[idx_at_stop, :]

        # store the data in a pyspark df. Create pandas df then convert to spark df        
        df_at_stop = pd.DataFrame(columns = ['label',
                                             'start','end',
                                             'latitude','longitude',
                                             'id_indiv'])                
        
        col_names_subset = ['label', 'start', 'end']
        for i in range(len(col_names_subset)):
            df_at_stop[col_names_subset[i]] =  traj_at_stop[:, i]   
        df_at_stop['latitude'], df_at_stop['longitude'] =  coord_at_stop[:, 0], coord_at_stop[:, 1]
        df_at_stop['id_indiv'] = id_indiv
        
        # convert to psypark df
        df_at_stop = sc.createDataFrame(df_at_stop)
        # change double to int
        schema_new = [IntegerType(), IntegerType(), IntegerType()]
        for i in range(len(col_names_subset)):
            df_at_stop = df_at_stop.withColumn(col_names_subset[i], df_at_stop[col_names_subset[i]].cast(schema_new[i]))
    
    else:
        if is_stop_found:
            print('\n   ===== The # of stops for this individual is <=1 =====')
            
        # create empty pyspark df
        df_at_stop = create_empty_spark_df()
 
    return df_at_stop


# list of df that is generated by applying nested functions
def loop_over_indiv(id_indiv, i):
    
    interv_print = 100
    if (i>=1) and (i%interv_print==0):
        print("\n ===== The index of individual in the loop is: {} =====".format(i))
    
    df_of_indiv = retrieve_data_indiv(id_indiv, i)
    
    df_at_stop = infer_indiv_stoppoint(df_of_indiv, id_indiv)
      
    return df_at_stop


def process_traj_indiv(df, days_need_min=30, is_save=False):
    '''
    Get the coordinates and time for each individual from the df that includes trajectory data for some time, e.g. 2 months.

    Parameters
    ----------
    df : spark df containing the trajectory data for all individuals for some time

    Returns
    -------
    Save the df for each individual that has more than 30 days' data; columns include: time, latitude, longitude
    is_indiv_save: a list of binary value indicating where the i-th individual data is saved. 

    '''
    
    # TODO: finding unique IDs is very slow. Need to improve the speed.
     
    id_uniq = [x.id_str for x in df.select('id_str').distinct().collect()]   
              
    print('\n ===== The total number of individuals is: {} ====='.format(len(id_uniq)))
    
    # stoppint_dfs_list = list(map(loop_over_indiv,
    #                              id_uniq, list( range(len(id_uniq)) )))
 
    # try 500 individuals first
    n_indiv_temp = 500     
    stoppint_dfs_list = list(map(loop_over_indiv,
                                 id_uniq[:n_indiv_temp],
                                 list( range(len(id_uniq[:n_indiv_temp])) )))
 
    for i in range(n_indiv_temp):
        stoppint_dfs_list[i].show()
    
    print('\n ===== The total number of individuals is: {} ====='.format(len(id_uniq)))
    
    # merge the returned dfs
    df_stoppint_merged = reduce(lambda x, y: append_dfs(x, y),
                                stoppint_dfs_list)
    
    return df_stoppint_merged


# def run_data_process()    
def main(is_save=False):
    
    # set the absolute path when run within python IDE
    # os.chdir("C:/Users/Administrator/OneDrive/GitHub/Scale_Mobility/src")
    # os.chdir("C:/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src")
    # os.chdir("/mnt/c/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src")   
    
    # TODO: move the setup of pyspark here within the main function after finalizing the previous functions
     
    path_data = '../data'
    location = 'Albany'
    date = '20200207'
    
    # TODO: change the functions that use days_need_min
    days_need_min = 30

    # load data, change column data type, and select columns for time and coordinates    
    df = load_data(path_data, location, date)

    df = rename_col(df)
    df = select_col(df)
    df = remove_error_entry(df)
    
    df_stoppint_merged = process_traj_indiv(df, days_need_min)
    
    df_stoppint_merged.show()
    
    # save trajectory data if needed    
    if is_save:
        save_df_traj(df, location, date) 
               
    return None


def get_exe_time():
    
    import time
    
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    
    time_diff = end_time-start_time
    
    print( "\n ===== Time for loading and selecting data: {} seconds =====".format( round(time_diff,3) ) )

    return None


def get_code_profile():
    
    import cProfile, pstats
    
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    
    print("\n\n\n")
    stats.print_stats()


##################
if __name__ == "__main__":
     
    main()

 






###################    
    # pyspark: 5963598 entires; 5.09 at the first time but much faster from the second time onward
    # pandas: 5963597; About 15 seconds when using pandas to load the data
    # dask: About 0.33 seconds