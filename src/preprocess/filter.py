# -*- coding: utf-8 -*-
"""
@author: Jinzhu Yu
"""

import pandas as pd
import dask
import dask.dataframe as dd
from datetime import datetime
import os
from pathlib import Path

# import and set up pyspark configuration
import findspark
findspark.init('C:/Spark/spark-3.0.0-bin-hadoop2.7')
# from pyspark import SparkConf, SparkContext
from pyspark.sql import*
from pyspark.sql.types import*
from pyspark.sql.functions import col, length
# #set an application name 
# #start spark cluster; if already started then get it else create it 
sc = SparkSession.builder.appName('data_preprocess').master("local[*]").getOrCreate()
#initialize SQLContext from spark cluster 
sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)




###################
def load_data(path_data='../data', location='Albany', date='20200207', package_for_df='spark'):
    '''import the raw data.
    
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
        package - the package used for loading the csv.gz file, including spark and dd (dask)
    '''
        
    #load raw data
    os.chdir(path_data)
    # path_datafile = os.path.join(os.getcwd(), '{}\\{}00.csv.gz'.format(location, date))
    path_datafile = os.path.join(os.getcwd(), '{}\\*.csv.gz'.format(location)) #load all csv.gz files at once
    
    # select the package used to load data, spark, pd (pandas), or dd (dask)
    if package_for_df == 'spark':
        df = sqlContext.read.csv(path_datafile, header=False)
    else:
        df = dd.read_csv(path_datafile, compression='gzip', error_bad_lines=False)

    return df 


def rename_col(df, package_for_df='spark'):
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
    if package_for_df == 'spark':
        for i in range(len(col_names_old)):
            df = df.withColumnRenamed(col_names_old[i], col_names_new[i])
    else:
        df = df.rename(columns=dict(zip(col_names_old, col_names_new)))
        
    # change column type
    if package_for_df == 'spark':
        schema_new = [IntegerType(), StringType(), IntegerType(), FloatType(), FloatType(),
                      FloatType(), IntegerType(), StringType(), StringType(), IntegerType()]
        for i in range(len(col_names_new)):
            df = df.withColumn(col_names_new[i], df[col_names_new[i]].cast(schema_new[i]))
    else:
        schema_new = [int, str, int, float, float, int, int, str, str, int]
        for i in range(len(col_names_new)):
            col = col_names_new
            df[col] == df[col].astype(schema_new[i])
              
    return df 
   

def select_col(df, package_for_df='spark'):
    '''select columns: id_str, time, latidude, and longitude, accuracy
    '''
    
    col_select = ['time', 'latitude', 'longitude', 'id_str']
    
    if package_for_df == 'spark':
        df_select = df.select(*col_select)
    else:
        df_select = df[col_select]
    
    return df_select

    
def remove_error_entry(df, package_for_df='spark'):
    
    '''remove entries with erreneous value of coordinates and 'id_str'.
       There can be errors in the latitude or longitude. E.g., the min of latitude is -14400
    '''
      
    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180
    id_str_len_min = 15
    
    if package_for_df == 'spark':
        df = df.filter((df.latitude<=lat_max) & (df.latitude>=lat_min) &
                       (df.longitude<=lon_max) & (df.longitude>=lon_min))
    
        
        df = df.filter(length(col('id_str')) > id_str_len_min)
    else:
        df = df[(df.latitude<=lat_max) & (df.latitude>=lat_min) &
                (df.longitude<=lon_max) & (df.longitude>=lon_min)]
        
        df = df[df.id_str.str.len() > id_str_len_min]
    
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
         within 'data_of_indiv' folder, which is on the same level of the 'data' folder'
         
    Parameters
    ----------
    df : spark df containing the trajectory data for all individuals for at least 30 days
         
    '''

    # create a new folder if it does not exist 
    path_parent = str(Path(os.getcwd()).parent)  # get parent path of src
    path_data_of_indiv =  '../data_of_indiv'
    
    if not os.path.isdir(path_data_of_indiv):
        path_data_of_indiv_abs = path_parent + path_data_of_indiv.replace('.', '')
        os.makedirs(path_data_of_indiv_abs)
        
    csv_path_and_name = path_data_of_indiv + '/' + '.csv'    
    df.write.csv(csv_path_and_name)
    
    return None


def retrieve_data_of_indiv(df, days_need_min=1, package_for_df='spark', is_save=False):
    '''
    Get the coordinates and time for each individual from the df that include trajectory data for some time, e.g. 2 months.

    Parameters
    ----------
    df : spark df containing the trajectory data for all individuals for some time

    Returns
    -------
    Save the df for each individual that has more than 30 days' data; columns include: time, latitude, longitude
    is_indiv_save: a list of binary value indicating where the i-th individual data is saved. 

    '''
    
    # TODO: finding unique IDs is very slow. Need to improve the speed.
    # id_unique = [i for i in df.select('id_str').distinct().collect()]
    

    # print('Using spark')
    id_uniq = [x.id_str for x in df.select('id_str').distinct().collect()]
    # else:
    #     print('Using dask')
    #     id_uniq = df['id_str'].unique().compute()

    def loop_over_indiv(id_indiv, i):
        
        if (i >= 1 & i % 100==0):
            print('\n ===== retrieving the data for {}-th individual among {} ====='. format(i, len(id_uniq)))
     
        df_of_indiv = df.filter(df.id_str == id_indiv)  #.collect()
        
        # only keep the df for individual with data for every single day in at least 30 days
        # retrieve time data and convert to datetime
        time_timestamp  = [row.time for row in df_of_indiv.collect()]
        # convert unix timestamp to string time
        time_str= list( map(lambda num: datetime.fromtimestamp(num).strftime('%Y-%m-%d'),
                            time_timestamp) )
        
        date_uniq = list(set(time_str))  # unique dates
        if (len(date_uniq) >= days_need_min) & is_save:
            save_df_indiv(df=df_of_indiv, id_indiv = id_indiv)
            
            return 1
        
        else:
            
            return 0
        

    is_indiv_save = list( map( loop_over_indiv, id_uniq[10], list( range(len(id_uniq[:10])) ) ) )
    
    return is_indiv_save

    
def main(is_save=False, package_for_df='spark'):
    
    # set the absolute path when run within python IDE
    # os.chdir("C:/Users/Administrator/OneDrive/GitHub/Scale_Mobility/src")
    # os.chdir("C:/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src")
    
    path_data = '../data'
    location = 'Albany'
    date = '20200207'

    # load data, change column data type, and select columns for time and coordinates
    package_for_df = 'spark'    
    df = load_data(path_data, location, date, package_for_df)

    df = rename_col(df)
    df = select_col(df)
    df = remove_error_entry(df)
    
    # save trajectory data if needed    
    if is_save:
        save_df_traj(df, location, date) 
        
    # retrieve the trajectory data of each individual
    days_need_min = 30
    retrieve_data_of_indiv(df, days_need_min, package_for_df)
        
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


###################
if __name__ == "__main__":
     
    main()

 
    
    # pyspark: 5963598 entires; 5.09 at the first time but much faster from the second time onward
    # pandas: 5963597; About 15 seconds when using pandas to load the data
    # dask: About 0.33 seconds