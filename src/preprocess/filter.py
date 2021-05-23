# -*- coding: utf-8 -*-
"""
@author: Jinzhu Yu
"""

#import pandas as pd
#import dask
#import dask.dataframe as dd
import numpy as np
from datetime import datetime
from functools import reduce
import os
from pathlib import Path

# import and set up pyspark configuration
import findspark
findspark.init()
# from pyspark.sql import SparkSession
from pyspark.sql import*
from pyspark.sql.types import*
from pyspark.sql.functions import col, length, lit
# #set an application name 
# #start spark cluster; if already started then get it else create it 
sc = SparkSession.builder.appName('data_preprocess').master("local[*]").getOrCreate()
#initialize SQLContext from spark cluster 
sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)



###################
# class TrajData:
    
# traj_data = TrajData()
# traj_data.main()
    
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
        df = df.filter((df.latitude<=lat_max) and (df.latitude>=lat_min) and
                       (df.longitude<=lon_max) and (df.longitude>=lon_min))
    
        
        df = df.filter(length(col('id_str')) > id_str_len_min)
    else:
        df = df[(df.latitude<=lat_max) and (df.latitude>=lat_min) and
                (df.longitude<=lon_max) and (df.longitude>=lon_min)]
        
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


def process_traj_indiv(df, days_need_min=1, package_for_df='spark', is_save=False):
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
            
    def retrive_data_indiv(id_indiv, i):
        
        if ( (i>=1) and (i%100==0) ):
            print('\n ===== retrieving the data for {}-th individual among {} ====='. format(i, len(id_uniq)))
     
        df_of_indiv = df.filter(df.id_str == id_indiv)  #.collect()
        
        # only keep the df for individual with data for every single day in at least 30 days
        # retrieve time data and convert to datetime
        time_timestamp  = [row.time for row in df_of_indiv.collect()]
        # convert unix timestamp to string time
        time_str= list( map(lambda num: datetime.fromtimestamp(num).strftime('%Y-%m-%d'),
                            time_timestamp) )
        
        date_uniq = list(set(time_str))  # unique dates
        if len(date_uniq) >= days_need_min:
            if is_save:
                save_df_indiv(df=df_of_indiv, id_indiv = id_indiv)
                
                is_indiv_save = 1
            else:
                is_indiv_save = 0
                
        return is_indiv_save, df_of_indiv

    def create_empty_spark_df():
        # create empty pyspark df for stop points
        field_temp = [StructField("id_indiv", StringType(), True),
                      StructField("label", StringType(), True),
                      StructField("start", IntegerType(), True),
                      StructField("end", IntegerType(), True),
                      StructField("latitude", FloatType(), True),
                      StructField("longitude", FloatType(), True)]
        schema_temp = StructType(field_temp)
        df_stoppint = sqlContext.createDataFrame(sc.sparkContext.emptyRDD(), schema_temp)
        
        return df_stoppint
    
    def infer_indiv_stoppoint(df = df_of_indiv):
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
        # dependent function from the package infostop
        # def compute_intervals(coords, coord_labels, max_time_between=86400, distance_metric="haversine"):
        # """Compute stop and moves intervals from the list of labels.
        
        # Parameters
        # ----------
        #     coords : np.array (shape=(N, 2) or shape=(N,3))
        #     coord_labels: 1d np.array of integers
    
        # Returns
        # -------
        #     intervals : array-like (shape=(N_intervals, 5))
        #         Columns are "label", "start_time", "end_time", "latitude", "longitude"
        
        # """  
        
        id_indiv = df.loc[0, 'id_str']
        
        # if np.any(np.isnan(np.vstack(df_temp[['latitude', 'longitude', 'time']].values)))==False:
        traj_array =  np.array(df.select('latitude','longitude','time').collect())
        time_array = traj_array[:, 2]
        coord_array = traj_array[:, :2]
            
        r1 , r2 = 30, 30
        min_staying_time, max_time_between = 600, 86400
        model_infostop = infostop.Infostop(r1 =r1, r2 =r2,
                                           label_singleton=False,
                                           min_staying_time = min_staying_time,
                                           max_time_between = max_time_between,
                                           min_size = 2)

        try:
            # what are labels: transition -1; if stops, then the stop id, such as 1, 2, 3,
            labels = model_infostop.fit_predict(traj_array)
            is_stop_found = True
        except:
            is_stop_found = False       
    
        # remove points with labels indicating transition stops
        # create empty pyspark df
        df_stoppint = create_empty_spark_df()

        if is_stop_found and ( np.max(labels)>1 ):
            # create empty df with columns names
            trajectory = infostop.postprocess.compute_intervals(labels, time_array)    

                       
            # keep the time and coordinates for stop points only among time and coordinates;
            # those for transition stops will be filtered out
            time_from_trajectory = trajectory[:, 0][:, None]
            idx_stoppoint = (time_array==time_from_trajectory).any(axis=-1)
            time_keep = time_array[idx_stoppoint]
            coord_keep = coord_array[idx_stoppoint, :]
            traj_data_stoppoint = trajectory[idx_stoppoint, :]

            # fill the respective columns
            col_names_temp = ['label', 'start', 'end']
            for i in range(len(col_names_temp)):
                df_stoppint.withColumn(col_names_temp[i], traj_data_stoppoint[:, i])
        
            df_stoppint.withColumn("latitude", coord_keep[:, 0])
            df_stoppint.withColumn("longitude", coord_keep[:, 1])
            
            df_stoppint.withColumn("id_indiv", lit(id_indiv)) # fill with the same value
            # trajectory[i] = ['label', 'start', 'end']            
     
        return df_stoppint


    n_indiv_temp = 5
    
    # list of df that is generated by applying nested functions
    infer_indiv_stoppoint(df = df_of_indiv); is_indiv_save, df_of_indiv = retrive_data_indiv(id_indiv, i)
    
    is_indiv_save = list( map( loop_over_indiv, id_uniq[n_indiv_temp], list( range(len(id_uniq[:n_indiv_temp])) ) ) )
    # is_indiv_save = list( map( loop_over_indiv, id_uniq, list( range(len(id_uniq)) ) ) )  
    # list of dfs 
    n_divide = 10
    
    df_stoppint_merged = reduce(DataFrame.union, list_of_dfs)  
    
    return is_indiv_save


# def run_data_process()    
def main(is_save=False, package_for_df='spark'):
    
    # set the absolute path when run within python IDE
    # os.chdir("C:/Users/Administrator/OneDrive/GitHub/Scale_Mobility/src")
    # os.chdir("C:/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src")
    # os.chdir("/home/jayz/Documents/GitHub/Scale_Mobility/src")
    
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
    
    # try on 5 individuals for now. See the value "n_indiv_temp" in the "retrieve_data_indiv" function
    retrieve_data_indiv(df, days_need_min, package_for_df)
        
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