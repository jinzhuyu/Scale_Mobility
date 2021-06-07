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

# import and set up pyspark configuration
import findspark

findspark.init()

from pyspark.sql import SparkSession
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, length  # , lit

# #set an application name
# #start spark cluster; if already started then get it else create it
sc = SparkSession.builder.appName('data_preprocess').master("local[*]").getOrCreate()
# initialize SQLContext from spark cluster
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

    # load raw data
    os.chdir(path_data)
    #path_datafile = os.path.join(os.getcwd(), '{}/{}00.csv.gz'.format(location, date))
    path_datafile = os.path.join(os.getcwd(), '{}/*.csv.gz'.format(location))  # load all csv.gz files at once

    # select the package used to load data, spark, pd (pandas), or dd (dask)
    if package_for_df == 'spark':
        df = sqlContext.read.csv(path_datafile, header=False)
    else:
        df = dd.read_csv(path_datafile, compression='gzip', error_bad_lines=False)

    # change back to the default trajectory
    #os.chdir('../src')
    os.chdir('/Volumes/SeagateDrive/Trajectory Data/src')
    return df


# df_of_indiv = sqlContext.read.csv('/home/jayz/Documents/GitHub/Scale_Mobility/data/data_indiv/indiv_1.csv', header=True)

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
    col_names_new = ['time_original', 'id_str', 'device_type', 'latitude', 'longitude', 'accuracy', 'timezone', 'class',
                     'transform', 'time']
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


def select_col(df):
    '''select columns: id_str, time, latidude, and longitude, accuracy
    '''

    col_select = ['time', 'latitude', 'longitude', 'id_str']
    df_select = df.select(*col_select)

    return df_select


def remove_error_entry(df, package_for_df='spark'):
    '''remove entries with erreneous value of coordinates and 'id_str'.
       There can be errors in the latitude or longitude. E.g., the min of latitude is -14400
    '''

    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180
    id_str_len_min = 15

    if package_for_df == 'spark':
        df = df.filter((df.latitude <= lat_max) & (df.latitude >= lat_min) &
                       (df.longitude <= lon_max) & (df.longitude >= lon_min))

        df = df.filter(length(col('id_str')) > id_str_len_min)
    else:
        df = df[(df.latitude <= lat_max) & (df.latitude >= lat_min) &
                (df.longitude <= lon_max) & (df.longitude >= lon_min)]

        df = df[df.id_str.str.len() > id_str_len_min]

    return df


def save_df_traj(df, id_indiv):
    '''save the processed df to the folder for the current location
         within 'data_processed' folder, which is on the same level of the 'data' folder'
    '''

    # create a new folder if the 'data_processed' folder or the folder for the
    # current location doesn't exist
    path_parent = str(Path(os.getcwd()).parent)  # get parent path of src
    path_data_processed = '../data_processed'
    path_location = path_data_processed # path for the data for current location/place/city

    if not os.path.isdir(path_data_processed):
        path_data_processed_abs = path_parent + path_data_processed.replace('.', '')
        os.makedirs(path_data_processed_abs)

    if not os.path.isdir(path_location):
        path_location_abs = path_parent + path_location.replace('.', '')
        os.makedirs(path_location_abs)

    csv_path_and_name = path_location + '/' + id_indiv + '.csv'
    #df.coalesce(1).write.csv(csv_path_and_name)
    df.to_csv(csv_path_and_name)

    return None


def save_df_indiv(df, id_indiv):
    print(id_indiv)
    '''save the df to the folder for the current individual
         within 'data_indiv' folder, which is on the same level of the 'data' folder'

    Parameters
    ----------
    df : spark df containing the trajectory data for all individuals that have data for at least 30 days

    '''

    # create a new folder if it does not exist
    path_parent = str(Path(os.getcwd()).parent)  # get parent path of src
    path_data_indiv = '../data_of_indiv'
    if not os.path.isdir(path_data_indiv):
        path_data_indiv_abs = path_parent + path_data_indiv.replace('.', '')
        os.makedirs(path_data_indiv_abs)
    csv_path_and_name = path_data_indiv + '/' + id_indiv+'.csv'
    #df.write.csv(csv_path_and_name)
    df.to_csv(csv_path_and_name)

    return None

def _map_to_pandas(rdds):
    """ Needs to be here due to pickling issues """
    return [pd.DataFrame(list(rdds))]

def toPandas_optimisation(df, n_partitions=None):
    """
    Returns the contents of `df` as a local `pandas.DataFrame` in a speedy fashion. The DataFrame is
    repartitioned if `n_partitions` is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                pandas.DataFrame
    """
    if n_partitions is not None:
        df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand


def retrieve_data_indiv(id_indiv, df,i, days_need_min=30, is_save=True):
    days_need_requirement=False
    df_of_indiv = df.filter(df.id_str == id_indiv)
    traj_all = df_of_indiv.toPandas()
    # only keep the df for individual with data for every single day in at least 30 days
    # retrieve time data and convert to datetime

    time_timestamp = [row for row in traj_all['time'] if np.isnan(row)==False]# convert unix timestamp to string time
    time_str = list(map(lambda num: datetime.fromtimestamp(num).strftime('%Y-%m-%d'), time_timestamp))

    date_uniq = list(set(time_str))  # unique dates
    if len(date_uniq) >= days_need_min:
        if is_save:
            save_df_indiv(df=traj_all, id_indiv=id_indiv)
        days_need_requirement=True
    return df_of_indiv,traj_all,days_need_requirement


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


def infer_indiv_stoppoint(df_of_indiv, id_indiv,traj_all):
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
    traj_all = np.array(traj_all[['latitude', 'longitude', 'time']].values)
    time_all = traj_all[:, 2]
    coord_all = traj_all[:, :2]

    # why the some values in the time array are the same? Precision is not high enough?
    time_all, uniq_indices = np.unique(time_all, return_index=True)
    traj_all = traj_all[uniq_indices, :]
    coord_all = coord_all[uniq_indices, :]

    # define model
    # max distance between time-consecutive points to label them as stationary, and max distance between stationary points to form an edge.
    r1, r2 = 30, 30
    min_staying_time, max_time_between = 600, 86400  # in seconds
    model_infostop = infostop.Infostop(r1=r1, r2=r2,
                                       label_singleton=False,
                                       min_staying_time=min_staying_time,
                                       max_time_between=max_time_between,
                                       min_size=2)

    # infer stops
    try:
        # labels are for stops: transition -1; positive integer indicates stop id, such as 1, 2, 3,
        labels = model_infostop.fit_predict(traj_all)
        is_stop_found = True
    except:
        #print("\n ===== Oops! Failed to find stop point for individual: {} ===== ".format(id_indiv))
        is_stop_found = False

        # get the coordinates of the stop points
    if is_stop_found and (np.max(labels) >= 2):
        print('   ===== The stops for this individual is found =====')
        # get stop id, t_start, t_end
        traj_at_stop = infostop.postprocess.compute_intervals(labels, time_all)
        # get coordinates for stop points only;
        # those for transition stops will be filtered out
        traj_at_stop = np.array(traj_at_stop)

        time_at_stop_start = traj_at_stop[:, 1]  # [:, None]
        idx_at_stop = np.where(np.in1d(time_all, time_at_stop_start))[0]  # .any(axis=-1)
        coord_at_stop = coord_all[idx_at_stop, :]

        # store the data in a pyspark df. Create pandas df then convert to spark df
        df_at_stop = pd.DataFrame(columns=['label',
                                           'start', 'end',
                                           'latitude', 'longitude',
                                           'id_indiv'])

        col_names_subset = ['label', 'start', 'end']
        for i in range(len(col_names_subset)):
            df_at_stop[col_names_subset[i]] = traj_at_stop[:, i]
        df_at_stop['latitude'], df_at_stop['longitude'] = coord_at_stop[:, 0], coord_at_stop[:, 1]
        df_at_stop['id_indiv'] = id_indiv
        save_df_traj(df_at_stop, id_indiv)


    return is_stop_found


# list of df that is generated by applying nested functions
def loop_over_indiv(df,id_indiv,i):
    #interv_print = 100
    print(id_indiv,i)
    #if (i >= 1) and (i % interv_print == 0):
        #print("\n ===== The index of individual in the loop is: {} =====".format(i))

    df_of_indiv,traj_all,days_need_requirement = retrieve_data_indiv(id_indiv,df, i)
    if days_need_requirement==True:
        df_at_stop = infer_indiv_stoppoint(df_of_indiv, id_indiv,traj_all)
    else:
        df_at_stop=0



    return df_at_stop


def process_traj_indiv(df,id_uniq, days_need_min=30,package_for_df='spark', is_save=True):
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



    # stoppint_dfs_list = list(map(loop_over_indiv,
    #                              id_uniq, list( range(len(id_uniq)) )))

    # n_divide = 10
    # try with 1000 individuals first
    n_indiv_temp = 1000
    begin_indiv=16
    stoppint_dfs_list = list(map(loop_over_indiv,[df for i in  range(len(id_uniq[:n_indiv_temp]))], id_uniq[begin_indiv:n_indiv_temp+2],
                                 list(range(len(id_uniq[begin_indiv:n_indiv_temp+2])))))

    print('\n ===== ALL FINISHED. The total number of individuals is: {} ====='.format(len(id_uniq)))



def load_data_after_process(path_data='../data',  package_for_df='spark'):
    '''import the raw data.

    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
        package - the package used for loading the csv.gz file, including spark and dd (dask)
    '''

    # load raw data
    os.chdir(path_data)
    path_datafile = os.path.join(os.getcwd(), 'albany_all.csv')
    #path_datafile = os.path.join(os.getcwd(), '{}/*.csv.gz'.format(location))  # load all csv.gz files at once

    # select the package used to load data, spark, pd (pandas), or dd (dask)
    if package_for_df == 'spark':
        df = sqlContext.read.csv(path_datafile, header=False)
    else:
        df = dd.read_csv(path_datafile, compression='gzip', error_bad_lines=False)

    # change back to the default trajectory
    #os.chdir('../src')
    os.chdir('/Volumes/SeagateDrive/Trajectory Data/src')
    return df


# def run_data_process()
def main(is_save=False, package_for_df='spark'):
    # set the absolute path when run within python IDE
    # os.chdir("C:/Users/Administrator/OneDrive/GitHub/Scale_Mobility/src")
    # os.chdir("C:/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src")
    # os.chdir("/home/jayz/Documents/GitHub/Scale_Mobility/src")
    # os.chdir("/mnt/c/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src")

    # TODO:
    # move the setup of pyspark here within the main function after finalizing the previous functions

    path_data = '/Volumes/SeagateDrive/Trajectory Data'
    location = 'Albany'
    date = '20200101'

    # TODO: change the functions that use days_need_min
    days_need_min = 30

    # load data, change column data type, and select columns for time and coordinates
    package_for_df = 'spark'
    df = load_data(path_data, location, date, package_for_df)
    df = rename_col(df)
    df = select_col(df)
    df = remove_error_entry(df)
    print("ready with load_data")

    case ='pre_process' ###pre_process id_uniq
    case = 'after_process'  ###after_process with obtained id_uniq
    if case=='pre_process':
        df.select('id_str').distinct().coalesce(1).write.csv("/Volumes/SeagateDrive/Trajectory Data/Albany_backup/Albany_ids_temp.csv")
    if case=='after_process':
        print(case)
        id_uniq=pd.read_csv('/Volumes/SeagateDrive/Trajectory Data/Albany_backup/Albany_ids.csv',names=['id_str'])
        id_uniq=list(id_uniq['id_str'].values)
        print('\n ===== The total number of individuals is: {} ====='.format(len(id_uniq)))

    process_traj_indiv(df, id_uniq,days_need_min)


    return None


def get_exe_time():
    import time

    start_time = time.time()

    main()

    end_time = time.time()

    time_diff = end_time - start_time

    print("\n ===== Time for loading and selecting data: {} seconds =====".format(round(time_diff, 3)))

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
