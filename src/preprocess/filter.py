# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:19:17 2021

@author: Jinzhu Yu
"""
import pandas as pd
import dask
import dask.dataframe as dd
import os
from pathlib import Path

# import and set up pyspark configuration
import findspark
findspark.init('C:/Spark/spark-3.0.0-bin-hadoop2.7')
# from pyspark import SparkConf, SparkContext
#import all the libraries of pyspark.sql
from pyspark.sql import*
from pyspark.sql.types import*
# #set an application name 
# conf = SparkConf().setMaster("local").setAppName("data_preprocess")
# #start spark cluster; if already started then get it else create it 
# sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder.appName('data_preprocess').getOrCreate()
#initialize SQLContext from spark cluster 
sqlContext = SQLContext(spark)


''' TODO: Loading the data seems to take most of the time.
          Check if using dask to load csv is faster than using pyspark
          ref.: https://medium.com/analytics-vidhya/optimized-ways-to-read-large-csvs-in-python-ab2b36a7914e
          
          dask_df.values.compute() is equivalent to pandas_df.values(), both convert df to a numpy array
              to be used in the infostop functions
'''


def load_data(path_data='../data', location='Albany', date='20200316'):
    '''import the raw data, rename columns, and select the columns for time and coordinates.
    
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
    '''
    
    #load raw data
    os.chdir(path_data)
    path_datafile = os.path.join(os.getcwd(), '{}\\{}00.csv.gz'.format(location, date))
    # path_datafile = os.path.join(os.getcwd(), 'Albany\\*.csv.gz') #load all csv.gz files at once
    df = sqlContext.read.csv(path_datafile, header=False)

    return df    


def load_data_pandas(path_data='../data', location='Albany', date='20200316'):
    '''import the raw data, rename columns, and select the columns for time and coordinates.
    
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
    '''
    
    #load raw data
    os.chdir(path_data)
    path_datafile = os.path.join(os.getcwd(), '{}\\{}00.csv.gz'.format(location, date))
    # path_datafile = os.path.join(os.getcwd(), 'Albany\\*.csv.gz') #load all csv.gz files at once
    df = pd.read_csv(path_datafile, compression='gzip', error_bad_lines=False)

    return df 

def load_data_dask(path_data='../data', location='Albany', date='20200316'):
    '''import the raw data, rename columns, and select the columns for time and coordinates.
    
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
    '''
    
    #load raw data
    os.chdir(path_data)
    path_datafile = os.path.join(os.getcwd(), '{}\\{}00.csv.gz'.format(location, date))
    # path_datafile = os.path.join(os.getcwd(), 'Albany\\*.csv.gz') #load all csv.gz files at once
    df = dd.read_csv(path_datafile, compression='gzip', error_bad_lines=False)

    return df 


def rename_and_select_col(df):
    '''rename the columns and select columns: time, latidude, and longitude
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
    
    #select columns: time, latidude, and longitude
    col_select = ['time', 'latitude', 'longitude']
    df_select = df.select(*col_select)
    
    return df_select

    
def remove_error_coord(df):
    
    '''remove entries with erreneous value of coordinates
       ## filter the dataframeto get entries with correct latitude and longitude
        ## There can be errors in the latitude or longitude. E.g., the min of latitude is -14400
        # df.agg({"latitude": "min"}).collect()[0]
    '''
        
    df = df.filter((df.latitude<=90) & (df.latitude>=-90) &
                   (df.longitude<=180) & (df.longitude>=-180))
    
    return df
    

def save_df(df, location='Albany', date='20200316'):
    '''save the processed df to the folder for the current location
         within 'data_processed' folder, which is on the same level of the 'data' folder'
    '''

    # create a new folder if the 'data_processed' folder or the folder for the
        # current location doesn't exist 
    path_parent = str(Path(os.getcwd()).parent)  # parent path of src
    path_data_processed =  '../data_processed'
    path_location =  '../data_processed/{}'.format(location)
    
    if not os.path.isdir(path_data_processed):
        path_data_processed_abs = path_parent + path_data_processed.replace('.', '')
        os.makedirs(path_data_processed_abs)

    if not os.path.isdir(path_location):
        path_location_abs = path_parent + path_location.replace('.', '')
        os.makedirs(path_location_abs)
        
    path_csv = path_location + '/' + date + '00.csv'    
    df.write.csv(path_csv)
    
    return None


# df2 = spark.createDataFrame([("a",), ("b",)], ["column name"])
# df2.show()
# from pyspark.sql.functions import col
# df2.filter(col("column name") == 'b').show()

    
def main(is_save=False):
    
    # set the absolute path when run within python IDE
    # os.chdir("C:/Users/Administrator/OneDrive/GitHub/Scale_Mobility/src")
    path_data = '../data'
    location = 'Albany'
    date = '20200316'

    # load data, change column data type, and select columns for time and coordinates    
    df = load_data(path_data, location)

    # df = rename_and_select_col(df)
    # df = remove_error_coord(df)

    # save data if needed    
    if is_save:
        save_df(df, location, date)
        
    return None


def get_exe_time():
    
    import time
    
    start_time = time.time()
    main()
    end_time = time.time()
    time_diff = end_time-start_time
    
    print("--- Time for loading and selecting data: {} seconds ---".format( round(time_diff, 3) ) )

    return None


if __name__ == "__main__":
    main()
    get_exe_time() 
    
    # pyspark: 5963598 entires; 5.09 at the first time but much faster from the second time onward
    # pandas: 5963597; About 15 seconds when using pandas to load the data
    # dask: About 0.33 seconds
    
    
    
    