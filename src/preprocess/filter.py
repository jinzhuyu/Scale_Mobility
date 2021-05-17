# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:19:17 2021

@author: Jinzhu Yu
"""

# import collections
import os

# import and set up pyspark configuration
import findspark
findspark.init('C:/Spark/spark-3.0.0-bin-hadoop2.7')
from pyspark import SparkConf, SparkContext
#import all the libraries of pyspark.sql
from pyspark.sql import*
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, TimestampType, DateType


#set an application name 
conf = SparkConf().setMaster("local").setAppName("find_stops")
#start spark cluster; if already started then get it else create it 
sc = SparkContext.getOrCreate(conf=conf)
#initialize SQLContext from spark cluster 
sqlContext = SQLContext(sc)



def main():

    os.chdir("C:/Users/Administrator/OneDrive/GitHub/Scale_Mobility/src")
    
    #load raw data
    os.chdir('../data')
    path_datafile = os.path.join(os.getcwd(), 'Albany\\2020031600.csv.gz')
    # path_datafile = os.path.join(os.getcwd(), 'Albany\\*.csv.gz') #load all csv.gz files at once
    df = sqlContext.read.csv(path_datafile, header=False)
    
    
    # remove erroneous entries
    
    # # def remove_error_entries(df)
    #     df.columns=['time_original','id_str','device_type','latitude','longitude','accuracy','timezone','class','transform','time']
    #     temp_index=[];time_list=[];lat_list=[];lon_list=[]
    #     # temp_index, time_list, lat_list, lon_list = [], [], [], []
    #     for index,str1,str2,str3 in zip([i for i in range(len(df))],df['time'],df['latitude'],df['longitude']):
    #         try:
    #             time_list.append(int(float(str1)))
    #         except:
    #             time_list.append(0)
    #             temp_index.append(index)
    #         try:
    #             lat_list.append(float(str(str2)))
    #         except:
    #             lat_list.append(0)
    #             temp_index.append(index)
    #             #print('except_str2',str2)
    #         try:
    #             lon_list.append(float(str(str3)))
    #         except:
    #             lon_list.append(0)
    #             temp_index.append(index)
    #             #print('except_str3', str3)
                
    #     df['time']=time_list
    #     df['latitude']=lat_list
    #     df['longitude']=[ i if i>-180 else -179 for i in lon_list ]
    #     df['longitude']=[ i if i<180 else 179 for i in df['longitude'] ]
    #     df=df.drop(temp_index)
    #     df = df.dropna()
        
    #     if not os.path.isdir(output_path):
    #         path_parent = str(Path(os.getcwd()).parent)
    #         path_output_gen = path_parent + output_path.replace('.', '')
    #         os.makedirs(path_output_gen)
            
    #     df.to_csv(output_path+file[0:10]+'.csv')
    
    
    
    # rename the columns
    col_names_old = df.columns
    col_names_new = ['time_original','id_str','device_type','latitude','longitude','accuracy','timezone','class','transform','time']
    for i in range(len(col_names_old)):
        df = df.withColumnRenamed(col_names_old[i], col_names_new[i])
    
    # change column type
        # # check out data type of each column
        # df.printSchema()  # all string, column names are '_c0' to '_c9'
        # # show top 5 rows
        # df.show(5)
    # change data type
    schema_new = [IntegerType(), StringType(), IntegerType(), FloatType(), FloatType(),
                  FloatType(), IntegerType(), StringType(), StringType(), IntegerType()]
    # https://sparkbyexamples.com/spark/spark-sql-date-and-time-functions/
    for i in range(len(col_names_new)):
        df = df.withColumn(col_names_new[i], df[col_names_new[i]].cast(schema_new[i]))
    
    
    # select columns: time, latidude, and longitude
    col_select = ['time', 'latitude', 'longitude']
    df_select = df.select(*col_select)
    
    # filter the dataframeto get entries with correct latitude and longitude
        ## There can be errors in the latitude or longitude. For example, the min of latitude is -14400
        # df.agg({"latitude": "min"}).collect()[0]
        # df.agg({"longitude": "min"}).collect()[0]   
    df_select = df_select.filter((df_select.latitude<=90) & (df_select.latitude>=-90) &
                                 (df_select.longitude<=180) & (df_select.longitude>=-180))
    

    ## save the selected df if needed    
    # if not os.path.isdir(output_path):
    #     path_parent = str(Path(os.getcwd()).parent)
    #     path_output_gen = path_parent + output_path.replace('.', '')
    #     os.makedirs(path_output_gen)
        
    # df.to_csv(output_path+file[0:10]+'.csv')

def get_exe_time():
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    time_diff = round(end_time-start_time, 3)
    print("--- Time elapsed: {} seconds ---".format(time_diff))

get_exe_time()  # 5.088 at the first time but much faster from the second time onward