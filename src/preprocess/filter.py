# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:19:17 2021

@author: Jinzhu Yu
"""

import collections
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

#load raw data
path = ("C:/Users/Administrator/OneDrive/GitHub/Scale_Mobility/data/Albany/2020031600.csv.gz")
# path = "../data/Albany/2020031600.csv.gz"
data_raw = sqlContext.read.csv(path, header=False)
# name the columns
colomn_names = ['time_original','id_str','device_type','latitude','longitude','accuracy','timezone','class','transform','time']
data_raw = data_raw.toDF(*colomn_names).collect()
# check data type of each column
data_raw.printSchema()
# change data type
new_schema = [DateType, StringType, IntegerType, FloatType, FloatType, FloatType, IntegerType, StringType, StringType, DateType]
# https://sparkbyexamples.com/spark/spark-sql-date-and-time-functions/

for i in range(len(colomn_names)):
    data_raw = data_raw.withColumn(col, data_raw[colomn_names[i]].cast(new_schema[i]))
# Note: Datetime type
# TimestampType: Represents values comprising values of fields year, month, day, hour, minute, and second, with the session local time-zone. The timestamp value represents an absolute point in time.
# DateType: Represents values comprising values of fields year, month and day, without a time-zone.
# df = sqlContext.createDataFrame(data_raw, schema=schema)
# filter the dataframe
df_select = data_raw.filter( (data_raw.) & (data_raw. >) ).collect()