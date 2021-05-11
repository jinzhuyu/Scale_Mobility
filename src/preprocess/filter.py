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
#set an application name 
conf = SparkConf().setMaster("local").setAppName("find_stops")
#start spark cluster; if already started then get it else create it 
sc = SparkContext.getOrCreate(conf=conf)
#initialize SQLContext from spark cluster 
sqlContext = SQLContext(sc)

#dataframe 
#set header property true for the actual header columns
path = "../data/Albany/2020031600.csv"
df = sqlContext.read.csv(path, header=False)

# input=sc.textFile("../data/Albany/2020031600.csv");
# input_filtered=input.filter(lambda row : return (row.split(",")[3]=="comedy" and row.split(",")[5]=="2018")  )
