#import all the libraries of pyspark.sql
from pyspark.sql import*
#import SparkContext and SparkConf
from pyspark import SparkContext, SparkConf

#set an application name 
conf = SparkConf().setMaster("local").setAppName("filter_data")
#start spark cluster 
#if already started then get it else start it 
sc = SparkContext.getOrCreate(conf=conf)
#initialize SQLContext from spark cluster 
sqlContext = SQLContext(sc)



#dataframe 
#set header property true for the actual header columns
path = "../data/Albany/2020031600.csv"
df = sqlContext.read.csv(path, header=False)

# input=sc.textFile("../data/Albany/2020031600.csv");
# input_filtered=input.filter(lambda row : return (row.split(",")[3]=="comedy" and row.split(",")[5]=="2018")  )
