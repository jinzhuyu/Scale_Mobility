import infostop
import datetime
import pandas as pd
import numpy as np
import sys,time
import collections
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import os
import pickle
from collections import defaultdict
os.chdir("/Users/lucinezhong/Documents/LuZHONGResearch/20210328Scale_Mobility/")

def pre_process(path,output_path,output_list):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:
        if file in output_list:
            df = pd.read_csv(path + "/" + file, compression='gzip',error_bad_lines=False)
            df.columns=['time_original','id_str','device_type','latitude','longitude','accuracy','timezone','class','transform','time']
            temp_index=[];time_list=[];lat_list=[];lon_list=[]
            for index,str1,str2,str3 in zip([i for i in range(len(df))],df['time'],df['latitude'],df['longitude']):
                try:
                    time_list.append(int(float(str1)))
                except:
                    time_list.append(0)
                    temp_index.append(index)
                try:
                    lat_list.append(float(str(str2)))
                except:
                    lat_list.append(0)
                    temp_index.append(index)
                    #print('except_str2',str2)
                try:
                    lon_list.append(float(str(str3)))
                except:
                    lon_list.append(0)
                    temp_index.append(index)
                    #print('except_str3', str3)
            df['time']=time_list
            df['latitude']=lat_list
            df['longitude']=[ i if i>-180 else -179 for i in lon_list ]
            df['longitude']=[ i if i<180 else 179 for i in df['longitude'] ]
            df=df.drop(temp_index)
            df = df.dropna()
            df.to_csv(output_path+file[0:10]+'.csv')

def merge(path,output_path):
    id_date=defaultdict()
    list_id = []
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    count=0
    for file in files[0:1]:
        print(file)
        df = pd.read_csv(path + "/" + file)
        for index, row in df.iterrows():
            if row['id_str'] not in id_date.keys():
                id_date[row['id_str']] = dict()
            if file not in id_date[row['id_str']].keys():
                id_date[row['id_str']][file] = []
            id_date[row['id_str']][file].append(index)
        list_id = list_id + list(pd.unique(df['id_str']))
        count+=1
    df_all_id = pd.DataFrame(columns=['id_str'])
    df_all_id['id_str'] = np.unique(list_id)
    df_all_id.to_csv(output_path + output_str + '_Albany_all_ids.csv')
    with open(output_path + output_str + '_Albany_all_ids_dict.pickle', 'wb') as handle:
        pickle.dump(id_date, handle)

def individual_data_process(datapath_1,datapath_2,output_path):
    r1=30; r2=30; min_staying_time=600; max_time_between=86400;
    r1=30; r2=30; min_staying_time=600; max_time_between=86400;

    files = os.listdir(datapath_1)  # 得到文件夹下的所有文件名称
    names = locals()
    for file in files:
        names[file] = pd.read_csv(datapath_1 + "/" + file, chunksize=1000)
    print("done with reading all trajectory")
    with open(datapath_2+ output_str + '_Albany_all_ids_dict.pickle', 'rb') as handle:
        individual_date = pickle.load(handle)
    unique_id_list = list(individual_date.keys())
    print("done with reading all individuals")

    countx=0
    labels_list=[]
    for individual,count_id in zip(unique_id_list,[i for i in range(len(unique_id_list))]):
        if len(list(individual_date[individual].keys()))>30 and len(individual)>10: ###AT LEAST ONE MONTH
            print(individual, count_id)
            df_temp = pd.DataFrame(columns=['latitude','longitude','time'])
            for file,value in individual_date[individual].items():
                value_temp = value
                try:
                    df_temp = pd.concat([df_temp, names[file].loc[value_temp, ['latitude', 'longitude', 'time']]])
                except:
                    print('wrong results')
            df_temp = df_temp.sort_values(by='time').reset_index(drop=True)
            df_temp.to_csv(output_path + output_str + '/individual_raw/' + individual + '.csv')


if __name__ == '__main__':
    output_str = 'Jan'
    # output_str='Apr'

    location_str='/Volumes/SeagateDrive/Trajectory Data/'
    datapath = location_str+"Albany"
    datapath_afterprocess = location_str+'Albany_after_process/'+output_str+'/'
    datapath_merge = location_str+'Albany_after_process/'
    datapath_individual= location_str+output_str+'_individual/individual_raw/'

    if output_str == 'Jan':
        start = datetime.datetime.strptime("01-01-2020", "%d-%m-%Y")
        end = datetime.datetime.strptime("29-02-2020", "%d-%m-%Y")
        date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
        output_list = [date.strftime('%Y%m%d') + '00.csv.gz' for date in date_generated]

    if output_str == 'Apr':
        start = datetime.datetime.strptime("01-03-2020", "%d-%m-%Y")
        end = datetime.datetime.strptime("24-04-2020", "%d-%m-%Y")
        date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
        output_list = [date.strftime('%Y%m%d') + '00.csv.gz' for date in date_generated]

    pre_process(datapath, datapath_afterprocess,output_list)
    merge(datapath_afterprocess,datapath_merge)
    individual_data_process(datapath_afterprocess,datapath_merge,datapath_individual)


