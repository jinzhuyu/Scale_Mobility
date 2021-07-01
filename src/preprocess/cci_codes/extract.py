import pickle
import pandas as pd
import os
from collections import defaultdict
import numpy as np
#import infostop
import sys
sys.stdout.flush()
import multiprocessing as mp
import process_utils
import time

def extract(path, path_data,dict_df_temp):
    '''
    given the path for id (path); the path for raw data (path data); and the dirctonary (dict_df_temp)
    then we could get each id's slice (among 2000 slices) and rows for record
    then we could extract files and write it as "path+'/'+date+'.csv'"
    '''
    traj_all=pd.DataFrame()
    for index,row in dict_df_temp.iterrows(): ####in case some ids having multiple slices
        date=str(row['date'])
        i=row['slice']
        record=list(map(int, row['record'].strip('][').split(',')))
        files= os.listdir(path_data + '/' + date)
        #files.remove('.DS_Store')
        files.remove('_SUCCESS')
        files.sort()
        file=files[i]
        print(date,i,file)
        df_file = process_utils.load_data_main(path_data, date, file)
        traj_all=pd.concat([traj_all,df_file.loc[record]])
    traj_all.sort_values(by='time').reset_index(drop=True)
    traj_all.to_csv(path+'/'+date+'.csv')





if __name__ == "__main__":
    #####parameters from .sh file
    date = sys.argv[1]
    id_list = sys.argv[2].split(',')
    id_num_list = sys.argv[3].split(',')
    
    #####create dircetory
    path_parent = '/gpfs/u/scratch/COVP/shared/'
    path_data='/gpfs/u/scratch/COVP/shared/US_raw_data/'
    
    ######genaret folder for all "individual_data"
    dirName1 = 'individual_data'
    if not os.path.exists(path_parent + dirName1):
        os.makedirs(path_parent + dirName1)
        print(dirName1, 'folder created')
    else:
        print(dirName1, 'folder already exit')
    path_individual = path_parent + dirName1
    
    ######genaret folder for each individual under "individual_data"
    for id_num in id_num_list:
        print(id_num)
        if not os.path.exists(path_individual +"/"+ id_num):
            os.makedirs(path_individual +"/"+ id_num)
            print(path_individual +"/"+ id_num, 'folder created')
    
    ######open the dictonary file at the specific date
    dict_df=pd.read_csv('/gpfs/u/scratch/COVP/shared/mapping_dict_sum/'+date+'.csv')
    
    ######begina parallel proprecssing on extracting the data
    record = []
    for id,id_num in zip(id_list,id_num_list):
        print(id,id_num)
        dict_df_temp=dict_df[dict_df['id_str']==id]
        process = mp.Process(target=extract, args=(path_individual +"/"+ id_num,path_data,dict_df_temp))
        process.start()
        record.append(process)
    for process in record:
        process.join()
    
