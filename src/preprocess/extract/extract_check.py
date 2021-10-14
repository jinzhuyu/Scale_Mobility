import pickle
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import infostop
import sys
sys.stdout.flush()
import multiprocessing as mp
from functools import partial
import process_utils
import time

        
        
def extract(each_in_all,path_data, date):
    '''
    given the path for id (path); the path for raw data (path data); and the dirctonary (dict_df_temp)
    then we could get each id's slice (among 2000 slices) and rows for record
    then we could extract files and write it as "path+'/'+date+'.csv'"
    '''
    #print(each_in_all,each_in_all[0])
    [each_str,each]=each_in_all
    path=path_individual + each
    if os.path.isfile(path+'/'+date+'.csv')==False:
        traj_all=pd.DataFrame()
        df_temp=pd.DataFrame()
        dict_each=dict_df_temp[dict_df_temp['id_str']==each_str]
        if len(dict_each)!=0:
            for index,row in dict_each.iterrows():
                slice=row['slice']
                print(each,slice)
                record=list(map(int, row['record'].strip('][').split(',')))
                slice_str=all_files[slice]
                df_temp=process_utils.load_data_main(path_data, date, slice_str)
                traj_all=pd.concat([traj_all,df_temp.loc[record]])
            traj_all.sort_values(by='time').reset_index(drop=True)
            traj_all.to_csv(path+'/'+date+'.csv')
        lst = [df_temp, traj_all,dict_each]
        del df_temp, traj_all,dict_each
        del lst




if __name__ == "__main__":
    date_check = sys.argv[1]
    for_run=int(sys.argv[2])
    nprocess=50

    ###read the ids
    df_data=pd.read_csv('/gpfs/u/scratch/COVP/shared/mapping_dict_sum/group_record_date.csv')
    
    ###paths
    path_data='/gpfs/u/scratch/COVP/shared/US_raw_data/'
    global path_individual
    path_individual='/gpfs/u/scratch/COVP/shared/individual_data/'
    
    ####read the slices name under the date
    global all_files
    all_files= os.listdir(path_data + '/' + date_check)
    all_files.remove('_SUCCESS')
    all_files.sort()
    
    ###read the existing ids
    id_num_list = os.listdir(path_individual)
    id_num_list=list(map(int, id_num_list))
    id_num_list=sorted(id_num_list)[for_run*10000:(for_run+1)*10000]
    id_list=[df_data['id_str'][int(id_num_str)] for id_num_str in id_num_list]
    id_dict=dict(zip(id_list,id_num_list))
    
    ####read the dictionaries for the specific dates
    dict_df=pd.read_csv('/gpfs/u/scratch/COVP/shared/mapping_dict_sum/'+date_check+'.csv')###whichis time consuming
    global dict_df_temp
    dict_df_temp=dict_df[dict_df['id_str'].isin(id_list)]
    del [dict_df]
    print(date_check,'number of ids for check',len(dict_df_temp))
    
    
    ###construct the dict whether the id has the record for date
    id_num_list_check=[]
    id_list_check=[]
    for id_str in dict_df_temp['id_str']:
        id_num=id_dict[id_str]
        id_num_str=str(id_num)
        files_each = os.listdir(path_individual + '/' + id_num_str)
        path=path_individual + id_num_str
        if os.path.isfile(path+'/'+date_check+'.csv')==False:
            id_list_check.append(id_str)
            id_num_list_check.append(id_num_str)
            
    print(date_check,'number of ids for check further',len(id_num_list_check))

    
    if len(id_num_list_check)!=0:
        print('begin extract')
        ######extract each id's data
        id_num_list_check=np.array_split(id_num_list_check, int(len(id_num_list_check)/nprocess)+1)
        id_list_check=np.array_split(id_list_check, int(len(id_list_check)/nprocess)+1)
        count=0
        for each_str,each in zip(id_list_check,id_num_list_check):
            print(count,each,each_str)
            with mp.Pool(len(each)) as pool:
                pool.map(partial(extract, path_data=path_data,date=date_check),list(zip(each_str,each)))
            pool.close()
            pool.join()
            count+=1
    print('done')
    '''
    except:
        f=open('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/extract/failed_extract_check.txt','a')
        print(date_check,for_run,file=f)
        f.close()
    '''

            
                

            
