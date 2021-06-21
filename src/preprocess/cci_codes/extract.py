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

def infer_indiv_stoppoint( id_indiv,traj_all,path_out):
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
        df_at_stop.to_csv(path_out+'/'+'id_'+id_num+'.csv')

    return is_stop_found




def extract(path, path_data,dict_df_temp):

    traj_all=pd.DataFrame()
    for index,row in dict_df_temp.iterrows():
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
    case = 'extract'
    if case == 'extract':
        f=open('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/extract_test.txt','a')
        date = sys.argv[1]
        id_list = sys.argv[2].split(',')
        id_num_list = sys.argv[3].split(',')
        print(id_list,file=f)
        print(id_num_list,file=f)
        path_parent = '/gpfs/u/scratch/COVP/shared/'
        path_data='/gpfs/u/scratch/COVP/shared/US_raw_data/'
        #####create dircetory
        dirName1 = 'individual_data'  ######genaret folder for mapping_dict
        if not os.path.exists(path_parent + dirName1):
            os.makedirs(path_parent + dirName1)
            print(dirName1, 'folder created')
        else:
            print(dirName1, 'folder already exit')
        path_individual = path_parent + dirName1

        for id_num in id_num_list:
            print(id_num,file=f)
            if not os.path.exists(path_individual +"/"+ id_num):
                os.makedirs(path_individual +"/"+ id_num)
                print(path_individual +"/"+ id_num, 'folder created')

        dict_df=pd.read_csv('/gpfs/u/scratch/COVP/shared/mapping_dict_sum/'+date+'.csv')
        record = []
        
        for id,id_num in zip(id_list,id_num_list):
            print(id,id_num,file=f)
            dict_df_temp=dict_df[dict_df['id_str']==id]
            process = mp.Process(target=extract, args=(path_individual +"/"+ id_num,path_data,dict_df_temp))
            process.start()
            record.append(process)
        for process in record:
            process.join()
        f.close()
        
    if case == 'infostops':
        f = open('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/fail_individal.txt', 'a')
        id_list = sys.argv[1]
        id_list=id_list.strip('][').split(',')
        id_list=[i[1:-1]for i in id_list]
        id_num_list = sys.argv[2]
        id_num_list=id_num_list.strip('][').split(',')
        print(id_num_list)
        path_parent = '/gpfs/u/scratch/COVP/shared/'
        path_individual='/gpfs/u/scratch/COVP/shared/individual_data'
        #####create dircetory
        dirName1 = 'individual_stopoints'  ######genaret folder for mapping_dict
        if not os.path.exists(path_parent + dirName1):
            os.makedirs(path_parent + dirName1)
            print(dirName1, 'folder created')
        else:
            print(dirName1, 'folder already exit')
        path_stopoints = path_parent + dirName1
        
        
        for id,id_num in zip(id_list,id_num_list):
            files_each = os.listdir(path_individual + '/' + id_num)
            combined_traj = pd.concat([pd.read_csv(f, ) for f in files_each])
            if len(files_each)<2000:
                print(id,id_num,'Unsuccess',file=f)
            infer_indiv_stoppoint(id_num, combined_traj,path_stopoints)
        
        f.close()
