
import pickle
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import sys
sys.stdout.flush()


def find_inaccuarate(path,date):
    finish=True
    for strx in [ str(i) for i in range(0,2000)]:
        if os.path.exists(path +'/'+date+'_'+strx+'.pickle')==False:
            finish=False
            f=open('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/finish.txt','a')
            print(date,finish,strx,file=f)
            f.close()
            


def sum_dicts(path_each,path_sum,date):
    df= pd.DataFrame(columns=['id_str','date','slice', '#record','record'])
    for strx in [str(i) for i in range(0, 2000)]:
        print(strx)
        if os.path.exists(path_each + '/' + date + '_' + strx + '.pickle'):
            with open(path_each + '/' + date + '_' + strx + '.pickle', 'rb') as handle:
                dictx = pickle.load(handle)
                df_temp = pd.DataFrame(dictx.items(), columns=['id_str', 'record'])
                df_temp['slice']=[strx for j in range(len(df_temp))]
                df_temp['date'] = [date for j in range(len(df_temp))]
                df_temp['#record']=list(map(lambda x: len(x), df_temp['record']))
                df = pd.concat([df, df_temp])
                print(df_temp['#record'].values[0],df_temp['record'].values[0])
    df.to_csv(path_sum+'/'+date+'.csv')


def rank_ids(path_sum,date):
    
    f=open(path_sum+'/statistics_date_df.txt','a')
    print('date','sum_id','sum_id(>5 record)','sum_id(>10 record)' 'sum_record','mean_record','std_record','median_record','max_record','min_record',file=f)
    
    df_each=pd.read_csv(path_sum+'/'+date+'.csv')
    df=df_each.groupby(['id_str','date']).agg({'#record':'sum'}).reset_index()
    df.to_csv(path_sum+'/'+date+'_group_record.csv')
    print('finished')
    
    sum_id=len(df)
    sum_id_5=len(df[df['#record']>5])
    sum_id_10=len(df[df['#record']>10])
    sum_record=df['#record'].sum()
    mean_record=df['#record'].mean()
    std_record=df['#record'].std()
    median_record=df['#record'].median()
    max_record=df['#record'].max()
    min_record=df['#record'].min()
    print(date[0:8],sum_id,sum_id_5,sum_id_10,sum_record,mean_record,median_record)
    print(date[0:8],sum_id,sum_id_5,sum_id_10,sum_record,mean_record,std_record,median_record,max_record,min_record,file=f)
    f.close()


    
def aggregate_all(path_sum):
    df=pd.read_csv('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/dates.csv')
    date_list =[str(int(i)) for i in df['datestr'].values if np.isnan(i)==False]
    df_all=pd.DataFrame()
    for date in date_list:
        print(date)
        df_temp=pd.read_csv(path_sum+'/'+date+'_group_record.csv')
        df_temp['date']=1
        df_all=pd.concat([df_all,df_temp])
        df_all=df_all.groupby(['id_str',]).agg({'#record':'sum','date':'sum'}).reset_index()
        print(date,'finished')
        df_all.to_csv(path_sum+'/group_record_date_temp.csv')
    df_all.sort_values(by='#record',ascending=False).reset_index(drop=True)
    df_all.to_csv(path_sum+'/group_record_date.csv')
    
    print(len(df_all),df_all['#record'].sum(),df_all['#record'].mean(),df_all['#record'].median(),'average_date_having_data',df_all['date'].sum(),'median_having_data',df_all['date'].mean())
    
    


if __name__ == "__main__":
    path_parent = '/gpfs/u/scratch/COVP/shared/'
    path_each = '/gpfs/u/scratch/COVP/shared/mapping_dict_each/'
    case ='sum_date'
    case = 'sum_all'
    if case=='sum_date':
        date = sys.argv[1]

        #find_inaccuarate(path, date)
        dirName2 = 'mapping_dict_sum'  ######genaret folder for indiviudal_data
        if not os.path.exists(path_parent + dirName2):
            os.makedirs(path_parent + dirName2)
            print(dirName2, 'folder created')
        else:
            print(dirName2, 'folder already exit')
        path_sum=path_parent + dirName2

        sum_dicts(path_each,path_sum,date)

    if case=='sum_all':
        date = sys.argv[1]
        path_sum = path_parent + 'mapping_dict_sum'
        #rank_ids(path_sum,date)
        aggregate_all(path_sum)



