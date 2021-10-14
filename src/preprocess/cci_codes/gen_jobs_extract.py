import os,sys
import numpy as np
from glob import glob
import pandas as pd

####creat the submit_vis.sh (run sh submit_vis.sh)
fnm = '/gpfs/u/home/COVP/COVPlzng/codes/preprocess/submit_vis.sh'
if os.path.exists(fnm):
    os.remove(fnm)

##### read the date list
df=pd.read_csv('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/dates.csv')
date_list=[str(int(i)) for i in df['datestr'].values if np.isnan(i)==False]

<<<<<<< Updated upstream
##### read the id list with [id_str, #record, date]
df=pd.read_csv('/gpfs/u/scratch/COVP/shared/mapping_dict_sum/group_record_date.csv')
id_list =[i for i in df['id_str'].values]

=======
##### read the id list with [id_str, #record, date] ####STASTITICS SUMMARY,  descending order according #records
df=pd.read_csv('/gpfs/u/scratch/COVP/shared/mapping_dict_sum/group_record_date.csv')
id_list =[i for i in df['id_str'].values]


>>>>>>> Stashed changes
#splite the list of ids;
num_id=10 ###number of ids needs to be process
id_splited=np.array_split(id_list, int(len(id_list)/num_id))
id_num_splitted=np.array_split([str(i) for i in range(len(id_list))], int(len(id_list)/num_id))

####oragend the list of ids to strings
for_run=0 ###index of splitted ids
first=True
for x,y in zip(list(id_splited[for_run]), list(id_num_splitted[for_run])):
    if first==True:
        id_list_rum=x
        id_num_list_rum=y
        first=False
    else:
        id_list_rum=id_list_rum+','+x
        id_num_list_rum=id_num_list_rum','+y
    
#print(id_list_rum)
#print(id_num_list_rum)

###write into the submit_vis.sh
with open(fnm,'w+') as file:
    file.write('#!/bin/bash\n')
    for date in date_list:
        cmd = '--wrap=\"python extract.py {} {} {}\"'.format(date,id_list_rum,id_num_list_rum)
        jnm = 'd_{}'.format(date)
        line = "sbatch -c 10 --mem-per-cpu 5g --gres gpu:1 -t 0:40:0 -J {0} -o {0}.out {1}\n".format(jnm,cmd)
        file.write(line)

