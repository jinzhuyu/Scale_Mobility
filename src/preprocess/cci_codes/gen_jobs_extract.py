import os,sys
import numpy as np
from glob import glob
import pandas as pd

fnm = '/gpfs/u/home/COVP/COVPlzng/codes/preprocess/submit_vis.sh'
if os.path.exists(fnm):
    os.remove(fnm)


df=pd.read_csv('/gpfs/u/scratch/COVP/shared/mapping_dict_sum/group_record_date.csv')
id_list =[i for i in df['id_str'].values]

df=pd.read_csv('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/dates.csv')
ptms =[str(int(i)) for i in df['datestr'].values if np.isnan(i)==False]


num_id=10
id_splited=np.array_split(id_list, int(len(id_list)/num_id))
id_num_splitted=np.array_split([str(i) for i in range(len(id_list))], int(len(id_list)/num_id))

first=True
for x,y in zip(list(id_splited[0]), list(id_num_splitted[0])):
    if first==True:
        id_list_rum=x
        id_num_list_rum=y
        first=False
    else:
        id_list_rum=id_list_rum+','+x
        id_num_list_rum=id_num_list_rum','+y
    
print(id_list_rum)
print(id_num_list_rum)


with open(fnm,'w+') as file:
    file.write('#!/bin/bash\n')
    for ptm in ptms:
        cmd = '--wrap=\"python extract.py {} {} {}\"'.format(ptm,id_list_rum,id_num_list_rum)
        jnm = 'd_{}'.format(ptm)
        line = "sbatch -c 10 --mem-per-cpu 5g --gres gpu:1 -t 0:40:0 -J {0} -o {0}.out {1}\n".format(jnm,cmd)
        file.write(line)

