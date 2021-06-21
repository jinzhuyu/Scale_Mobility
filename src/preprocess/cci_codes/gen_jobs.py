import os,sys
import numpy as np
from glob import glob
import pandas as pd

fnm = '/gpfs/u/home/COVP/COVPlzng/codes/preprocess/submit_vis.sh'
if os.path.exists(fnm):
    os.remove(fnm)


df=pd.read_csv('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/dates.csv')
ptms =[str(int(i)) for i in df['datestr'].values if np.isnan(i)==False]

opt = 'sgd'
source = 'test'

with open(fnm,'w+') as file:
    file.write('#!/bin/bash\n')

    for ptm in ptms[0:1]:
        cmd = '--wrap=\"python dict_built.py {}\"'.format(ptm)
        jnm = 'dict_{}'.format(ptm)
        line = "sbatch -c 10 --mem-per-cpu 5g --gres gpu:1 -t 0:40:0 -J {0} -o {0}.out {1}\n".format(jnm,cmd)
        file.write(line)

