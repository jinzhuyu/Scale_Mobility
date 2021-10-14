import os,sys
import numpy as np
from glob import glob
import pandas as pd

fnm = '/gpfs/u/home/COVP/COVPlzng/codes/preprocess/extract/submit_vis_check.sh'
if os.path.exists(fnm):
    os.remove(fnm)


df=pd.read_csv('/gpfs/u/home/COVP/COVPlzng/codes/preprocess/dates.csv')
ptms =[str(int(i)) for i in df['datestr'].values if np.isnan(i)==False]

for_run=1

with open(fnm,'w+') as file:
    file.write('#!/bin/bash\n')
    for ptm in ptms:
        cmd = '--wrap=\"python extract_check.py {} {}\"'.format(ptm,for_run)
        jnm = 'check_{}'.format(ptm[2:8])
        line = "sbatch -c 5 --mem-per-cpu 5g --gres gpu:1 -t 1:00:00 -J {0} -o {0}.out {1}\n".format(jnm,cmd)
        file.write(line)


