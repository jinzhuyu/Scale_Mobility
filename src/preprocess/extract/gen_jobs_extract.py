import os,sys
import numpy as np
from glob import glob
import pandas as pd

fnm = '/gpfs/u/home/COVP/COVPlzng/codes/preprocess/extract/submit_extract.sh' ##sh ubmit_vis.sh -> begins run
if os.path.exists(fnm):
    os.remove(fnm)



today='2021-07-13' ###begin date
hour=0 ###begin hour

amount=10
run_index_start=0
run_index_end=1

with open(fnm,'w+') as file:
    file.write('#!/bin/bash\n')
    for num_for_run in range(run_index_start,run_index_end):
        if hour<10:
            begin_str=today+'T0'+str(hour)+':00:00'
        else:
            begin_str=today+'T'+str(hour)+':00:00'
            
        cmd = '--wrap=\"python extract.py  {} {}\"'.format(run_index_start*amount,(run_index_start+1)*amount)
        jnm = '{}'.format("ids_"+str(num_for_run))
        line = "sbatch -c 5 --mem-per-cpu 1g --gres gpu:1 --begin={0} -t 3:00:00 -J {1} -o {1}.out {2}\n".format(begin_str,jnm,cmd)
        file.write(line)
    hour+=1



