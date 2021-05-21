import infostop1

# from infostop1.infostop11 import models #import Infostop
# model = infostop11.Infostop()

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


def infer_stop_points(pathraw,path_output):
    '''
    infer the stop point of each individual

    Parameters
    ----------
    pathraw : TYPE
        File name contains the ID of each individual.
    path_output : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    r1 = 30;r2 = 30;min_staying_time = 600;max_time_between = 86400;
    files = os.listdir(pathraw)
    for file in files[1:len(files)]:
        df_temp=pd.read_csv(pathraw+file)
        individual=file[0:-4]
        if np.any(np.isnan(np.vstack(df_temp[['latitude', 'longitude', 'time']].values)))==False:
            model_infostop = infostop.Infostop(r1 =r1,
                                    r2 =r2,
                                    label_singleton=False,
                                    min_staying_time = min_staying_time,
                                    max_time_between = max_time_between,#86400
                                    min_size = 2)
            finding_stops=True
            #labels = model_infostop.fit_predict(df_temp[['latitude', 'longitude', 'time']].values)

            try:
                # what are labels: transition -1; if stops, then the stop id, such as 1, 2, 3,
                labels = model_infostop.fit_predict(df_temp[['latitude','longitude','time']].values)
            except:
                finding_stops=False
                
            # remove labels that are transitions    
            if finding_stops==True and np.max(labels)>1:
                count = 0
                df_ouput = pd.DataFrame(columns=['individual', 'label', 'start', 'end', 'latitude', 'longitude'])
                position_dict=dict(zip(df_temp['time'].values,df_temp[['latitude','longitude']].values))
                trajectory = infostop.postprocess.compute_intervals(labels, df_temp['time'].values)
                for i in range(len(trajectory)):
                    time_here=trajectory[i][1]
                    lat_temp=position_dict[time_here][0]
                    lon_temp=position_dict[time_here][1]
                    df_ouput.loc[count]=[individual]+trajectory[i]+[position_dict[time_here][0],position_dict[time_here][1]]
                    count+=1
                df_ouput.to_csv(path_output+individual+'.csv',index = False)
                # stop point for individual, time and cooridinates at stop points

os.chdir("./preprocess")
def main():
    output_str = 'Jan'
    path_individual= output_str+'_individual/individual_raw/'
    path_individaul_stopoints= output_str+'_individual/individual_stopoints/'
    stop_points(path_individual, path_individaul_stopoints)


if __name__ == '__main__':
    main()
else:
    print("The main() function in 'stopoint' did not execute")