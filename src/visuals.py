# -*- coding: utf-8 -*-
# [pydocstyle]
# inherit = false
# ignore = D100,D203,D405
# match = *.py
"""
Created on Tue Aug 17 23:20:17 2021

@author: Jinzh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from datetime import datetime
from matplotlib import cm
# import matplotlib.dates as mdates
# from matplotlib.dates import DateFormatter
import os
import random

# TODO: two sequences with cross-correlation https://stats.stackexchange.com/questions/19367/creating-two-random-sequences-with-50-correlation
# TODO: use OOP instead for better maintenance

# class DataPlots:
    
#     def __init__():
#         pass

def plot_explore_propensity(traj_df_1, traj_df_2):
    """
    Plot the propensity for exploration over time.
    the progression of number of unique locations visited over time

    Parameters
    ----------
    traj_df : pandas df.
        tranjectories of users, columns: user_id, lat(location_id),
        long (location_id), time, num_uniq_loc_visited

    Returns
    -------
    None.

    """
    # get user ids
    # user_ids = traj_df['user_id'].unique()
    # for each user
        # plot the number of unique locations visited at each time point
    # for each user:
    
    # or for each time point
    
    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(traj_df_1['time_diff'], traj_df_1['num_uniq_loc_visited'],
               color='red')
    ax.scatter(traj_df_2['time_diff'], traj_df_2['num_uniq_loc_visited'],
               color='blue')

    ax.set(xlabel=r't(h)',
           ylabel=r'S(t)',
           title='Propensity for exploration over time')
    # plt.xticks(rotation=20)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax.semilogx()
    ax.loglog()
    ax.set_ylim(bottom=10**0)
    ax.set_xlim(left=10**0)
    
    ax.legend(['With travel and sleep time','without travel and sleep time'], loc='upper left')

    plt.show()
    
    return None


def get_user_visit_freq(df):
    '''
    get location visitation relative frequency for each user
    
    '''
    
    user_ids = df['user_id'].unique()
    
    # location_relative_freq_list = []
    
    for i in user_ids:
    
        location_freq = df.loc[df['user_id']==i, 'latitude'].value_counts()
        location_relative_freq = location_freq/location_freq.sum()
        location_relative_freq_sort = location_relative_freq.sort_values(ascending=False)
        
        location_relative_freq_df_temp = location_relative_freq_sort.to_frame()
        location_relative_freq_df_temp['user_id'] = i
        location_relative_freq_df_temp['rank'] = list(range(1,location_relative_freq_df_temp.shape[0]+1))
      
        # location_relative_freq_list.append()
        if i==0:
            location_relative_freq_df = location_relative_freq_df_temp
        else:
            location_relative_freq_df = pd.concat([location_relative_freq_df, location_relative_freq_df_temp], axis=0)
      
    return location_relative_freq_df


def plot_visit_freq(df1, df2):
    '''
    plot f_k (visitation frequency) vs k (k-th most visited location of a user for different S values, 20, 40, and 60)
    
    Parameters
    ----------
    traj_df_1 : df with travel and sleep time.
    traj_df_2 : df without travel and sleep time.

    '''
    # df_temp_1 = visit_freq_df_1.loc[visit_freq_df_1['user_id']==0]
    # df_temp_2 = visit_freq_df_2.loc[visit_freq_df_2['user_id']==0]
       
    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(df1['rank'], df1['latitude'],
               color='red')
    ax.scatter(df2['rank'], df2['latitude'],
               color='blue')

    ax.set(xlabel='Rank',
           ylabel= 'Relative frequency',
           title='Location frequency over location rank')
    # plt.xticks(rotation=20)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax.semilogy()
    ax.loglog()
    ax.set_ylim(top=10**0)
    ax.set_xlim(left=10**0)
    
    ax.legend(['With travel and sleep time','without travel and sleep time'], loc='upper right')

    plt.show()  
   
    
# autocorrelation function at lag k for a stationary stochastic process
# For nonstationary process, ACF depends on the pivot point besides the lag
def get_autocorr(data_list, lag_list=None):
    '''https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
    '''  
    # exception handle
    # lag is None
    n = len(data_list)
    if lag_list == None:
       lag_list = list(range(1, n)) 
    # maximum value of lag
    if n <= max(lag_list):
        raise ValueError('Oops! Length of series ({}) <= maximum lag ({}) >'.format(n, max(lag_list)))
    # lag is not list
    if np.isscalar(lag_list):
        lag_list = list(lag_list)        
    
    mean = np.mean(data_list)
    var = np.sum((data_list - mean) ** 2) / float(n)

    def ACF(lag):
        ACF = ((data_list[:n - lag] - mean) * (data_list[lag:] - mean)).sum() / float(n) / var
        return round(ACF, 5)
    
    autocorr_list = list(map(ACF, lag_list))
    
    return autocorr_list


#TODO: difference between entrop values for weekly location sequences
# def entropy_diff(data_list, lag_list):

# !!! jaccard similarity does not consider sequence of locations
def get_jaccard(loc_id_list, time_point_list, time_point_eval, lag_list):
    '''
    

    Parameters
    ----------
    loc_id_list : TYPE
        DESCRIPTION.
    time_point_list : all the time point in the data
    time_point_eval : starting time point to evalue the jaccard similarity
    lag_list : TYPE
        DESCRIPTION.

    Returns
    -------
    jaccard_idx_arr : TYPE
        DESCRIPTION.

    '''
    
    n_time_eval = len(time_point_eval)
    n_lag = len(lag_list)
    # jaccard_idx_dict = {}
    jaccard_idx_arr = np.ones((n_time_eval, n_lag))*-1
    # time= time_point_list[2]
    # lag = lag_list_unix[1]
    for i in range(n_time_eval):
        
        # time = time_point_eval[i]
        # jaccard_idx_dict[time] = {}
        idx_time = 1   #time_point_list.index(time)
        time_point_list_select = time_point_list[idx_time:]        
        # print('\n===== time: {}'.format(time))
        for j in range(n_lag):
            # print('\n===== lag: {}'.format(lag))
        # def jaccard(lag, time):
            # use data from currrent time and onward
            lag = lag_list[j]
            
            try:
                # divide the locations into two groups depending on the lag: locations after t and those after t+lag
                # identify the index of location at the separation time point in the first and in the end 
                idx_separ_start = next(k for k, value in enumerate(time_point_list_select) \
                                       if value > (time_point_list_select[0]+lag))
                idx_separ_end = next(k for k, value in enumerate(time_point_list_select) \
                                     if value > (time_point_list_select[-1]-lag))
                
                #define Jaccard similarity function
                loc_id_list_no_lag = loc_id_list[idx_time:-idx_separ_end]  # current time point to the last lag before the end
                loc_id_list_lag = loc_id_list[idx_separ_start:]  # current time point+lag to the end
                intersec = len( list( set(loc_id_list_no_lag).intersection(loc_id_list_lag) ) )
                union = (len(loc_id_list_no_lag) + len(loc_id_list_lag)) - intersec
                
                if union == 0:
                    print('\n===== Union should not be empty! The current lag is: {}'.format(lag))
                    jaccard_score = -1
                else:
                    jaccard_score = float(intersec) / union
                # return jaccard_idx
                # jaccard_idx_dict[time].update({lag:jaccard_idx})
                jaccard_idx_arr[i,j] = jaccard_score
            except:
                continue
    # jaccard_list = list(map(jaccard, lag_list))
    # jaccard_list = [jaccard(x,y) for x in lag_list for y in time_point_list]
    
    return jaccard_idx_arr
    
# jaccard_idx_arr = get_jaccard(loc_id_list_disc, time_point_list, lag_list=lag_list_unix)

  
def plot_traj_in_date_interval(traj_df_1, traj_df_2, date_interval_lb=24*30*11.5, date_interval_ub=24*30*12):  
    
    traj_in_date_interval_1 = traj_df_1.loc[traj_df_1['time_diff'].between(date_interval_lb, date_interval_ub)]
    traj_in_date_interval_2 = traj_df_2.loc[traj_df_2['time_diff'].between(date_interval_lb, date_interval_ub)]
    
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(traj_in_date_interval_1['time_diff'], traj_in_date_interval_1['latitude'],
               color='darkred')
    ax.scatter(traj_in_date_interval_2['time_diff'], traj_in_date_interval_2['latitude'],
               color='navy')

    ax.set(xlabel='t(h)',
           ylabel='Location ID',
           title='Trajectory by location ID for two weeks after 11.5 months')
    # plt.xticks(rotation=20)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax.semilogx()
    # ax.loglog()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=date_interval_lb-1, right=date_interval_ub+1)
    
    ax.legend(['With travel and sleep time','without travel and sleep time'], loc='upper left')

    plt.show()
    
    return None



def plot_location_distri_in_date_interval(traj_df_1, traj_df_2, date_interval_lb=24*30*11.5, date_interval_ub=24*30*12):  
    
    traj_in_date_interval_1 = traj_df_1.loc[traj_df_1['time_diff'].between(date_interval_lb, date_interval_ub)]
    traj_in_date_interval_2 = traj_df_2.loc[traj_df_2['time_diff'].between(date_interval_lb, date_interval_ub)]
    
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(traj_in_date_interval_1['latitude'], alpha=0.5, color='red')
    # ax.hist(traj_in_date_interval_2['latitude'],
    #             color='blue')

    ax.set(xlabel='Location ID',
           ylabel='Frequency',
           title='Distribution of locations for two weeks after 11.5 months')
    # plt.xticks(rotation=20)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax.semilogx()
    # ax.loglog()
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(left=date_interval_lb-1, right=date_interval_ub+1)
    
    ax.legend(['With travel and sleep time','without travel and sleep time'], loc='upper left')

    plt.show()
    
    return None
    

def plot_temp_simil_in_date_interval(traj_df_1, date_interval_lb=24*30*11, date_interval_ub=24*30*12, simil_type='jaccard'):  
    
    traj_in_date_interval_1 = traj_df_1.loc[traj_df_1['time_diff'].between(date_interval_lb, date_interval_ub)]
    
    user_id_list = [10, 11, 12]
    fig, ax = plt.subplots(figsize=(6, 4))
    lag_list = [ele*2 for ele in range(1,60)]
    for i in user_id_list:
        loc_id_list = traj_in_date_interval_1.loc[traj_in_date_interval_1['user_id']==i, 'latitude'].tolist()
        simil = get_autocorr(loc_id_list, lag_list)
        ax.plot(lag_list, simil, label='Agent ID: {}'.format(i))
    
    ax.set(xlabel='Time lag',
           ylabel='Weekly similarity using {}'.format(simil_type),
           title='Weekly similarity at different lags')
    ax.legend(loc='upper left')
    ax.set_ylim([-0.2, 0.4])
    plt.show()
    
    return None


def plot_weekly_simil(traj_df_1, user_id_list, date_interval_lb=24*30*8, date_interval_ub=24*30*12, simil_type='auto-correlation'):  
    
    traj_in_date_interval_1 = traj_df_1.loc[traj_df_1['time_diff'].between(date_interval_lb, date_interval_ub)]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    n_week = 12
    lag_list = [ele for ele in range(1,n_week+1)]
    for i in user_id_list:
        print('\n===== user id: {} ====='.format(i))
        loc_id_list = traj_in_date_interval_1.loc[traj_in_date_interval_1['user_id']==i, 'latitude'].tolist()
        # simil = get_jaccard(loc_id_list, lag_list)
        simil = get_autocorr(loc_id_list, lag_list)
        ax.plot(lag_list, simil, label='Agent ID: {}'.format(i))
    
    ax.set(xlabel='Time lag (week)',
           ylabel='Weekly similarity using {}'.format(simil_type),
           title='Weekly similarity at different time lags')
    ax.legend(loc='upper left')
    # ax.set_ylim([0, 1])
    plt.show()
    
    return None


                           
def get_time_diff(df):
    ''' 
    add another column to df, 'time_diff', contains the difference between the column 'datetime' to the initial time point
    '''
   
    traj_datetime = pd.to_datetime(df['datetime'])
    seconds_in_hour = 3600
    time_diff_timestamp = (traj_datetime - traj_datetime[0]).astype('timedelta64[s]') / seconds_in_hour

    df['time_diff'] = time_diff_timestamp
    
    return df



# import mobility data sample of certain users

def get_location_id(user_id, lag_list):
    """X."""
    file_name = '../data_processed/mobility_data_sample/{}.csv'.format(user_id)
    real_mobility_sample = pd.read_csv(file_name)
    # get location id
    loc_id_list = real_mobility_sample['label'].tolist()
    # calculate autocorrelation (acf)
    acf = get_autocorr(loc_id_list, lag_list)
    
    return acf



# plot acf
def plot_acf(user_id_list):
    user_id_list = [3,4,5,8,9,10]
    fig, ax = plt.subplots(figsize=(6, 4))
    lag_list = [ele*5 for ele in range(1,20)]
    for i in user_id_list:
        acf = get_location_id(i, lag_list)
        ax.plot(lag_list, acf, label='Agent ID: {}'.format(i))
    
    ax.set(xlabel='Lag',
           ylabel='Autocorrelation',
           title='Autocorrelation at different time lags')
    ax.legend()
    plt.show()

# TODO: the time difference between two locations are not equal 


def main():
    
    import os
    os.chdir('c:/code/Scale_Mobility/src')
    
    from utils import format_fig
    # format figures
    format_fig()   
    
    # syn_traj_wt_travel_sleep.to_csv('syn_traj_wt_travel_sleep_one_year.csv')
    # syn_traj_wo_travel_sleep.to_csv('syn_traj_wo_travel_sleep_one_year.csv')
    
    # import data
    syn_traj_wt_travel_sleep = pd.read_csv('../outputs/simulated_mobility_trajectories/syn_traj_wt_travel_sleep_one_year.csv')
    syn_traj_wo_travel_sleep = pd.read_csv('../outputs/simulated_mobility_trajectories/syn_traj_wo_travel_sleep_one_year.csv')      
    

    
    # get time difference
    syn_traj_wt_travel_sleep = get_time_diff(syn_traj_wt_travel_sleep)
    syn_traj_wo_travel_sleep = get_time_diff(syn_traj_wo_travel_sleep)
    
    # autocorrelation over time at lags: 24h, 1 week, 1 month, 6 month, 1 year
    lag_list= [ele*24 for ele in [1, 7, 30]]
    autocorr_list = get_autocorr(data_list, lag_list)
    
    # plot_temp_simil_in_date_interval(syn_traj_wt_travel_sleep)
    user_id_list = [ele for ele in range(1,10)]
    plot_weekly_simil(syn_traj_wt_travel_sleep, user_id_list)
    

    # plot trajectories by location id from 11.5 month to 12 month
    id_temp = 5    
    plot_traj_in_date_interval(syn_traj_wt_travel_sleep[syn_traj_wt_travel_sleep['user_id'] == id_temp],
                               syn_traj_wo_travel_sleep[syn_traj_wo_travel_sleep['user_id'] == id_temp])

    plot_location_distri_in_date_interval(syn_traj_wt_travel_sleep[syn_traj_wt_travel_sleep['user_id'] == id_temp],
                                          syn_traj_wo_travel_sleep[syn_traj_wo_travel_sleep['user_id'] == id_temp],
                                          date_interval_lb=24*30*11.75, date_interval_ub=24*30*12)
     
    plot_location_distri_in_date_interval(syn_traj_wt_travel_sleep[syn_traj_wt_travel_sleep['user_id'] == id_temp],
                                          syn_traj_wo_travel_sleep[syn_traj_wo_travel_sleep['user_id'] == id_temp],
                                          date_interval_lb=24*30*11.5, date_interval_ub=24*30*11.75)
    
    # plot exploration propensity over time for all users
    plot_explore_propensity(syn_traj_wt_travel_sleep, syn_traj_wo_travel_sleep)

    # plot exploration propensity over time for 1 user only
    id_temp = 5
    plot_explore_propensity(syn_traj_wt_travel_sleep[syn_traj_wt_travel_sleep['user_id'] == id_temp],
                            syn_traj_wo_travel_sleep[syn_traj_wo_travel_sleep['user_id'] == id_temp])
    
   
    
    # plot f_k (visitation frequency) vs k (k-th most visited location of a user for different S values, 20, 40, and 60)
    visit_freq_df_wt_travel_sleep = get_user_visit_freq(syn_traj_wt_travel_sleep)
    visit_freq_df_wo_travel_sleep = get_user_visit_freq(syn_traj_wo_travel_sleep)
    
    plot_visit_freq(visit_freq_df_wt_travel_sleep[visit_freq_df_wt_travel_sleep['user_id'] == id_temp],
                    visit_freq_df_wo_travel_sleep[visit_freq_df_wo_travel_sleep['user_id'] == id_temp])

    plot_visit_freq(visit_freq_df_wt_travel_sleep, visit_freq_df_wo_travel_sleep)
    






#####09152021
# plot real mobility patterns using data of 10 highly mobility individuals
    
def import_real_mombil_data(path):
    # import mobility data of several individuals

    from glob import glob
    file_names = glob(path + '/*.csv')
    indiv_mobil_data = pd.concat([pd.read_csv(file) for file in file_names])
    for name in ['Unnamed: 0', 'Unnamed: 0.1']:
        if name in indiv_mobil_data.columns:
            indiv_mobil_data = indiv_mobil_data.drop(columns=[name])

    # indiv_mobil_data_raw.columns: label-location _id; start and end are in unix time format;
    # -1 in travel-related columns means the individual is in travel
    
    return indiv_mobil_data



def discretize_stay_time(loc_id_list, start_time_list, end_time_list, time_interv=10):
    ''' duscrete stay time of individuals
        For example, 30 mins stay at location 1 -> three times at location 1 using interv_witdh of 10 mins
        start_time, end_time: unix time format
        time_interv: discretization granularity, in mins.
    '''
    # stay time in mins at each location
    n_sec = 60
    stay_time_list = [(end_time_list[i] - start_time_list[i])/n_sec for i in range(len(start_time_list))]
    # discretize the stay time at each location
    n_disc_list = [int(round(ele/time_interv, 0)) for ele in stay_time_list]
    # get time point within start and end time at a location
    time_point_nest_list = [np.linspace(start_time_list[i], end_time_list[i],
                                        num=n_disc_list[i], endpoint=True).astype(int).tolist() \
                            for i in range(len(stay_time_list))]    
    time_point_list = [ele for sublist in time_point_nest_list for ele in sublist]  
    
    loc_id_nest_list = [[loc_id_list[i]]*n_disc_list[i] for i in range(len(loc_id_list))]
        # datetime.fromtimestamp(unix_time)
    loc_id_list = [ele for sublist in loc_id_nest_list for ele in sublist]
    
    return loc_id_list, time_point_list


# in the simulated data, extract stay time as well.
def plot_indiv_jaccard_sample(indiv_mobil_data):
    
    agent_id_list = indiv_mobil_data['id_indiv'].unique()
    n_agent = len(agent_id_list)
    
    scale_y = 100
    n_sec_in_day = 3600*24
    color = iter(cm.rainbow(np.linspace(0,1,n_agent)))        
    
    # plt.figure(figsize=(6,4))
    
    # lag_list = list(range(30,50))
    for agent_id in agent_id_list:
        # for id in id_indiv
        loc_id_list = indiv_mobil_data.loc[indiv_mobil_data['id_indiv']==agent_id, 'label'].tolist()
        start_time_list = indiv_mobil_data.loc[indiv_mobil_data['id_indiv']==agent_id, 'start'].tolist()
        end_time_list = indiv_mobil_data.loc[indiv_mobil_data['id_indiv']==agent_id, 'end'].tolist() 
        loc_id_list_disc, time_point_list = discretize_stay_time(loc_id_list, start_time_list, end_time_list, time_interv=5)
         
        n_day_max = int((end_time_list[-1] - start_time_list[0])/n_sec_in_day//1)
        lag_list = list(range(n_day_max))
        # if n_day_max >=30:
        #     lag_list = [1,2,3,7,15,30]
        # else:
        #     lag_list = list(range(n_day_max))
        lag_list_unix = [ele*n_sec_in_day for ele in lag_list]  
        
        # jaccard_list = get_jaccard(loc_id_list_disc, time_point_list, lag_list_unix)



        # jaccard_list = [ele*scale_y for ele in jaccard_list]
        # jaccard_index = get_jaccard(loc_id_list_disc, time_point_list, lag_list_unix)
        

        # jaccard_list = [ele*scale_y for ele in jaccard_list]
        
        # plt.plot(lag_list, jaccard_list, label='Agent {}'.format(agent_id), color=next(color))
        
        jaccard_idx_arr = get_jaccard(loc_id_list_disc, time_point_list, lag_list=lag_list_unix)
        
        jaccard_idx_arr[jaccard_idx_arr == -1] = np.NaN
        
        plt.plot()
        for i in np.linspace(2, round(jaccard_idx_arr.shape[0]/2), num=10).astype(int):
            plt.plot(lag_list, jaccard_idx_arr[i,:]*scale_y, '-', label='{}-th time point'.format(i))
        # plt.ylim(bottom=0)
        
        
        plt.xlabel('Lag (day)')
        plt.ylabel(r'Jaccard similarity ($\times 10^{-2})$')
            
        plt.legend()
        
        plt.xlim(right=max(lag_list)+5)
        plt.ylim(top=3, bottom=-0.1)
        plt.title('Jaccard similarity over different lag values at different time for agent {}'.format(agent_id))        
        plt.show()



from joblib import Parallel, delayed
##################
# 092202021
# plot the jaccard similarity of real mobility data. csv file for each individual are stored in folder named by date.
# get folder name list and individual id list

dir_raw_mob_data = '../data/mobility_data_6_month/'
folder_list = list(os.listdir(dir_raw_mob_data))


# indiv_id_list = [f for f in sorted(os.listdir('../data_processed/mobility_data_6_month/{}'.format(folder_list[0])))]
# indiv_id_list = [x.replace('.csv', '') for x in indiv_id_list][1:-1]
# indiv_id = indiv_id_list[0]
# df = import_indiv_mobil(indiv_id, folder_list)

# TODO: add individual id, stay duration. 
# TODO: relabel locations. right now lavels of individuals share location ids for the same day.
# TODO: save data of for example 10000 individuals as one file.

# save as pickle for faster readin

def import_indiv_mobil(indiv_id, folder_list):
    def loop(folder_name):
        # read individual data within each folder
        path_temp = '../data_processed/mobility_data_6_month/{}/{}.csv'.format(folder_name,indiv_id)
        if os.path.exists(path_temp):
            df = pd.read_csv(path_temp)
        else:
            df = pd.DataFrame()
        return df        
    df_list = Parallel(n_jobs=8,verbose=0)(delayed(loop)(folder_name) for folder_name in folder_list)
    df = pd.concat(df_list, ignore_index=True)
    return df
from time import time
start = time()
indiv_id = '00001'
import_indiv_mobil(indiv_id, folder_list)
end = time()
print("Time after parallelization:", end - start)

def plot_indiv_jaccard(indiv_id_list, n_rand_indiv=50):
    fig, ax = plt.subplots(figsize=(8, 6))
    scale_jaccard = 1000
    n_sec_in_day = 3600*24
    # select a random list of individuals
    # random.seed(11235)
    random_indiv_id_list = [random.randint(0,len(indiv_id_list)) for i in range(n_rand_indiv)]
    loop_idx = 1
    for indiv_id in [indiv_id_list[i] for i in random_indiv_id_list]:
        
        print('\n===== Individual {} ====='.format(loop_idx))
        # get indiv mobil data
        indiv_df =  import_indiv_mobil(indiv_id, folder_list) 
        # calculate jaccard for each individual

        loc_id_list = indiv_df['label'].tolist()
        start_time_list = indiv_df['start'].tolist()
        end_time_list = indiv_df['end'].tolist() 
        loc_id_list_disc, time_point_list = discretize_stay_time(loc_id_list, start_time_list,
                                                                 end_time_list, time_interv=10)         
        lag_list = [1,7,30,60, 90]    #list(range(n_day_max))
        lag_list_unix = [ele*n_sec_in_day for ele in lag_list]     
        # jaccard_list = get_jaccard(loc_id_list_disc, time_point_list, lag_list_unix)
        # jaccard_list = [ele*scale_y for ele in jaccard_list]
        # jaccard_index = get_jaccard(loc_id_list_disc, time_point_list, lag_list_unix)
        # jaccard_list = [ele*scale_y for ele in jaccard_list]
        # plt.plot(lag_list, jaccard_list, label='Agent {}'.format(agent_id), color=next(color))
        
        jaccard_idx_arr = get_jaccard(loc_id_list_disc, time_point_list, time_point_eval=[1577908663], lag_list=lag_list_unix)
        

        #plot this individual's result
        plt.plot(lag_list, jaccard_idx_arr[0]*scale_jaccard, '-', label='Individual {}'.format(indiv_id))
        
        loop_idx += 1
    # format fig
    plt.xlabel('Lag (day)')
    plt.ylabel(r'Jaccard similarity ($\times 10^{-2})$')    
    plt.legend()
    plt.xlim(right=max(lag_list)+5)
    plt.ylim(top=3, bottom=-0.1)
    plt.title('Jaccard similarity over different lag values')        
    plt.show()
    
    
n_rand_indiv=20
plot_indiv_jaccard(indiv_id_list, n_rand_indiv)


# ### 3d plot
# y = time_point_list
# x = lag_list
# X, Y = np.meshgrid(x, y)
# Z = jaccard_idx_arr

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50)  #, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_zlim([0, 0.002])


############    
if __name__ == '__main__':
    
    import os
    os.chdir('c:/code/Scale_Mobility/src')
    
    from my_utils import format_fig
    # format figures
    format_fig()   
    
    path = '../data_processed/mobility_data_sample'
    # path = '../data_processed/mobility_6_month'
    # C:\code\Scale_Mobility\data_processed\mobility_data_6_month\2020010100
    indiv_mobil_data = import_real_mombil_data(path)
    plot_indiv_jaccard(indiv_mobil_data)  
    
df = pd.read_csv('../data_processed/mobility_data_6_month/2020010200/00009.csv')    
indiv_id_list = df['id_str'].unique().tolist()

for i in range(len(indiv_id_list)):
    
    indiv_id = df['id_str'].unique()[i]
    df_indiv_temp = df.loc[df['id_str']==indiv_id]
    n_points = len(df_indiv_temp.index)
    n_points
    print('\n# of points is {} of the {}-th indivi {}'.format(n_points, i, indiv_id))
    
    start_dt = datetime.fromtimestamp(df_indiv_temp['start'].item())
    end_dt = datetime.fromtimestamp(df_indiv_temp['end'].item())
    
    time_duration = (end_dt - start_dt).seconds/3600
    
