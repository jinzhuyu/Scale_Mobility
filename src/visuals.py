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
# import matplotlib.dates as mdates
# from matplotlib.dates import DateFormatter



# TODO: two sequences with cross-correlation https://stats.stackexchange.com/questions/19367/creating-two-random-sequences-with-50-correlation


# set default plot parameters
def set_default_plot_param():
	# %matplotlib inline
	# from matplotlib import pyplot as plt
    
    plt.style.use('classic')
    
    # plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams['font.weight']= 'normal'
    plt.rcParams['figure.figsize'] = [6, 6*3/4]
   
    plt.rcParams['figure.facecolor'] = 'white'

    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.axisbelow'] = True 
    plt.rcParams['axes.titlepad'] = 14  # title to figure
    plt.rcParams['axes.labelpad'] = 3 # x y labels to figure
    plt.rc('axes', titlesize=15, labelsize=14, linewidth=1)    # fontsize of the axes title, the x and y labels
    
    # plt.rc('lines', linewidth=1.8, markersize=6, markeredgecolor='none')
    
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.rcParams['axes.formatter.useoffset'] = False # turn off offset
    # To turn off scientific notation, use: ax.ticklabel_format(style='plain') or
    # plt.ticklabel_format(style='plain')

    
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.framealpha"] = 0.5

    
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 500
    
#    plt.rc('text', usetex=False)
set_default_plot_param()



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
    traj_df_1 : TYPE
        DESCRIPTION.
    traj_df_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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

    return None     
    
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

# !!! jaccard similarity does not require equivalent time between locations
def get_jaccard(loc_id_list, lag_list):
    
    n = len(loc_id_list)    
    def jaccard(lag):
        print('\n===== Lag = {} ====='.format(lag))
        #define Jaccard Similarity function
        intersec = len(list(set(loc_id_list[:n - lag]).intersection(loc_id_list[lag:])))
        union = (len(loc_id_list[:n - lag]) + len(loc_id_list[lag:])) - intersec
        jaccard = float(intersec) / union
        return jaccard
    
    jaccard_list = list(map(jaccard, lag_list))
    
    return jaccard_list
 
   
def plot_traj_in_date_interval(traj_df_1, traj_df_2, date_interval_lb=24*30*11.5, date_interval_ub=24*30*12):  
    
    traj_in_date_interval_1 = traj_df_1.loc[traj_df_1['time_diff'].between(date_interval_lb, date_interval_ub)]
    traj_in_date_interval_2 = traj_df_2.loc[traj_df_2['time_diff'].between(date_interval_lb, date_interval_ub)]
    
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(traj_in_date_interval_1['time_diff'], traj_in_date_interval_1['latitude'],
               color='red')
    ax.scatter(traj_in_date_interval_2['time_diff'], traj_in_date_interval_2['latitude'],
                color='blue')

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
    ax.legend()
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
    ax.legend()
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
    os.chdir('C:/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src')
    
    # syn_traj_wt_travel_sleep.to_csv('syn_traj_wt_travel_sleep_one_year.csv')
    # syn_traj_wo_travel_sleep.to_csv('syn_traj_wo_travel_sleep_one_year.csv')
    
    # import data
    syn_traj_wt_travel_sleep = pd.read_csv('../outputs/simulated_mobility_trajectories/syn_traj_wt_travel_sleep_one_year.csv')
    syn_traj_wo_travel_sleep = pd.read_csv('../outputs/simulated_mobility_trajectories/syn_traj_wo_travel_sleep_one_year.csv')      
    
    # format figures
    set_default_plot_param()
    
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
    
    
    # plot locations over time after a long enough time, say the last two weeks in the 12-th month
    
    

# ######
# main()