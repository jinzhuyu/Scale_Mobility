# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:20:17 2021

@author: Jinzh
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


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
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['axes.labelpad'] = 8
    plt.rc('axes', titlesize=16, labelsize=15, linewidth=0.9)    # fontsize of the axes title, the x and y labels
    
    # plt.rc('lines', linewidth=1.8, markersize=6, markeredgecolor='none')
    
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)

    plt.rcParams['axes.formatter.useoffset'] = False # turn off offset
    # To turn off scientific notation, use: ax.ticklabel_format(style='plain') or
    # plt.ticklabel_format(style='plain')

    
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.framealpha"] = 0.5

    
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 800
    
#    plt.rc('text', usetex=False)
set_default_plot_param()



def plot_explore_propensity(traj_df_1, traj_df_2):
    '''
    plot the propensity for exploration over time
    the progression of number of unique locations visited over time

    Parameters
    ----------
    traj_df : pandas df.
        tranjectories of users, columns: user_id, lat(location_id),
        long (location_id), time, num_uniq_loc_visited

    Returns
    -------
    None.

    '''
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
    ax.semilogy()
    # ax.loglog()
    ax.set_ylim(top=10**0)
    ax.set_xlim(left=1)
    
    ax.legend(['With travel and sleep time','without travel and sleep time'], loc='upper right')

    plt.show()       
    
    
    

def get_time_diff(df):
    '''
    add another column to df, 'time_diff', contains the difference between the column 'datetime' to the initial time point

    '''
   
    traj_datetime = pd.to_datetime(df['datetime'])
    seconds_in_hour = 3600
    time_diff_timestamp = (traj_datetime - traj_datetime[0]).astype('timedelta64[s]') / seconds_in_hour

    df['time_diff'] = time_diff_timestamp
    
    return df



def main():
    
    import os
    os.chdir('C:/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/src')
    
    # import data
    syn_traj_wt_travel_sleep = pd.read_csv('../outputs/simulated_mobility_trajectories/syn_traj_wt_travel_sleep.csv')
    syn_traj_wo_travel_sleep = pd.read_csv('../outputs/simulated_mobility_trajectories/syn_traj_wo_travel_sleep.csv')
       
    
    # format figures
    set_default_plot_param()
    
    
    
    # plot exploration propensity over time for all users
    # get time difference
    syn_traj_wt_travel_sleep = get_time_diff(syn_traj_wt_travel_sleep)
    syn_traj_wo_travel_sleep = get_time_diff(syn_traj_wo_travel_sleep) 
    plot_explore_propensity(syn_traj_wt_travel_sleep, syn_traj_wo_travel_sleep)

    # plot exploration propensity over time for 10 user only
    plot_explore_propensity(syn_traj_wt_travel_sleep[syn_traj_wt_travel_sleep['user_id'] == 5],
                            syn_traj_wo_travel_sleep[syn_traj_wo_travel_sleep['user_id'] == 5])
    
   
    
    # plot f_k (visitation frequency) vs k (k-th most visited location of a user for different S values, 20, 40, and 60)
    visit_freq_df_wt_travel_sleep = get_user_visit_freq(syn_traj_wt_travel_sleep)
    visit_freq_df_wo_travel_sleep = get_user_visit_freq(syn_traj_wo_travel_sleep)
    
    
    plot_visit_freq(visit_freq_df_wt_travel_sleep[visit_freq_df_wt_travel_sleep['user_id'] == 7],
                    visit_freq_df_wo_travel_sleep[visit_freq_df_wo_travel_sleep['user_id'] == 7])


    plot_visit_freq(visit_freq_df_wt_travel_sleep, visit_freq_df_wo_travel_sleep)

######
main()