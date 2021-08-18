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
    
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams['font.weight']= 'normal'
    plt.rcParams['figure.figsize'] = [6, 6*3/4]
   
    plt.rcParams['figure.facecolor'] = 'white'

    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.axisbelow'] = True 
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['axes.labelpad'] = 3
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
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(traj_df_1['datetime'], traj_df_1['num_uniq_loc_visited'],
               color='red')
    ax.scatter(traj_df_2['datetime'], traj_df_2['num_uniq_loc_visited'],
               color='blue')

    ax.set(xlabel="Date",
           ylabel="# of unique locations visited",
           title="Propensity for exploration over time")
    plt.xticks(rotation=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylim(bottom=-10)
    
    ax.legend(['With travel and sleep time','without travel and sleep time'])

    plt.show()   
    
plot_explore_propensity(synthetic_trajectories_with_travel_time, synthetic_trajectories)
# plot_explore_propensity(synthetic_trajectories)
