import numpy as np 
import matplotlib as plt
import pandas as pd

def format_fig():

    from matplotlib import pyplot as plt
    
    plt.style.use('classic')
    
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams['font.weight']= 'normal'
    plt.rcParams['figure.figsize'] = [6, 6*3/4]
   
    plt.rcParams['figure.facecolor'] = 'white'

    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.axisbelow'] = True 
    plt.rcParams['axes.titlepad'] = 15  # title to figure
    plt.rcParams['axes.labelpad'] = 2 # x y labels to figure
    plt.rc('axes', titlesize=15, labelsize=14, linewidth=1.1)    # fontsize of the axes title, the x and y labels
    
    
    plt.rcParams['ytick.right'] = False
    plt.rcParams['xtick.top'] = False
    # plt.rcParams['xtick.minor.visible'] = True
    # plt.rcParams['ytick.minor.visible'] = True

    # plt.rc('lines', linewidth=1.8, markersize=6, markeredgecolor='none')
    
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    

    # plt.rcParams['xtick.major.size'] = 5
    # plt.rcParams['ytick.major.size'] = 5

    plt.rcParams['axes.formatter.useoffset'] = False # turn off offset
    # To turn off scientific notation, use: ax.ticklabel_format(style='plain') or
    # plt.ticklabel_format(style='plain')

    
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.framealpha"] = 0.5

    
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 400
    
#    plt.rc('text', usetex=False)




def get_exe_time(func):
    
    import time
    
    start_time = time.time()
    
    func
    
    end_time = time.time()
    
    time_diff = end_time-start_time
    
    print( "\n ===== Time for executing the {} function: {} seconds =====".format( func, round(time_diff,3) ) )

    return None



def get_code_profile(func):
    
    import cProfile, pstats
    
    profiler = cProfile.Profile()
    profiler.enable()

    func

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    
    print("\n\n\n")
    print("\n ===== Profile of the code for the {} function=====".format(func))
    stats.print_stats()
    


def append_dfs(df1, df2):
    import pyspark.sql.functions as F    
    list1 = df1.columns
    list2 = df2.columns
    for col_temp in list2:
        if(col_temp not in list1):
            df1 = df1.withColumn(col_temp, F.lit(None))
    for col_temp in list1:
        if(col_temp not in list2):
            df2 = df2.withColumn(col_temp, F.lit(None))
    return df1.unionByName(df2)



def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the earth distance and bearing in degree between two points 
    on the earth (specified in decimal degrees)
    """
    #check input type
    if (not isinstance(lon1, list)) and (not isinstance(lon1, pd.Series)):
        raise TypeError("Only list or pandas series are supported as input coordinates")
    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
               np.multiply(np.cos(lat1), 
                           np.multiply(np.cos(lat2), 
                                       np.power(np.sin(np.divide(dlon, 2)), 2))))
    c = np.multiply(2, np.arcsin(np.sqrt(a)))
    r = 6371*1e3  # global average radius of earth in m. Use 3956 for miles.
    dist = c*r
    
    bearing = np.arctan2(np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon), np.sin(dlon)*np.cos(lat2)) 
    bearing = np.degrees(bearing)
    bearing_deg = (bearing + 360) % 360
    return np.round(dist, 4), np.round(bearing_deg, 4)