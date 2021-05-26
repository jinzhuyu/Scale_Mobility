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
    


import pyspark.sql.functions as F
def append_dfs(df1, df2):
    
    list1 = df1.columns
    list2 = df2.columns
    for col_temp in list2:
        if(col_temp not in list1):
            df1 = df1.withColumn(col_temp, F.lit(None))
    for col_temp in list1:
        if(col_temp not in list2):
            df2 = df2.withColumn(col_temp, F.lit(None))
    return df1.unionByName(df2)