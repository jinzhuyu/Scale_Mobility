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