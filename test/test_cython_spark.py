def test_cython(n):

    import findspark; findspark.init()
    
    import pyspark
    
    import pyximport; pyximport.install()
    
    import factorial
    
    
    # Spark Context
    from pyspark import SparkContext
    sc = SparkContext()
    
    nums= sc.parallelize(list(range(10, n))) 
    
    squared = nums.map(factorial.cal_factorial).collect()
    # for num in squared:
    #     print('  %i ' % (num))


def test_math(n):

    import findspark; findspark.init()
    
    import pyspark
        
    import math
    
    
    # Spark Context
    from pyspark import SparkContext
    sc = SparkContext()
    
    nums= sc.parallelize(list(range(10, n))) 
    
    squared = nums.map(math.factorial).collect()
    # for num in squared:
    #     print('  %i ' % (num))

##################
if __name__ == "__main__":

    import time
   
   	# import os
   	# os.chdir('/mnt/c/Users/Jinzh/OneDrive/GitHub/Scale_Mobility/test')
   
    n = 20
   	 
    t0 = time.time() 
    test_cython(n)
    t1 = time.time()
       
    test_math(n)
    t2 = time.time()
       
    print('Cython time: ', t1-t0)
    print('Math factorial time: ', t2 - t1)
