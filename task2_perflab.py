# TASK 2: Optimising

# imports
import numpy as np

from functions_perflab import *

#TODO - your code here

if __name__ == "__main__":
    n=[2**3,2**4,2**5,2**6,2**7] #record problem size
    time1=profile_matmul(10, n, np.matmul) #record runtime for the three functions
    time2=profile_matmul(10,n,matmul1)
    time3 = profile_matmul(10, n, matmul2)
    plot_polynomial_performance(time1, n) #plot the log runtime versus log number size of the three functions independently
    plot_polynomial_performance(time2,n)
    plot_polynomial_performance(time3, n)
    pr = cProfile.Profile() #unrelevant test
    pr.enable()
    profile_matmul(10, n, np.matmul)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats()
    print(ps)



