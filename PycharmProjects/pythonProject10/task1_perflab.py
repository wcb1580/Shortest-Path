# TASK 1: Profiling

# imports
from functions_perflab import *


if __name__ == "__main__":

    # create a profiling object
    pr = cProfile.Profile()

    # enable the profiler, run the thing to profile, and then disable the profiler
    pr.enable()
    multiply_square_matrices(10, 50, matmul1)
    pr.disable()

    # get and print useful stats from the profiler
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats()

    # print total time to screen
    print('Total time taken:', ps.total_tt)
