# TASK 3: Parallel Implementation

# imports
from functions_perflab import *


if __name__ == '__main__':

    # values for a good desktop, for a potato machine maybe try reducing multiplications and/or n
    multiplications = 100
    n = 50
    matmul = matmul1
    verbose = False

    # reduces chance of strange error when using starmap


    # time the serial run of the matrix multiplication
    time_serial = time_serial_multiply_square_matrices(multiplications, n, matmul, verbose=True)

    # time parallel runs of the matrix multiplication, up to max number of CPUs
    # TODO - if this is struggling to work, try using command within function that uses limit_cpu
    ncpu, time_parallel = time_parallel_multiply_square_matrices(multiplications, n, matmul, cpu_cap=None, verbose=False)
    plot_runtime_ncpu(time_parallel, ncpu)

