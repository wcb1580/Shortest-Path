# ENGSCI233: Lab - Performance

# imports
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
import itertools
import time
import psutil
import os
import platform
import cProfile
import pstats
import io
import time


def multiply_square_matrices(multiplications, n, matmul):
    """
    Performs matrix multiplication for many square matrices.
    
    Parameters
    ----------
    multiplications : int
        Number of matrix multiplications to perform.
    n : int
		Number of rows and columns in square matrices being multiplied.
	matmul : callable
        Function to use to perform matrix multiplication
    """
    # iteratively multiply square matrices of random numbers, then return
    for i in range(multiplications):
        a = square_matrix_rand(n)
        b = square_matrix_rand(n)
        _ = matmul(a, b)
    return


def square_matrix_rand(n):
    """
    Create a square matrix of random values between -1 and 1.

    Parameters
    ----------
    n : int
        Size of the square matrix (n x n).

    Returns
    -------
    matrix : numpy array
        Square matrix containing uniformly distributed random numbers between -1 and 1.
    """
    matrix = np.random.rand(n, n) * 2. - 1.
    return matrix


def matmul1(a, b):
    """
    Multiply two matrices using "naive" method.

        Parameters
        ----------
        a : numpy array
            Left multiplying matrix i.e. A.
        b : numpy array
            Right multiplying matrix i.e. B.

        Returns
        -------
        c : numpy array
            Matrix product i.e. AB = C.

        Raises
        ------
        ValueError
            If inner dimensions of matrix product are inconsistent
    """
    # check dimension consistency precondition
    if a.shape[1] != b.shape[0]:
        raise ValueError('Dimension inconsistency: A must have the same number of columns as B has rows.')

    # compute matrix product as dot products of rows and columns
    product = np.zeros((a.shape[0], b.shape[1]))
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            for k in range(a.shape[1]):
                product[i, j] += a[i, k] * b[k, j]
    return product


# TODO - complete in Task 2
def matmul2(a, b):
    """
    Multiply two matrices together, making use of built-in NumPy dot. Utilises NumPy dot product to replace
    third and final nested for loop of naive implementation.

        Parameters
        ----------
        a : numpy array
            Left multiplying matrix i.e. A.
        b : numpy array
            Right multiplying matrix i.e. B.

        Returns
        -------
        c : numpy array
            Matrix product i.e. AB = C.

        Raises
        ------
        ValueError
            If inner dimensions of matrix product are inconsistent
    """
    if a.shape[1] != b.shape[0]:
        raise ValueError('Dimension inconsistency: A must have the same number of columns as B has rows.')
    product = np.zeros((a.shape[0], b.shape[1]))
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            product[i, j] += np.dot(a[i, :] , b[:, j]) # record the result matrix using np.dot function
    return product





# TODO - complete in Task 1
def profile_matmul(multiplications, n, matmul):
    """
    The function profile_matmul() plots a list of runtime as a result for a list of question size in the input
    Inputs
    ------
    multiplications: Integer
        Number of time to perform matrix multiplications
    n: list
        A list of size of square matrix
    matmul:callable
        A function to perform matrix multiplication
    Returns
    ------
    return_list:list
        A list contains the runtime relates to different problem sizes as in the input

    """
    return_list=[]
    for i in range (len(n)):#loop through the problem size list independently
        pr = cProfile.Profile() #enable cProfile
        pr.enable()
        multiply_square_matrices(multiplications, n[i], matmul)# perform the operation
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime') #record the runtime of the function
        return_list.append(ps.total_tt) #append the runtime attribute in cProfile
    return return_list





# TODO - complete in Task 1
def plot_polynomial_performance(times, n):
    """
        The function plot_polynomial_performance() takes two inputs, a list of runtime and a list of relevant problem size to the runtime,
            and it plots a graph of data points from the inputs, and demonstrate a best fit line for these data points
        Inputs
        ------
        times:list
            A list of runtimes with different problem size
        n:list
            A list of different problem size
        Returns
        -----
        A graph shows the data points of log(runtimes) in respective to log(problem size) with a besst fit line


    """
    # create the empty figure
    f, ax = plt.subplots(1, 1)
    ax.set_xlabel("Problem Size in logarithm")
    ax.set_ylabel("Runtime in logarithm")
    # show the plot
    ax.plot(np.log(n),np.log(times),'bx',label="log times against log problem size")
    x=np.log(n)
    y=np.log(times)
    z=np.polyfit(x,y,1) # record the fitted coefficient for the best fit line as (gradient,intercept)
    ax.plot(x,x*z[0]+z[1],'r-',label="fitted polynomial") # use the gradient in the record of np.polyfit() to plot a best fit line
    value=z[0]
    ax.plot([],[],'',label='a={}'.format(value)) #show the a value for Bigi O notation O(n^a)
    ax.legend()
    plt.show()


# TODO - document in Task 3
def time_serial_multiply_square_matrices(multiplications, n, matmul, verbose=False):
    """The function time_serial_multiply_square_matrices calculates the run time using a single CPU for function multiply_square_matrices, and print the time based on the input boolean,and returns the time as result .
        Inputs
        ------
        multiplications: integer
            Number of times to perform matrix multiplication
        n: integer
            Number of rows and columns in square matrices being multiplied.
        matmul:callable
            A function used for matrix multiplication
        verbose:Boolean
            A boolean to determine whether the runtime should be printed(True) or not
        Returns
        ------
        time_taken: float
            A float shows amount of time required to perform multiply_square_matrices using a single CPU.
    """
    tic = time.perf_counter() #initial time before running the function
    multiply_square_matrices(multiplications, n, matmul)
    toc = time.perf_counter() #afterward time after running the function
    time_taken = toc - tic #record the runtime of serial running
    if verbose:
        print('time serial: ', time_taken)
    return time_taken


# TODO - document in Task 3
def time_parallel_multiply_square_matrices(multiplications, order, matmul, cpu_cap=None, verbose=False):
    """The function time_parallel_multiply_square_matrices record the runtimes for different number of CPUs to perform parallel running to the function multi_square_matrices,
        and returns the relevant runtime with its relevant CPU number in two lists.
        Input
        ------
        multiplications: integer
            A number indicates amount of times of matrix multiplication
        order: int
            The columns/rows of the square matrix used in multiply_square_matrices
        matmul:callable
            A function that is required as one of the input argument of multi_square_matrices
        cpu_cap:None or integer
            An argument states the maximum number of CPU used for parallel running, if is set to None, the function would use all the avaliable CPUs in the machine
        verbose:Boolean
            A boolean that controls displaying runtime and CPU number to the screen
        Returns
        ------
        ncpu_used:list
            States the number of CPU used in each parallel run
        time_parallel:list
            Record the runtime for each of the parallel run"""
    time_parallel = []
    ncpu_used = []

    if cpu_cap is None: # if no maximum number of cpu is given, use all the avaliable cpu in the machine
        max_ncpu = os.cpu_count()
    else:
        max_ncpu = cpu_cap+1 #use the given cpu_cap as the maximum cpu used in parallel running

    for ncpu in range(2, max_ncpu):  #loop through to used different number of cpu in parallel running in an accending order

        multiplications_per_cpu = [int(multiplications / ncpu)] * ncpu
        i = 0
        while sum(multiplications_per_cpu) < multiplications: #obtain the evaluated attribute(number of times of matrix multiplication) for the parallel running.
            multiplications_per_cpu[i] += 1
            i += 1
            if i >= len(multiplications_per_cpu):
                i = 0

        # TODO - if this is struggling to work, try using command that uses limit_cpu
        with multiprocessing.Pool(ncpu) as pool:
        #with multiprocessing.Pool(ncpu, limit_cpu) as pool:
            tic = time.perf_counter()
            pool.starmap(multiply_square_matrices, zip(multiplications_per_cpu, itertools.repeat(order),
                                                       itertools.repeat(matmul))) #process the function multiply_square_matrices using in parallel running with different numebr of cpu

            toc = time.perf_counter()

        time_parallel.append(toc - tic) #record the runtime of parallel running with different number of cpu
        ncpu_used.append(ncpu) #record the number of used cpu in the return list

        if verbose:
            print('time parallel with ', ncpu, 'CPU: ', toc - tic)

    return ncpu_used, time_parallel


def limit_cpu():
    """
    Called at every process start. For Windows will attempt to reduce CPU priority and reduce chance of computer
    freezing due to full CPU utilisation.
    """
    if platform.system() == 'Windows':
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


# TODO - complete in Task 3
def plot_runtime_ncpu(runtime, ncpu):
    """
        The function plot_runtime_ncpu plots takes a list of runtimes and a list of cpu number relevant to the runtimes,
        and plots two graphs for parallel speedup and parallel efficiency in a single figure
        Inputs
        ------
        runtime: list
            A list of runtime
        ncpu:
            A list of number of CPU used in parallel running relevants to the runtime list
        Returns
        -----
            Two graphs, one is parallel speedup versus number of CPU, and the other one is parallel efficiency versus number of CPU
            using the relevant functions.
    """
    multiplications=100 #set the number of times of matrix multiplications
    n=50 #set the size of the matrix
    Sp=[]
    Ep=[]
    serial=time_serial_multiply_square_matrices(multiplications, n, matmul1, verbose=False) #record the runtime in one cpu(serial running)
    # create the empty figure with two subplots
    f, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(len(runtime)):#loop through the runtime list
        Sp.append(serial/runtime[i]) #record the parallel speedup in ia new list
        Ep.append(serial/(runtime[i]*ncpu[i])) #record the parallel efficiency in a new list
    #plot the two graphs in a single figure
    ax1.plot(ncpu,Sp,'r*',label='Parallel Speedup')
    ax1.set_xlabel("cpus")
    ax1.set_ylabel("Parallel Speedup")
    ax2.plot(ncpu,Ep,'bo',label='Parallel Efficiency')
    ax2.set_xlabel("cpus")
    ax2.set_ylabel("Parallel Efficiency")


    # show the plot
    plt.show()
