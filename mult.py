#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import math
import threading
from timeit import repeat

import numpy as np
from numba import jit, prange, njit,vectorize
from numba.typed import List

nthreads = 2
size = 10**6

@vectorize('float64(float64, float64)')
def func_np_v(a, b):
    """
    Control function using Numpy.
    """
    return np.exp(2.1 * a + 3.2 * b)

@njit('double[:](double[:], double[:])', nopython=True, nogil=True, parallel=True)
def func_np(a,b):
    return func_np_v(a,b)

@njit('void(double[:], double[:], double[:])', nopython=True, nogil=True, parallel=True)
def inner_func_nb(result, a, b):
    """
    Function under test.
    """
    for i in prange(len(result)):
        result[i] = func_np_v(a[i],b[i])#np.exp(2.1 * a[i] + 3.2 * b[i])

def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before we start the benchmark
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                          number=5, repeat=2)) * 1000))
    return res

def make_singlethread(inner_func):
    """
    Run the given function inside a single thread.
    """
    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result
    return func

def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args]
                  for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt

# @vectorize('str(str)')
@njit(nopython=True)
def concat_blarg(inp):
    # return inp + "blarg"
    for i in range(len(inp)):
        inp[i] += "blarg"
    return inp


# print(concat_blarg(np.array(['USA', 'Japan', 'UK', '', 'India', 'China'])))

# inp = ['USA', 'Japan', 'UK', '', 'India', 'China']*1000
# timefunc(None,"blarg",concat_blarg,inp)

func_nb = make_singlethread(inner_func_nb)
func_nb_mt = make_multithread(inner_func_nb, nthreads)

a = np.random.rand(size)
b = np.random.rand(size)

correct = timefunc(None, "numpy (1 thread)", func_np, a, b)
timefunc(correct, "numba (1 thread)", func_nb, a, b)
timefunc(correct, "numba (%d threads)" % nthreads, func_nb_mt, a, b)