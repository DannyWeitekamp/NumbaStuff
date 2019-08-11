import numpy as np
import numba
from numba import types, njit,guvectorize,uint32,vectorize,prange
from numba.typed import List,Dict
import numba.unicode
import time
from timeit import repeat
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# import hpat

def timefunc(s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # # Make sure the function is compiled before we start the benchmark
    res = func(*args, **kwargs)
    # if correct is not None:
    #     assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                          number=10000, repeat=2)) * 1000))
    return res


# float_array = types
@njit(nogil=True, parallel=False,fastmath=True,cache=True)
def dict_stuff():
    d = dict()
    # l = List()
    # l.append(1)
    for i in range(100):
        j = i // 10
        # d[j] 
        if(not j in d):
            l = List()
            l.append(np.array([i],dtype=np.uint32))
            d[j] = l
        else:
            d[j].append(inp.array([i],dtype=np.uint32))
        # l = d.get(j,List(types.uint32))
        # l.append(i)
        # d[j] = l
        # print(j)
    # for j in list(d)
    return d

# @njit
# def foo():
#     # Make dictionary
#     d = Dict.empty(
#         key_type=types.unicode_type,
#         value_type=float_array,
#     )
#     # Fill the dictionary
#     d["posx"] = np.arange(3).astype(np.float64)
#     d["posy"] = np.arange(3, 6).astype(np.float64)
#     return d
# print(foo())

print(dict_stuff())
