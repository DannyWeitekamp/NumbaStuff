import numpy as np
import numba
from numba import types, njit,guvectorize,uint32,vectorize,prange
from numba.typed import List
import numba.unicode
import time
from timeit import repeat
# import hpat

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


# using HPAT since it supports time.time()
# @vectorize
# @guvectorize([(uint32[:,:], uint32[:,:])], '(n,d)->(n,d)',nopython=True)
@njit('uint32[:,:](uint32[:,:])', nopython=True, nogil=True, parallel=True)
def upper(str_list):
	n,d = str_list.shape
	out = np.empty((n,d),dtype=np.uint32)	
	for i in prange(n):
		for j in range(d):
			if(str_list[i][j] >= 97 and str_list[i][j] <= 122):
				out[i][j] = str_list[i][j] - 32
			else:
				out[i][j] = str_list[i][j]
	return out

# @guvectorize([(uint32[:,:], uint32[:,:])], '(n,d)->(n,d)',nopython=True)
@njit('uint32[:,:](uint32[:,:])', nopython=True, nogil=True, parallel=True)
def lower(str_list):
	n,d = str_list.shape
	out = np.empty((n,d),dtype=np.uint32)	
	for i in prange(n):
		for j in range(d):
			if(str_list[i][j] >= 65 and str_list[i][j] <= 90):
				out[i][j] = str_list[i][j] + 32
			else:
				out[i][j] = str_list[i][j]
	return out

# @vectorize(nopython=True)
# @guvectorize([(uint32[:,:], uint32[:])], '(n,d)->(n)', nopython=True)
# def str_len(str_list,out):
@njit('uint32[:](uint32[:,:])', nopython=True, nogil=True, parallel=True)
def str_len(str_list):
	n,d = str_list.shape
	out = np.empty((n,),dtype=np.uint32)	
	for i in prange(n):
		out[i] = d
		for j in range(d):
			if(str_list[i][j] == 0):
				out[i] = j
				break
	return out

@njit('int32(int32,int32,uint32[:,:],uint32[:,:])', nopython=True, nogil=True)
def _assign1_fixed(i,d1,out,str_list_a):
	for j in range(d1):
		if(str_list_a[0][j] != 0):
			out[i][j] = str_list_a[0][j]
		else:
			j -= 1
			break
	return j

@njit('int32(int32,int32,uint32[:,:],uint32[:,:])', nopython=True, nogil=True)
def _assign1_dynam(i,d1,out,str_list_a):
	for j in range(d1):
		if(str_list_a[i][j] != 0):
			out[i][j] = str_list_a[i][j]
		else:
			j -= 1
			break
	return j

@njit('int32(int32,int32,int32,uint32[:,:],uint32[:,:])', nopython=True, nogil=True)
def _assign2_fixed(k,i,d2,out,str_list_b):
	for j in range(d2):
		k += 1
		if(str_list_b[0][j] != 0):
			out[i][k] = str_list_b[0][j]
		else:
			break
	return k

@njit('int32(int32,int32,int32,uint32[:,:],uint32[:,:])', nopython=True, nogil=True)
def _assign2_dynam(k,i,d2,out,str_list_b):
	for j in range(d2):
		k += 1
		if(str_list_b[i][j] != 0):
			out[i][k] = str_list_b[i][j]
		else:
			break
	return k

# @guvectorize([(uint32[:,:], uint32[:,:],uint32[:,:])], '(n,d1),(n,d2)->(n,d1+d2)', nopython=True)
@njit('uint32[:,:](uint32[:,:],uint32[:,:])', nopython=True, nogil=True, parallel=True)
def concatenate(str_list_a,str_list_b):
	n1,d1 = str_list_a.shape
	n2,d2 = str_list_b.shape
	n = max(n1,n2)
	out = np.empty((n,d1+d2),dtype=np.uint32)

	if(n1 != 1 and n2 != 1):
		for i in prange(n):
			k = _assign1_dynam(i,d1,out,str_list_a)
			k = _assign2_dynam(k,i,d2,out,str_list_b)
			for k in range(k,d1+d2): out[i][k] = 0
	elif(n1 == 1 and n2 != 1):
		for i in prange(n):
			k = _assign1_fixed(i,d1,out,str_list_a)
			k = _assign2_dynam(k,i,d2,out,str_list_b)
			for k in range(k,d1+d2): out[i][k] = 0
	elif(n1 != 1 and n2 == 1):
		for i in prange(n):
			k = _assign1_dynam(i,d1,out,str_list_a)
			k = _assign2_fixed(k,i,d2,out,str_list_b)
			for k in range(k,d1+d2): out[i][k] = 0
	else:
		for i in prange(n):
			k = _assign1_fixed(i,d1,out,str_list_a)
			k = _assign2_fixed(k,i,d2,out,str_list_b)
			for k in range(k,d1+d2): out[i][k] = 0
	return out
		# for i in range(n):

			# _ia = 0 if n1 == 1 else i
			# _ib = 0 if n2 == 1 else i
			# _ia = i
			# _ib = i
			# for j in range(d1):
			# 	if(str_list_a[_ia][j] != 0):
			# 		out[i][j] = str_list_a[_ia][j]
			# 	else:
			# 		j -= 1
			# 		break
			# k = j 
			# k = _assign1_dynam(int(i),int(d1),out,str_list_a)
			# for j in range(d2):
			# 	k += 1
			# 	if(str_list_b[_ib][j] != 0):
			# 		out[i][k] = str_list_b[_ib][j]
			# 	else:
			# 		break
			# for k in range(k,d1+d2):
			# 	out[i][k] = 0

				
# timefunc(None,"str_len",str_len,s_list)
# timefunc(None,"concatenate",concatenate
			
		
	

def decode_str(s):
	if(not isinstance(s,(list,tuple))): s = [s]
	l = len(s)
	s = np.array(s, dtype=str)
	return s.view(np.uint32).reshape(l,-1)

def encode_str(s):
	n,d = s.shape
	return s.view("<U%i" % d)

# @njit( nopython=True, nogil=True, parallel=True)
# def concatenate_right(str_list_a,str_b):
# 	# str_list_b = str_b.decode("utf-8")
# 	n,d1 = str_list_a.shape
# 	# n,d2 = str_list_b.shape
# 	d2 = len(str_b)
# 	# str_list_b = np.array(str_b, dtype=np.uint8).reshape(1,d2)
# 	out = np.empty((n,d1+len(str_b)),dtype=np.uint32)

# 	#Copy Paste from str_len() because of issue with nesting...
# 	for i in range(n):
# 		# l_a[i] = d
# 		for j in range(d1):
# 			if(str_list_a[i][j] != 0):
# 				out[i][j] = str_list_a[i][j]
# 			else:
# 				j -= 1
# 				break
# 		k = j 
# 		for j in range(len(str_b)+1):
# 			k += 1
# 			if(j < d2):
# 				out[i][k] = str_b[j]
# 			else:
# 				break
# 		for k in range(k,d1+d2):
# 			out[i][k] = 0
# 	return out



@njit('uint32[:](uint32[:,:],uint32[:,:])', nopython=True, nogil=True, parallel=True)
# @guvectorize([(uint32[:,:], uint32[:])], '(n,d)->(n)', nopython=True)
def test_curry(str_list_a,str_list_b):
	return str_len(concatenate(str_list_a,str_list_b))




# def concatenate()				

	# return returnnp.char.capitalize(s_list)
    # out = []
    # cmp_str = str_list[0]
    # t1 = time.time()
    # for i in range(len(str_list)):
    #     out.append(str_list[i] == "blarg")
    # t2 = time.time()
    # print("exec time", t2-t1)
    # return out
d = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ '])
d2 = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ '])
np.char.add(d,d2)
n = int(1e6)

s_list = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ '], dtype=str)
# print(s_list.dtype)
s_list = s_list.view(np.uint32).reshape(3,-1)

print(encode_str(upper(s_list)))
print(encode_str(lower(s_list)))
# print(encode_str(concatenate(s_list,s_list)))
# print(s_list.shape,con.shape)
print("V")
print(decode_str(" fooperz").shape)
print(decode_str([" A23"," B23"," C23"]).shape)
print("&")
print(encode_str(concatenate(s_list,decode_str([" A23"," B23"," C23"]))))
print(encode_str(concatenate(s_list,decode_str(" fooperz"))))
print(encode_str(concatenate(decode_str("fooperz "),s_list)))
print(test_curry(decode_str("fooperz "),s_list))
# print(concatenate_right(s_list,"moop").view("U1"))
# s_list = List()
# for i in range(n):
    # s_list.append(str(np.random.ranf()))
s_list = np.array(['abc','snake','plateblxyz']*n, dtype=str)
# print(s_list.dtype)
s_list = s_list.view(np.uint32).reshape(3*n,-1)



timefunc(None,"upper",upper,s_list)
timefunc(None,"str_len",str_len,s_list)
timefunc(None,"concatenate",concatenate,s_list,s_list)
rand = (np.random.random((3,n))*1000).view(np.uint32).reshape(3,-1)
timefunc(None,"concatenate-random",concatenate,rand,rand)
timefunc(None,"concatenate-left",concatenate,decode_str("fooperz "),s_list)
timefunc(None,"concatenate-right",concatenate,s_list,decode_str("fooperz "))
timefunc(None,"test_curry",test_curry,s_list,decode_str("fooperz "))

d = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ ']*n)
# timefunc(None,"concatenate-numpy",np.char.add,d,d)
