import numpy as np
import numba
from numba import types, njit,guvectorize,uint32,vectorize,prange
from numba.typed import List
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

@njit(nogil=True, parallel=False,fastmath=True)
def first_nonzero(partial_match,start=0):
    for k in range(start,len(partial_match)):
        if(partial_match[k] != 0):
            return k,partial_match[k]

def gen_partial_matches(partial_match,pair_matches,pair_index_reg):
    # print("IN ",partial_match)
    d = len(partial_match)
    new_partial_matches = List()
    full = True
    for i in range(d):
        v = partial_match[i]
        if(v == 0):
            full = False
            break

    if(full):
        # print("FULLL", partial_match)
        new_partial_matches.append(partial_match)
        return new_partial_matches

    
    
    for i in range(d):
        v = partial_match[i]
        if(v != 0):
            for j in range(d):
                # print(i,j)
                if(partial_match[j] == 0 and pair_index_reg[i][j][0] != -1):
                    s,l = pair_index_reg[i][j][0],pair_index_reg[i][j][1]
                    for k in range(s,s+l):
                        print(pair_matches[k])
                        new_partial_match = np.copy(partial_match)
                        if(pair_matches[k][0] == j and pair_matches[k][1] != i):
                            v_k = pair_matches[k][2]
                        # elif(pair_matches[k][1] == j):
                        elif(pair_matches[k][0] != i):
                            v_k = pair_matches[k][3]
                        else:
                            continue

                        ok = True
                        for _v in partial_match:
                            if(_v == v_k):
                                ok = False

                        if(ok):
                            new_partial_match[j] = v_k
                            new_partial_matches.append(new_partial_match)
                        # print(v,v_k)
    # print(new_partial_matches)
    final_partial_matches = List()
    if(len(new_partial_matches) > 0):
        for z in new_partial_matches:
            for pm in gen_partial_matches(z,pair_matches,pair_index_reg):
                final_partial_matches.append(pm)
    else:
        final_partial_matches.append(partial_match)
        # return 
    # print("OUT", final_partial_matches)
    return final_partial_matches



        

@njit(nogil=True, parallel=False,fastmath=True,cache=True)
def compute_adjacencies(split_ps, concept_slices,
                    where_part_vals):
    d = len(concept_slices)-1
    adjacencies = np.zeros((d,d),dtype=np.uint8)

    for i in prange(d):
        concept = split_ps[concept_slices[i]:concept_slices[i+1]]
        # print("concept", concept,concept_slices[i],concept_slices[i+1])
        for j in range(d):
            for k in range(len(concept)):
                if(concept[k] == where_part_vals[j]):
                    adjacencies[i,j] = 1    
                    break
                
    return adjacencies

@njit(nogil=True, parallel=False,fastmath=True)
def gen_cand_slices(inds_j,
                    elems,elems_slices,
                    sel):
    candidate_slices = np.empty((len(inds_j),len(sel)),dtype=np.uint8)
    # print("BOOP",len(inds_j))
    for k in prange(len(inds_j)):
        ind = inds_j[k]
        # print(elems[elems_slices[ind]:elems_slices[ind+1]])
        candidate_slices[k] = elems[elems_slices[ind]:elems_slices[ind+1]][sel]
        # cands = 
    return candidate_slices


@njit(nogil=True, parallel=False,fastmath=True,cache=True)
def fill_partial_matches_at(partial_matches,i,pair_matches,pair_index_reg):
    # print("i",i)
    d = partial_matches.shape[1]
    i_elms = dict()


    for j in range(d):
        if(pair_index_reg[i][j][0] != -1):
            s,l = pair_index_reg[i][j][0],pair_index_reg[i][j][1]
            for k in range(l):
                pm = pair_matches[s+k]
                dat = np.empty(3,dtype=np.uint32)
                dat[0] = s
                dat[1] = l
                dat[2] = k

                if(not pm[2] in i_elms):
                    lst = List()
                    lst.append(dat)
                    i_elms[pm[2]] = lst
                else:
                    i_elms[pm[2]].append(dat)
            # if(l > 1):
            #     # old = len(partial_matches)
            #     # new_partial_matches = np.empty((old*l,d))
                

            #         # i_elms.add(pm[2])
            #         # print(pm,"k")
            #         # partial_matches[:,pm[0]] = pm[2]
            #         # partial_matches[:,pm[1]] = pm[3]
            #         # new_partial_matches[k*old:(k+1)*old] = partial_matches
            #     # partial_matches = new_partial_matches
            # elif(l != 0):
            #     pm = pair_matches[s]
            #     dat = np.empty(3,dtype=np.uint32)
            #     dat[0] = s
            #     dat[1] = l
            #     dat[2] = 0

            #     if(not pm[2] in i_elms):
            #         lst = List()
            #         lst.append(dat)
            #         i_elms[pm[2]] = lst
            #     else:
            #         i_elms[pm[2]].append(dat)
                # print(pm,"one",i)
                # partial_matches[:,pm[0]] = pm[2]
                # partial_matches[:,pm[1]] = pm[3]

    # old = len(partial_matches)
    # new_partial_matches = np.empty((old*len(i_elms),d))
    # i_elms = list(i_elms)
    # print("elems_i",i_elms)
    par_ms_list = List() 
    for elem in i_elms:
        da = i_elms[elem]
        # print("dat",da)

        
        # print("elem:",elem)
        # print()
        # par_ms = partial_matches.copy()
        # print(partial_matches)
        already_matches = np.empty(len(partial_matches),dtype=np.uint8)
        for j in range(len(partial_matches)):
            already_matches[j] = (partial_matches[j,:i] == elem).any() | (partial_matches[j,i+1:] == elem).any()
        # print(already_matches)
        par_ms_sel = (((partial_matches[:,i] == 0) | (partial_matches[:,i] == elem)) & ~already_matches).nonzero()[0]
        # print(par_ms_sel)
        par_ms = partial_matches[par_ms_sel,:].copy()
        

        par_ms[:,i] = elem

        # print("par_ms")
        # print(par_ms)
        
        # print("elem: ",elem)
        # print(par_ms)
        
        # for r in da:
        da_s = dict()
        for r in da:
            if(not r[0] in da_s):
                lst2 = List()
                lst2.append(r)
                da_s[r[0]] = lst2
            else:
                da_s[r[0]].append(r)
        # print("da_s")                   
        # print(da_s)                   

        for s in da_s:
            da = da_s[s]
            for r in da:
                k = r[2]
                l = len(da)
                # print(pair_matches[s+k])
                # print("")
                # print(s,l,k)
                if(l > 1):
                    raise ValueError()
                    old = len(par_ms)
                    new_partial_matches = np.empty((old*l,d),dtype=partial_matches.dtype)
                    for k in range(l):
                        pm = pair_matches[s+k]
                        # print(par_ms[:,pm[1]])
                        # print("here1k",pm[1], pm[3])
                        # print(elm,"k")
                        # partial_matches[:,pm[0]] = pm[2]
                        par_ms[:,pm[1]] = np.where(par_ms[:,pm[1]] == 0, pm[3],par_ms[:,pm[1]])
                        new_partial_matches[k*old:(k+1)*old] = par_ms
                    par_ms = new_partial_matches
                elif(l != 0):
                    pm = pair_matches[s+k]
                    # print(par_ms[:,pm[1]])
                    # print("here1",pm[1], pm[3])
                    # print(pm,"one",i)
                    # partial_matches[:,pm[0]] = pm[2]
                    par_ms[:,pm[1]] = np.where(par_ms[:,pm[1]] == 0, pm[3],par_ms[:,pm[1]])
                # par_ms
                # print("r",r)
        # print("^",par_ms)
        par_ms_list.append(par_ms)



            # pm = pair_matches[s+k]

            # print(pm)
        
        # elem = i_elms[k]
        # partial_matches[:,i] = elem
        # new_partial_matches[k*old:(k+1)*old] = partial_matches
    # partial_matches = new_partial_matches
    # np.array()
    n = 0
    for par_ms in par_ms_list:
        n += len(par_ms)
    partial_matches = np.empty((n,d),dtype=partial_matches.dtype)
    k = 0
    for par_ms in par_ms_list:
        # print(par_ms)
        partial_matches[k:k+len(par_ms),:] = par_ms
        k += len(par_ms)

    # partial_matches = np.concatenate(tuple(*par_ms_list))
    # print("OUT!")
    # print(partial_matches)
    return partial_matches

# def fill_partial_matches_around(partial_matches,i,pair_matches,pair_index_reg):
#     #just the index we start from
#     # for i in range(d):
#     print(i)
#     print(partial_matches )
#     # partial_matches = List()
#     d = partial_matches.shape[1]

#     for j in range(d):
#         if(pair_index_reg[i][j][0] != -1):
#             s,l = pair_index_reg[i][j][0],pair_index_reg[i][j][1]
#             if(l > 1):
#                 old = len(partial_matches)
#                 new_partial_matches = np.empty((old*l,d))
#                 for k in range(l):
#                     pm = pair_matches[s+k]
#                     print(pm,"k")
#                     # partial_matches[:,pm[0]] = pm[2]
#                     partial_matches[:,pm[1]] = pm[3]
#                     new_partial_matches[k*old:(k+1)*old] = partial_matches
#                 partial_matches = new_partial_matches
#             elif(l != 0):
#                 pm = pair_matches[s]
#                 print(pm,"one",i)
#                 # partial_matches[:,pm[0]] = pm[2]
#                 partial_matches[:,pm[1]] = pm[3]

#     print(partial_matches.shape)
#     print(partial_matches )

#     return partial_matches
#                 # break
    

@njit(nogil=True, parallel=False,fastmath=True,cache=True)
def match_iterative(split_ps, concept_slices,
                    elems,elems_slices,
                    concept_cands,cand_slices,
                    elem_names,
                    where_part_vals):
    d = len(concept_slices)-1
    adjacencies = compute_adjacencies(split_ps, concept_slices,where_part_vals)
    # adjacencies_nz = adjacencies.nonzero()
    pair_matches = List()
    # print("SHEEEE::",len(pair_matches))
    pair_index_reg = -np.ones((d,d,2),dtype=np.int16)
    # for a in range(len(adjacencies_nz[0])):
    for i in range(d):
        for j in range(d):
            # i,j = adjacencies_nz[0][a],adjacencies_nz[1][a]

            if(adjacencies[i][j] == 1):

                inds_i = concept_cands[cand_slices[i]:cand_slices[i+1]]
                inds_j = concept_cands[cand_slices[j]:cand_slices[j+1]]
                

                ps_i = split_ps[concept_slices[i]:concept_slices[i+1]]
                elem_names_i = elem_names[inds_i]
                elem_names_j = elem_names[inds_j]

                sel = (ps_i == where_part_vals[j]).nonzero()[0]
                # print("inds_i")
                # print(inds_i.shape)
                # print(elems)
                candidate_slices = gen_cand_slices(inds_i,elems,elems_slices,sel)

                # print("candidate_slices")
                # print(candidate_slices.shape)
                
                # print("elem_names_j")
                # print(elem_names_j)
                # print(elems_i)

                ps_i = split_ps[concept_slices[i]:concept_slices[i+1]]

                # consistencies = np.zeros((len(candidate_slices),len(elems_slices)-1),dtype=np.uint8)
                # ok_k = np.zeros(len(elem_names_i),dtype=np.uint8)
                # ok_r = np.zeros(len(elem_names_j),dtype=np.uint8)
                # assigned = False
                # print("SHEEEE::",i,j,len(pair_matches))
                pair_index_reg[i,j,0] = len(pair_matches)
                # pair_index_reg[j,i,0] = len(pair_matches)

                for k in range(len(inds_i)):
                    v = candidate_slices[k]
                    # print(v)
                    for r in range(len(elem_names_j)):
                        # print((candidate_slices[r], v))
                        # print((candidate_slices[r] == v))
                        con = (elem_names_j[r] == v).all()
                        # consistencies[k][inds_j[r]] = con
                        if(con):
                            # ok_k[r] = 1
                            # ok_r[k] = 1
                            pair_match = np.empty(4,dtype=np.uint16)
                            # # pair_match2 = np.empty(4,dtype=np.uint16)

                            pair_match[0] = i
                            pair_match[1] = j
                            pair_match[2] = elem_names_i[k]
                            pair_match[3] = elem_names_j[r]

                            # print(pair_match)
                            pair_matches.append(pair_match)

                            

                            # pair_match2[0] = elem_names_j[k]
                            # pair_match2[1] = elem_names_i[r]

                            # partial_match[i] = elem_names_i[r]
                            # partial_match[j] = elem_names_j[k]
                            # print(inds_i[r],inds_j[k])
                            # print(partial_match)
                            # if(not assigned):
                # print(i,j)
                # print(consistencies)

                
                # pair_matches.append(consistencies)
                            # pair_matches.append(pair_match2)

                pair_index_reg[i,j,1] = len(pair_matches)-pair_index_reg[i,j,0]
                # pair_index_reg[j,i,1] = len(pair_matches)-pair_index_reg[j,i,0]

            # for pm in consistencies.nonzero():
    # print(pair_index_reg[:,:,0])
    # print(pair_index_reg[:,:,1])
    # for p in pair_matches: 
    #     print(p)

    partial_matches = np.zeros((1,d),dtype=np.uint16)
    for i in range(d):
        # partial_matches = np.zeros((1,d),dtype=np.uint16)
        partial_matches = fill_partial_matches_at(partial_matches,i,pair_matches,pair_index_reg)
        # print(partial_matches)
        # fill_partial_matches_around(partial_matches,i,pair_matches,pair_index_reg)
    print("OUT",partial_matches)
    #just the index we start from
    # for i in range(d):
    #     print(i)
    #     # partial_matches = List()
    #     partial_matches = np.zeros((1,d),dtype=np.uint16)

    #     for j in range(d):
    #         if(pair_index_reg[i][j][0] != -1):
    #             s,l = pair_index_reg[i][j][0],pair_index_reg[i][j][1]
    #             if(l > 1):
    #                 old = len(partial_matches)
    #                 new_partial_matches = np.empty((old*l,d))
    #                 for k in range(l):
    #                     pm = pair_matches[s+k]
    #                     print(pm,"k")
    #                     # partial_matches[:,pm[0]] = pm[2]
    #                     partial_matches[:,pm[1]] = pm[3]
    #                     new_partial_matches[k*old:(k+1)*old] = partial_matches
    #                 partial_matches = new_partial_matches
    #             elif(l != 0):
    #                 pm = pair_matches[s]
    #                 print(pm,"one",i)
    #                 # partial_matches[:,pm[0]] = pm[2]
    #                 partial_matches[:,pm[1]] = pm[3]
    #                 # break
    #     print(partial_matches.shape )
    #     print(partial_matches )



        # for i in range(d):
        #     for j in range(d):
        #         c = pair_index_reg[i,j]
        #         if(c != -1):
        #             inds_i = concept_cands[cand_slices[i]:cand_slices[i+1]]
        #             inds_j = concept_cands[cand_slices[j]:cand_slices[j+1]]
                    

        #             ps_i = split_ps[concept_slices[i]:concept_slices[i+1]]
        #             elem_names_i = elem_names[inds_i]
        #             elem_names_j = elem_names[inds_j]

        #             print(i,j)
        #             print(elem_names_i)
        #             print(elem_names_j)
                    
        #             print(pair_matches[c].shape)
        #     print()
        #         # print(i,j)

        # for p in pair_matches:
        #     print(p)

            # print(candidate_slices,v)
            # blehh[k] = (candidates_slice == v)
        # inds_i = consistencies.
        # inds_j = consistencies.
        #     print("consistencies")
            # print(consistencies)
            # print(inds_i,inds_j)
            # print(ok_k,ok_r)

    # print(pair_index_reg[:,:,0])
    # print(pair_index_reg[:,:,1])
    # # new_matches = List()
    # assigned = np.zeros(d,dtype=np.uint8)
    # bleep = List()
    # for k in range(len(pair_matches)):
    #     p = pair_matches[k]
    #     # print(p)
        

    #     # return
    #     partial_matches = List()
    #     partial_match = np.zeros(d,dtype=np.uint16)
    #     # print(p)
    #     # print(partial_match)
    #     partial_match[p[0]] = p[2]
    #     partial_match[p[1]] = p[3]
        
    #     for p in gen_partial_matches(partial_match,pair_matches,pair_index_reg):
    #         bleep.append(p)
    #     # return
        # for 
        # if(pair_matches[])
        # print(partial_match)

    # for i in range(d):
    #     for j in range(d):
    #         print(p1)
        # k1,v1 = first_nonzero(p1)
        # for p2 in pair_matches:
        #     if(p2[k1] == v1):
        #         print(p2)

            # inds_i = concept_cands[cand_slices[i]:cand_slices[i+1]]
            # inds_j = concept_cands[cand_slices[j]:cand_slices[j+1]]
            
            # elem_names_i = elem_names[inds_i]
            # elem_names_j = elem_names[inds_j]


    # print("Bleep")
    # for p in bleep:
    #     print(p)
    #     # print(p)
    # print(adjacencies.shape)
    # print(adjacencies)
    # print(np.matmul(adjacencies,adjacencies))
        # ns_i = split_ns[concept_slices[i]:concept_slices[i+1]]
        # _,elem_names_j = elem_names[]
            # ps_j,ns_j = split_ps[j],split_ns[j]
        # print(i,j)
    # print(adjacencies)
    # print(adjacencies.nonzero())
    # return adjacencies.nonzero()
    # for i in range(d):
    #     for j in range(d):
    #         if(i > j)

    # return 

# @njit(nogil=True, parallel=True,cache=True)
def numpy_mi(split_ps,elems,elem_names,concept_cand_indicies,where_part_vals):
    adjacencies = np.empty((len(split_ps),len(split_ps)))
    for i,concept in enumerate(split_ps):
        adjacencies[i,:] = (concept == np.expand_dims(where_part_vals,-1)).any(axis=1)

    # print(adjacencies)
    return adjacencies
    

where_part_vals = np.array([ 2,  3,  4, 10, 11, 12, 13], dtype=np.uint8)

elems = \
[np.array([14, 20, 18, 21, 1, 1, 1], dtype=np.uint8),
 np.array([5, 1, 1, 20, 21, 8, 1], dtype=np.uint8),
 np.array([5, 1, 1, 9, 7, 16, 18], dtype=np.uint8),
 np.array([5, 1, 1, 17, 15, 19, 8], dtype=np.uint8),
 np.array([5, 1, 1, 6, 22, 1, 16], dtype=np.uint8),
 np.array([5, 18, 1, 6, 21, 9, 1], dtype=np.uint8),
 np.array([5, 8, 1, 6, 7, 17, 20], dtype=np.uint8),
 np.array([5, 16, 1, 6, 15, 1, 9], dtype=np.uint8),
 np.array([5, 6, 20, 1, 1, 7, 1], dtype=np.uint8),
 np.array([5, 6, 9, 1, 1, 15, 21], dtype=np.uint8),
 np.array([5, 6, 17, 1, 1, 22, 7], dtype=np.uint8),
 np.array([5, 6, 19, 1, 1, 1, 15], dtype=np.uint8)]

split_ps = \
[np.array([[ 5, 10,  4,  1,  1, 11,  1]], dtype=np.uint8),
 np.array([[ 5,  1,  1,  4,  2, 12,  1]], dtype=np.uint8),
 np.array([[ 5,  3,  1, 10,  2, 13,  1]], dtype=np.uint8),
 np.array([[14,  4,  3,  2,  1,  1,  1]], dtype=np.uint8),
 np.array([[ 5, 10, 13,  1,  1, 15,  2]], dtype=np.uint8),
 np.array([[ 5,  1,  1, 13, 11, 16,  3]], dtype=np.uint8),
 np.array([[ 5, 12,  1, 10, 11, 17]], dtype=np.uint8)]
# print(split_ps.shape)
og_split_ps = split_ps

concept_cand_indicies = \
[np.array([7, 8, 9], dtype=np.uint8),
  np.array([0, 1, 2, 4, 5], dtype=np.uint8),
  np.array([4, 5], dtype=np.uint8),
  np.array([0], dtype=np.uint8),
  np.array([8, 9], dtype=np.uint8),
  np.array([1, 2, 5], dtype=np.uint8),
  np.array([5], dtype=np.uint8)]

elem_names = \
np.array([ 6, 18,  8, 16, 19, 20,  9, 17, 21,  7, 15, 22], dtype=np.uint8)

import itertools
def flatten_n_slice(lst):
    flat = [x if isinstance(x,list) else x.reshape(-1) for x in lst]
    lens = [0] + [len(x)  for x in flat]
    slices = np.cumsum(lens)
    out = np.array([x for x in itertools.chain(*flat)])
    print(out)
    print(slices)
    return out,slices
    # print(slices)





# @njit(nogil=True, parallel=False,fastmath=True)
# def foo(s):
#     return 0
# timefunc("foo",foo,s)

split_ps, concept_slices = flatten_n_slice(split_ps)
elems, elems_slices = flatten_n_slice(elems)
concept_cands, cand_slices = flatten_n_slice(concept_cand_indicies)




# timefunc("match_iterative",match_iterative,
#                     split_ps, concept_slices,
#                     elems,elems_slices,
#                     concept_cands,cand_slices,
#                     elem_names,
#                     where_part_vals)
# timefunc("numpy_mi",numpy_mi,og_split_ps,elems,elem_names,concept_cand_indicies,where_part_vals)
print(match_iterative(split_ps, concept_slices,
                    elems,elems_slices,
                    concept_cands,cand_slices,
                    elem_names,
                    where_part_vals))
print(numpy_mi(og_split_ps,elems,elem_names,concept_cand_indicies,where_part_vals))








# test_list = [np.arange(6),np.arange(8),np.arange(6),np.arange(8),np.arange(6),np.arange(8)]
# sel_list = [0,2,4]
# construct_sub_list(test_list,sel_list)

# timefunc(None,"construct_sub_list",construct_sub_list,test_list,sel_list)

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
# d = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ '])
# d2 = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ '])
# np.char.add(d,d2)
# n = int(1e6)

# s_list = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ '], dtype=str)
# # print(s_list.dtype)
# s_list = s_list.view(np.uint32).reshape(3,-1)

# print(encode_str(upper(s_list)))
# print(encode_str(lower(s_list)))
# # print(encode_str(concatenate(s_list,s_list)))
# # print(s_list.shape,con.shape)
# print("V")
# print(decode_str(" fooperz").shape)
# print(decode_str([" A23"," B23"," C23"]).shape)
# print("&")
# print(encode_str(concatenate(s_list,decode_str([" A23"," B23"," C23"]))))
# print(encode_str(concatenate(s_list,decode_str(" fooperz"))))
# print(encode_str(concatenate(decode_str("fooperz "),s_list)))
# print(test_curry(decode_str("fooperz "),s_list))
# # print(concatenate_right(s_list,"moop").view("U1"))
# # s_list = List()
# # for i in range(n):
#     # s_list.append(str(np.random.ranf()))
# s_list = np.array(['abc','snake','plateblxyz']*n, dtype=str)
# # print(s_list.dtype)
# s_list = s_list.view(np.uint32).reshape(3*n,-1)



# timefunc(None,"upper",upper,s_list)
# timefunc(None,"str_len",str_len,s_list)
# timefunc(None,"concatenate",concatenate,s_list,s_list)
# rand = (np.random.random((3,n))*1000).view(np.uint32).reshape(3,-1)
# timefunc(None,"concatenate-random",concatenate,rand,rand)
# timefunc(None,"concatenate-left",concatenate,decode_str("fooperz "),s_list)
# timefunc(None,"concatenate-right",concatenate,s_list,decode_str("fooperz "))
# timefunc(None,"test_curry",test_curry,s_list,decode_str("fooperz "))

# d = np.array(['AaBc%$@ ','snAkE123 ','pateblxyzZ ']*n)
# # timefunc(None,"concatenate-numpy",np.char.add,d,d)



# elems = List()
# elems.append(np.array([14, 20, 18, 21, 1, 1, 1], dtype=np.uint8))
# elems.append(np.array([5, 1, 1, 20, 21, 8, 1], dtype=np.uint8))
# elems.append(np.array([5, 1, 1, 9, 7, 16, 18], dtype=np.uint8))
# elems.append(np.array([5, 1, 1, 17, 15, 19, 8], dtype=np.uint8))
# elems.append(np.array([5, 1, 1, 6, 22, 1, 16], dtype=np.uint8))
# elems.append(np.array([5, 18, 1, 6, 21, 9, 1], dtype=np.uint8))
# elems.append(np.array([5, 8, 1, 6, 7, 17, 20], dtype=np.uint8))
# elems.append(np.array([5, 16, 1, 6, 15, 1, 9], dtype=np.uint8))
# elems.append(np.array([5, 6, 20, 1, 1, 7, 1], dtype=np.uint8))
# elems.append(np.array([5, 6, 9, 1, 1, 15, 21], dtype=np.uint8))
# elems.append(np.array([5, 6, 17, 1, 1, 22, 7], dtype=np.uint8))
# elems.append(np.array([5, 6, 19, 1, 1, 1, 15], dtype=np.uint8))
# # List([np.array([14, 20, 18, 21, 1, 1, 1], dtype=np.uint8),
# #  np.array([5, 1, 1, 20, 21, 8, 1], dtype=np.uint8),
# #  np.array([5, 1, 1, 9, 7, 16, 18], dtype=np.uint8),
# #  np.array([5, 1, 1, 17, 15, 19, 8], dtype=np.uint8),
# #  np.array([5, 1, 1, 6, 22, 1, 16], dtype=np.uint8),
# #  np.array([5, 18, 1, 6, 21, 9, 1], dtype=np.uint8),
# #  np.array([5, 8, 1, 6, 7, 17, 20], dtype=np.uint8),
# #  np.array([5, 16, 1, 6, 15, 1, 9], dtype=np.uint8),
# #  np.array([5, 6, 20, 1, 1, 7, 1], dtype=np.uint8),
# #  np.array([5, 6, 9, 1, 1, 15, 21], dtype=np.uint8),
# #  np.array([5, 6, 17, 1, 1, 22, 7], dtype=np.uint8),
# #  np.array([5, 6, 19, 1, 1, 1, 15], dtype=np.uint8)])

# split_ps = List()
# elems.append(np.array([ 5, 10,  4,  1,  1, 11,  1], dtype=np.uint8))
# elems.append(np.array([ 5,  1,  1,  4,  2, 12,  1], dtype=np.uint8))
# elems.append(np.array([ 5,  3,  1, 10,  2, 13,  1], dtype=np.uint8))
# elems.append(np.array([14,  4,  3,  2,  1,  1,  1], dtype=np.uint8))
# elems.append(np.array([ 5, 10, 13,  1,  1, 15,  2], dtype=np.uint8))
# elems.append(np.array([ 5,  1,  1, 13, 11, 16,  3], dtype=np.uint8))
# elems.append(np.array([ 5, 12,  1, 10, 11, 17,  4], dtype=np.uint8))