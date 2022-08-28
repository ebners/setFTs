import numpy as np

#returns all indicator vectors of size 'ground_set_size' as a numpy array
def get_indicator_set(ground_set_size,reverse=False):
    inds = np.asarray([int2indicator(A, ground_set_size) for A in range(2**ground_set_size)]).astype(np.int32)
    return inds

# Transform integer representing set A to indicator vector representing A. Enumeration done lexycographically. 0 -> \empty, 1 -> [1,0,...], 2 -> [0,1,0,...], 3->[1,1,0,...], , 4->[0,0,1,...]
def int2indicator(A, n_groundset, reverse=False):
    indicator = [int(b) for b in bin(2**n_groundset + A)[3:][::-1]]
    indicator = np.asarray(indicator, dtype=bool)
    if reverse:
        indicator = np.flip(indicator, axis=0)
    return indicator
