import numpy as np
import pandas as pd
from math import ceil
from numba import cuda
from numba import jit

##########################################################################

def alg1CPU(fs, bpa, be, pl):

    # iset is the index of one of the nodes in the lattice
    for iset in range(be.shape[0]): 
        # k is the index of one of the focal elements in fs (the focal set)
        for k in range(bpa.shape[0]):
            el = fs[k] # el is the current focal element to study
            if (iset & el) == el: # belief: the focal element is a subset of the node
                be[iset] += bpa[k]
            if (iset & el) > 0: # plausibility: the intersection is not empty
                pl[iset] += bpa[k]    
    return be, pl

##########################################################################

@jit
def alg1JIT(fs, bpa, be, pl):

    # iset is the index of one of the nodes in the lattice
    for iset in range(be.shape[0]): 
    # be[iset] = 0 # commented because I'm passing array of zeros
    # pl[iset] = 0 # commented because I'm passing array of zeros
    # k is the index of one of the focal elements in fs
        for k in range(bpa.shape[0]):
            el = fs[k] # el is the current focal element to study
            if (iset & el) == el: # belief: the focal element is a subset of the node
                be[iset] += bpa[k]
            if (iset & el) > 0: # plausibility: the intersections is not empty
                pl[iset] += bpa[k]
    return be, pl

##########################################################################

@cuda.jit
def alg1GPU(fs, bpa, be, pl):

    iset = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    nfocal = fs.shape[0]
    nnodes = be.shape[0]

    if iset < nnodes:
        be[iset] = 0 # gpu does not initialize to zeros so this is necessary
        pl[iset] = 0 # gpu does not initialize to zeros so this is necessary

        for k in range(nfocal):
            el = fs[k]
            if (iset & el) == el: # belief
                be[iset] += bpa[k]
            if (iset & el) > 0: # plausibility
                pl[iset] += bpa[k]
