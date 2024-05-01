import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef class Node:
    cdef cnp.ndarray spec, usage


cdef class Nodes:
    cdef cnp.ndarray spec, usage
    cdef void tick(self, unsigned long)



