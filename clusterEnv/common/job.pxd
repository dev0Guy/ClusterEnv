import numpy as np
cimport numpy as cnp

cnp.import_array()

cpdef enum JobStatus:
    NONEXISTENT = 0
    PENDING
    RUNNING
    COMPLETE

cdef class Job:
    cdef JobStatus status
    cdef cnp.ndarray spec
    cdef unsigned long submission
    cdef public unsigned long wait_time, run_time

cdef class Jobs:
    cdef cnp.ndarray spec
    cdef cnp.ndarray status
    cdef cnp.ndarray submission, wait_time, run_time, length

    cdef void tick(self, unsigned long)

