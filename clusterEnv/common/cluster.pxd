from .job cimport Job, Jobs, JobStatus
from .node cimport Node, Nodes

cimport numpy as cnp


ctypedef unsigned int NodeIndex
ctypedef unsigned int JobIndex

cdef class StepReturn:
    cdef:
        dict observation
        double reward
        bint terminate
        bint trounced
        dict info

cdef class Cluster:
    cdef unsigned int current_time, num_steps, max_episode_steps
    cdef Nodes nodes
    cdef Jobs jobs

    cdef inline dict observation(self)

    cdef inline (NodeIndex, JobIndex) convert_action(self, unsigned int) except *

    cpdef void schedule(self, NodeIndex, JobIndex) except *




