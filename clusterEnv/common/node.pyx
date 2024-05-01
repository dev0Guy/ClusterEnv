import numpy as np

cdef class Node:

    def __cinit__(
        self,
        cnp.ndarray[double, ndim=2] spec,
        cnp.ndarray[double, ndim=2] usage = None,
    ):
        self.spec = spec
        if usage is None:
            usage = np.zeros(spec[:].shape, dtype=spec.dtype)
        self.usage = usage

    def __str__(self) -> str:
        return f"{type(self).__name__}(spec={self.spec}, usage={self.usage})"

cdef class Nodes:
    def __cinit__(
        self,
        cnp.ndarray[double, ndim=3] spec,
    ):
        self.spec = spec
        self.usage = np.zeros(spec[:].shape, dtype=spec.dtype)

    cdef void tick(self, unsigned long time):
        self.usage = np.roll(self.usage, shift=-1, axis=-1)
        self.usage[:, :, -1] = 0

    def __getitem__(self, size_t index) -> Node:
        return Node(self.spec[index], self.usage[index])

    def __len__(self) -> size_t:
        return len(self.spec)