import numpy as np
from libc.stdint cimport uint64_t, uint32_t, uint8_t, uint16_t
from constant cimport size_t


_TYPE_MAP = \
{
    np.int8: uint8_t,
    np.int16: uint16_t,
    np.int32: uint32_t,
    np.int64: uint64_t
}

cdef cppclass Model:
    pass

cdef cppclass Width:
    pass

cdef struct BenchmarkConfig:
    int fd_in
    size_t threads
    size_t buf_per_thread
    bool query


cpdef QueryFromBytes(Model model, BenchmarkConfig config, uint8_t Width):
    pass

cpdef ConvertToBytes(Model &model, int fd_in, Width=uint8_t):
    pass

cpdef class Worker:
    pass

cpdef class Py_RecyclingThreadPool:
    pass


