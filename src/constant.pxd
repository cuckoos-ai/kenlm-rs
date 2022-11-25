cimport numpy as np
from enum import Enum
from libc.stdint cimport uintptr_t
from cpython import (PyUnicode_AsUTF8String, PyUnicode_DecodeUTF8, PyBytes_CheckExact,
                     PyBytes_FromStringAndSize, PyBytes_GET_SIZE, PyBytes_AS_STRING)

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_int64 INT64_t            # Signed 64 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer
ctypedef np.npy_uint64 UINT64_t          # Unsigned 64 bit integer
ctypedef uintptr_t size_t


cdef enum ARPALoadComplain:
    ALL = 1
    EXPENSIVE = 2
    NONE = 3


cdef enum WriteMethod:
    WRITE_MMAP = 1  # Map the file directly.
    WRITE_AFTER = 2  # Write after we're done.


# Left rest options. Only used when the model includes rest costs.
cdef enum RestFunction:
    REST_MAX = 1 # Maximum of any score to the left
    REST_LOWER = 2 # Use lower-order files given below.


cpdef enum FilterMode:
    MODE_COPY = 1
    MODE_SINGLE = 2
    MODE_MULTIPLE = 3
    MODE_UNION = 4
    MODE_UNSET = 5


cpdef enum WarningAction:
    THROW_UP = 1
    COMPLAIN = 2
    SILENT = 3

# cdef enum WarningAction:
#     THROW_UP = 1
#     COMPLAIN = 2
#     SILENT = 3

cpdef enum Format:
    FORMAT_ARPA = 1
    FORMAT_COUNT = 2

