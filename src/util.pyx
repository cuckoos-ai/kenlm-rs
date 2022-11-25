import io
from typing import Iterator

cimport ngram
import numpy as np
from enum import Enum
from libcpp.vector cimport vector
from libcpp.string cimport string


class Py_LoadMethod(Enum):
    LAZY = LAZY
    POPULATE_OR_LAZY = POPULATE_OR_LAZY
    POPULATE_OR_READ = POPULATE_OR_READ
    READ = READ
    PARALLEL_READ = PARALLEL_READ


cpdef int Py_GuessPhysicalMemory():
    return GuessPhysicalMemory()

cpdef int Py_ParseSize(string arg):
    return ParseSize(& arg)

cpdef int Py_ParseBitCount(string inp_):
	return ngram.ParseBitCount(inp_)

class Py_StringPiece:
    cdef StringPiece sp

    cpdef __cinit__(self, str string_):
        self.sp = StringPiece(string_)

    cpdef __dealloc__(self):
        del self.sp
    def __iter__(self):
        """
        An string iterator to iterate over string
        Returns
        -------

        """
        yield

class Py_ChainConfig:
    cdef ChainConfig *chain_cfg_
    def __init__(self, size_t in_entry_size_, size_t in_block_count, size_t in_total_memory):
        self.chain_cfg_.entry_size = in_entry_size_
        self.chain_cfg_.block_count = in_block_count
        self.chain_cfg_.total_memory = in_total_memory

    @property
    def in_entry_size(self):
        return self.chain_cfg_.entry_size
    @property
    def in_block_count(self):
        return self.chain_cfg_.block_count
    @property
    def in_total_memory(self):
        return self.chain_cfg_.total_memory


class Py_Chain:
    cdef Chain * __chainer

    def __cinit__(self, Py_ChainConfig config):
        # TODO: FixMe - Do type conversion
        self.__chainer = Chain(config.chain_cfg_)

    def __dealloc__(self):
        del self.__chainer

    def __len__(self):
        pass

cpdef class FixedArrayMixin:
    cpdef T * begin(self):
        pass
    cpdef T * end(self):
        pass
    cpdef T & back(self):
        pass
    cpdef size_t size(self):
        pass
    cpdef bool empty(self):
        pass
    cpdef T & operator[](self, size_t i):
        pass
    def push_back(self):
        pass
    def pop_back(self):
        pass
    def clear(self):
        pass

cpdef class Py_Chains(FixedArrayMixin):
    cdef Chains chains
    cpdef __cinit__(self):
        pass
    cpdef __dealloc__(self):
        del self.chains


ctypedef FixedArray[scoped_ptr[SplitWorker]] FixedArrayWorker

cpdef class Py_ChainPosition:
    cdef ChainPosition c_pos_

    cpdef __cinit__(self, **kwds):
        self.c_pos_ = ChainPosition(kwds)
    cpdef __dealloc__(self):
        del self.c_pos_

    cpdef Py_Chain GetChain(self):
        pass


cpdef class PySorts(FixedArrayMixin):
    cdef Sorts _sorted
    cpdef __cinit__(self, size_t number):
        # Sorts[ContextOrder]
        self._sorted = Sorts()
    cpdef __dealloc__(self):
        del self._sorted

class Py_FilePiece:
    cdef FilePiece file_piece
    cdef size_t min_buffer

    def __cinit__(self, io.FileIO fp, bool show_progress = False, size_t min_buffer = 1048576):
        self.fp = fp
        self.show_progress = show_progress
        self.min_buffer = min_buffer

    def begin(self) -> Iterator:
        pass

    def end(self) -> Iterator:
        pass

    def peek(self) -> str:
        pass

    def ReadDelimited(self, bool[:] delim = np.zeros((256,), dtype=bool)) -> Py_StringPiece:
        pass

    def ReadWordSameLine(self, Py_StringPiece to, bool[:] delim = np.zeros((256,), dtype=bool)) -> bool:
        pass

    def ReadLine(self, str delim = '\n', bool strip_cr = True) -> Py_StringPiece:
        pass

    def ReadLineOrEOF(self, Py_StringPiece to, char delim = '\n', bool strip_cr = True) -> bool:
        pass

    def ReadFloat(self) -> np.float:
        pass

    def ReadDouble(self) -> np.double:
        pass

    def ReadLong(self) -> np.long:
        pass

    def ReadULong(self) -> np.uint:
        pass

    def SkipSpaces(self, bool[:] delim = np.zeros((256,), dtype=bool)) -> None:
        # Skip spaces defined by isspace.
        pass

    def Offset(self) -> np.intc:
        pass

    def FileName(self) -> str:
        pass

    def UpdateProgress(self) -> None:
        # Force a progress update.
        pass

    def __dealloc__(self):
        del self.file_piece

cpdef class Py_RecyclingThreadPool:
    cdef :
        size_t queue
        size_t workers
        Construct handler_construct
        Request poison

    cpdef __cinit__(self, size_t queue, size_t workers, Construct handler_construct,
                    Request poison, HandlerType HandlerT = None):
        self._pool = RecyclingThreadPool[HandlerT](queue, workers, handler_construct, poison)
    cpdef __dealloc__(self):
        del self._pool
    def PopulateRecycling(self, Request request):
        self._pool.PopulateRecycling(request)

    cpdef Request Consume(self):
        self._pool.Consume()

    def Produce(self, const Request &request):
        self._pool.Produce(request)