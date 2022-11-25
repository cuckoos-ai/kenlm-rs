// import io
// from typing import Iterator

// cimport ngram
// import numpy as np
// from enum import Enum
// from libcpp.vector cimport vector
// from libcpp.string cimport string


// class Py_LoadMethod(Enum):
//     LAZY = LAZY
//     POPULATE_OR_LAZY = POPULATE_OR_LAZY
//     POPULATE_OR_READ = POPULATE_OR_READ
//     READ = READ
//     PARALLEL_READ = PARALLEL_READ


// cpdef int Py_GuessPhysicalMemory():
//     return GuessPhysicalMemory()

// cpdef int Py_ParseSize(string arg):
//     return ParseSize(& arg)

// cpdef int Py_ParseBitCount(string inp_):
// 	return ngram.ParseBitCount(inp_)

// class Py_StringPiece:
//     cdef StringPiece sp

//     cpdef __cinit__(self, str string_):
//         self.sp = StringPiece(string_)

//     cpdef __dealloc__(self):
//         del self.sp
//     def __iter__(self):
//         """
//         An string iterator to iterate over string
//         Returns
//         -------

//         """
//         yield

// class Py_ChainConfig:
//     cdef ChainConfig *chain_cfg_
//     def __init__(self, size_t in_entry_size_, size_t in_block_count, size_t in_total_memory):
//         self.chain_cfg_.entry_size = in_entry_size_
//         self.chain_cfg_.block_count = in_block_count
//         self.chain_cfg_.total_memory = in_total_memory

//     @property
//     def in_entry_size(self):
//         return self.chain_cfg_.entry_size
//     @property
//     def in_block_count(self):
//         return self.chain_cfg_.block_count
//     @property
//     def in_total_memory(self):
//         return self.chain_cfg_.total_memory


// class Py_Chain:
//     cdef Chain * __chainer

//     def __cinit__(self, Py_ChainConfig config):
//         # TODO: FixMe - Do type conversion
//         self.__chainer = Chain(config.chain_cfg_)

//     def __dealloc__(self):
//         del self.__chainer

//     def __len__(self):
//         pass

// cpdef class FixedArrayMixin:
//     cpdef T * begin(self):
//         pass
//     cpdef T * end(self):
//         pass
//     cpdef T & back(self):
//         pass
//     cpdef size_t size(self):
//         pass
//     cpdef bool empty(self):
//         pass
//     cpdef T & operator[](self, size_t i):
//         pass
//     def push_back(self):
//         pass
//     def pop_back(self):
//         pass
//     def clear(self):
//         pass

// cpdef class Py_Chains(FixedArrayMixin):
//     cdef Chains chains
//     cpdef __cinit__(self):
//         pass
//     cpdef __dealloc__(self):
//         del self.chains


// ctypedef FixedArray[scoped_ptr[SplitWorker]] FixedArrayWorker

// cpdef class Py_ChainPosition:
//     cdef ChainPosition c_pos_

//     cpdef __cinit__(self, **kwds):
//         self.c_pos_ = ChainPosition(kwds)
//     cpdef __dealloc__(self):
//         del self.c_pos_

//     cpdef Py_Chain GetChain(self):
//         pass


// cpdef class PySorts(FixedArrayMixin):
//     cdef Sorts _sorted
//     cpdef __cinit__(self, size_t number):
//         # Sorts[ContextOrder]
//         self._sorted = Sorts()
//     cpdef __dealloc__(self):
//         del self._sorted

// class Py_FilePiece:
//     cdef FilePiece file_piece
//     cdef size_t min_buffer

//     def __cinit__(self, io.FileIO fp, bool show_progress = False, size_t min_buffer = 1048576):
//         self.fp = fp
//         self.show_progress = show_progress
//         self.min_buffer = min_buffer

//     def begin(self) -> Iterator:
//         pass

//     def end(self) -> Iterator:
//         pass

//     def peek(self) -> str:
//         pass

//     def ReadDelimited(self, bool[:] delim = np.zeros((256,), dtype=bool)) -> Py_StringPiece:
//         pass

//     def ReadWordSameLine(self, Py_StringPiece to, bool[:] delim = np.zeros((256,), dtype=bool)) -> bool:
//         pass

//     def ReadLine(self, str delim = '\n', bool strip_cr = True) -> Py_StringPiece:
//         pass

//     def ReadLineOrEOF(self, Py_StringPiece to, char delim = '\n', bool strip_cr = True) -> bool:
//         pass

//     def ReadFloat(self) -> np.float:
//         pass

//     def ReadDouble(self) -> np.double:
//         pass

//     def ReadLong(self) -> np.long:
//         pass

//     def ReadULong(self) -> np.uint:
//         pass

//     def SkipSpaces(self, bool[:] delim = np.zeros((256,), dtype=bool)) -> None:
//         # Skip spaces defined by isspace.
//         pass

//     def Offset(self) -> np.intc:
//         pass

//     def FileName(self) -> str:
//         pass

//     def UpdateProgress(self) -> None:
//         # Force a progress update.
//         pass

//     def __dealloc__(self):
//         del self.file_piece

// cpdef class Py_RecyclingThreadPool:
//     cdef :
//         size_t queue
//         size_t workers
//         Construct handler_construct
//         Request poison

//     cpdef __cinit__(self, size_t queue, size_t workers, Construct handler_construct,
//                     Request poison, HandlerType HandlerT = None):
//         self._pool = RecyclingThreadPool[HandlerT](queue, workers, handler_construct, poison)
//     cpdef __dealloc__(self):
//         del self._pool
//     def PopulateRecycling(self, Request request):
//         self._pool.PopulateRecycling(request)

//     cpdef Request Consume(self):
//         self._pool.Consume()

//     def Produce(self, const Request &request):
//         self._pool.Produce(request)


// ---- Fix PYX below
// Utils.pyx
// 
// cimport constant
// cimport builder
// from libcpp.string cimport string
// from libc.stdint cimport uint64_t
// from libcpp.vector cimport vector


// cdef extern from "<iostream>" nogil:
// 	cdef cppclass ostream:
// 		ostream& write(const char *, int) except +

// 	cdef cppclass istream:
// 		istream& read(char *, int) except +


// cdef extern from "util/mmap.hh" namespace "util" nogil:
// 	cdef enum LoadMethod:
// 		LAZY = 1
// 		POPULATE_OR_LAZY = 2
// 		POPULATE_OR_READ = 3
// 		READ = 4
// 		PARALLEL_READ = 5


// cdef extern from "util/fixed_array.hh" namespace "util" nogil:
// 	cdef cppclass FixedArray[T]:
// 		T *begin() except +
// 		T *end() except +
// 		T & back() except +
// 		size_t size() except +
// 		bool empty() except +
// 		T & operator[](size_t i) except +
// 		void push_back() except +
// 		void pop_back() except +
// 		void clear() except +


// cdef extern from "util/file_piece.hh" namespace "util":
// 	cdef cppclass LineIterator:
// 		pass

// 	# Sigh this is the only way I could come up with to do a _const_ bool.
// 	# It has ' ', '\f', '\n', '\r', '\t', and '\v' (same as isspace on C locale).
// 	const bool kSpaces[256]

// 	cdef cppclass FilePiece:
// 		# Takes ownership of fd.  name is used for messages.
// 		FilePiece(const char *file, ostream * show_progress = NULL, size_t min_buffer = 1048576) except +

// 		# Read from an istream. Don't use this if you can avoid it. Raw fd IO is
// 		# much faster. But sometimes you just have an istream like Boost's HTTP
// 		# server and want to parse it the same way. name is just used for messages and FileName().
// 		FilePiece(istream & stream, const char *name = NULL, size_t min_buffer = 1048576) except +

// 		LineIterator begin() except +
// 		LineIterator end() except +
// 		char peek() except +

// 		# Leaves the delimiter, if any, to be returned by get().  Delimiters defined by isspace().
// 		StringPiece ReadDelimited(const bool *delim = kSpaces) except +

// 		# Read word until the line or file ends.
// 		bool ReadWordSameLine(StringPiece & to, const bool *delim = kSpaces) except +

// 		# Read a line of text from the file.
// 		# Unlike ReadDelimited, this includes leading spaces and consumes the
// 		# delimiter. It is similar to getline in that way.
// 		#
// 		# If strip_cr is true, any trailing carriate return (as would be found on
// 		# a file written on Windows) will be left out of the returned line.
// 		#
// 		# Throws EndOfFileException if the end of the file is encountered. If the
// 		# file does not end in a newline, this could mean that the last line is never read.
// 		StringPiece ReadLine(char delim = '\n', bool strip_cr = True) except +

// 		# Read a line of text from the file, or return false on EOF.
// 		# This is like ReadLine, except it returns false where ReadLine throws
// 		# EndOfFileException.  Like ReadLine it may not read the last line in the
// 		# file if the file does not end in a newline.
// 		#
// 		# If strip_cr is true, any trailing carriate return (as would be found on
// 		# a file written on Windows) will be left out of the returned line.
// 		bool ReadLineOrEOF(StringPiece & to, char delim = '\n', bool strip_cr = True) except +
// 		float ReadFloat() except +
// 		double ReadDouble() except +
// 		long int ReadLong() except +
// 		unsigned long int ReadULong() except +

// 		# Skip spaces defined by isspace.
// 		void SkipSpaces(const bool *delim = kSpaces) except +

// 		uint64_t Offset() except +
// 		const string & FileName() except +
// 		# Force a progress update.
// 		void UpdateProgress() except +


// cdef extern from "util/usage.cc" namespace "util":
// 	cdef uint64_t GuessPhysicalMemory() except +
// 	cdef uint64_t ParseSize(string & arg) except +


// cdef extern from "util/string_piece.hh" nogil:
// 	cdef cppclass StringPiece:
// 		ctypedef size_t size_type

// 		StringPiece(const string& string_) except +
// 		StringPiece(const char * string_) except +



// cdef extern from "util/file.cc" namespace "util":
// 	cdef cppclass scoped_fd:
// 		scoped_fd()
// 		int get() const
// 		int release()

// 	# If it's a directory, add a /.  This lets users say -T /tmp without creating /tmpAAAAAA
// 	cdef void NormalizeTempPrefix(string & base) except +
// 	cdef int CreateOrThrow(const char *name) except +
// 	cdef int OpenReadOrThrow(const char *name) except +


// cdef extern from "util/usage.cc" namespace "util":
// 	cdef void PrintUsage(ostream & out) except +


// cdef extern from "util/scoped.hh" namespace "util":
// 	cdef cppclass MallocException:
// 		MallocException(size_t requested) except +


// cdef extern from "util/usage.hh" namespace "util":
// 	cdef:
// 		# Time in seconds since process started.  Zero on unsupported platforms.
// 		double WallTime() except +

// 		# User + system time, process-wide.
// 		double CPUTime() except +

// 		# User + system time, thread-specific.
// 		double ThreadTime() except +

// 		# Resident usage in bytes.
// 		uint64_t RSSMax() except +

// 		void PrintUsage(ostream & to) except +

// 		# Determine how much physical memory there is.  Return 0 on failure.
// 		uint64_t GuessPhysicalMemory() except +

// 		# Parse a size like unix sort.  Sadly, this means the default multiplier is K.
// 		uint64_t ParseSize(const string & arg) except +


// cdef extern from "util/thread_pool.hh" namespace "util":
// 	# cdef cppclass HandlerT:
// 	#     pass
// 	#
// 	# cdef cppclass Construct:
// 	#     pass
// 	# ctypedef HandlerT Handler
// 	# ctypedef HandlerT.Request Request

// 	cdef cppclass Worker[HandlerT]:
// 		pass

// 	cdef cppclass ThreadPool[HandlerT]:
// 		ThreadPool[Construct](size_t queue_length, size_t workers, Construct handler_construct,
// 		                      Request poison) except +

// 	cdef cppclass RecyclingHandler[HandlerT]:
// 		pass

// 	cdef cppclass RecyclingThreadPool[HandlerT]:
// 		void PopulateRecycling(const Request & request) except +
// 		HandlerT.Request Consume() except +
// 		void Produce(const Request & request) except +

// cdef extern from "util/tokenize_piece.hh" namespace "util":
// 	cdef cppclass Find:
// 		pass

// 	cdef bool SkipEmpty = False

// 	cdef cppclass TokenIter:
// 		pass


// cdef extern from "util/stream/sort.hh" namespace "util::stream":
// 	cdef cppclass Compare:
// 		pass
// 	cdef cppclass Combine:
// 		pass
// 	cdef cppclass Sorts(FixedArray):
// 		Sorts(size_t number) except +
// 		void push_back(Chain & chain, SortConfig & config, const Compare & compare = Compare(),
// 		               const Combine & combine = Combine()) except +


// cdef extern from "util/stream/chain.hh" namespace "util::stream":
// 	cdef cppclass Chain:
// 		Chain(const ChainConfig & config) except +
// 		size_t EntrySize() except +
// 		void SetProgressTarget(uint64_t target) except +
// 		void ActivateProgress() except +
// 		size_t BlockCount() except +
// 		ChainPosition Add() except +
// 		void Wait(bool release_memory = True) except +
// 		void Start() except +
// 		bool Running() except +

// 	cdef cppclass ChainPosition:
// 		Chain & GetChain() except +



// cdef extern from "lm/builder/adjust_counts.hh" namespace "util::stream":
// 	cdef struct DiscountConfig:
// 		vector[builder.Discount] overwrite
// 		# If discounting fails for an order, copy them from here.
// 		builder.Discount fallback
// 		# What to do when discounts are out of range or would trigger divison by
// 		# zero. It does something other than THROW_UP, use fallback_discount.
// 		constant.WarningAction bad_action


// cdef extern from "util/stream/multi_stream.hh" namespace "util::stream":
// 	cdef cppclass Chains(FixedArray):
// 		Chains() except +


// cdef extern from "util/stream/config.hh" namespace "util::stream":
// 	cdef struct SortConfig:
// 		# Filename prefix where temporary files should be placed.
// 		string temp_prefix
// 		# Size of each input/output buffer.
// 		size_t buffer_size
// 		# Total memory to use when running alone.
// 		size_t total_memory

// 	cdef struct ChainConfig:
// 		ChainConfig() except +
// 		ChainConfig(size_t in_entry_size, size_t in_block_count, size_t in_total_memory) except +

// 		size_t entry_size  # Number of bytes in each record.

// 		size_t block_count  # Number of blocks in the chain.

// 		# Total number of bytes available to the chain. This value will be divided amongst the blocks in the chain.
// 		# Chain's constructor will make this a multiple of entry_size.
// 		size_t total_memory

// cpdef class Py_ChainPosition:
//     cdef ChainPosition c_pos_