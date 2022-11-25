cimport constant as ct
cimport util as c_util
cimport template as tp
from util import Py_FilePiece

# ctypedef Format.Output FormatOutput
# ctypedef Format.Multiple FormatMultiple

cdef struct Config:
	size_t batch_size = 25000
	size_t threads = boost.thread.hardware_concurrency()
	FilterMode mode = MODE_COPY
	bool phrase = False
	bool context = False
	FormatEnum format = FORMAT_ARPA

cpdef class Py_FilterConfig:
	cdef Config filter_cnf
	cdef size_t batch_size = 25000
	cdef size_t threads = None
	cdef bool phrase = False
	cdef bool context = False
	cdef ct.FilterMode mode
	cdef FormatEnum format = FORMAT_ARPA

ctypedef enum FilterMode:
	MODE_COPY, MODE_SINGLE, MODE_MULTIPLE, MODE_UNION, MODE_UNSET

ctypedef enum FormatEnum:
	FORMAT_ARPA, FORMAT_COUNT

cdef void RunThreadedFilter(
	Config & config, c_util.FilePiece & in_lm, tp.Filter & filter_,
	tp.Output & output, dtype=[Format, OutputBuffer])

cdef void RunContextFilter(
	Config & config, c_util.FilePiece & in_lm, tp.Filter filter_,
	tp.Output & output, dtype=[Format, OutputBuffer])

cdef void DispatchBinaryFilter(
	Config & config, c_util.FilePiece & in_lm, const tp.Binary & binary,
	tp.Output & out, dtype=[Format, Binary])

cpdef void DispatchFilterModes(
	Py_FilterConfig & config, c_util.istream & in_vocab,
	Py_FilePiece & in_lm, const char *out_name, ct.Format format_type=None
)
