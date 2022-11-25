cimport constant as ct
cimport template as tp
cimport util


class Py_FilterConfig:
	cdef Config filter_cnf
	cdef size_t batch_size = 25000
	cdef size_t threads
	cdef bool phrase = False
	cdef bool context = False
	cdef ct.FilterMode mode
	cdef FormatEnum format

	def __cinit__(
		self,
		ct.Py_Format format_,
		ct.Py_FilterMode mode,
		size_t threads,
		size_t batch_size,
		bool phrase,
		bool context
	):
		pass

	def __dealloc__(self):
		del self


cdef void RunThreadedFilter(
	Config & config,
	util.FilePiece & in_lm,
	tp.Filter & filter_,
	tp.Output & output,
	dtype=[Format, OutputBuffer]
):
	if config.threads == 1:
		tp.Format.RunFilter(in_lm, filter_, output)
	else:
		ctypedef Controller[Filter, OutputBuffer, Output] Threaded

		cdef Threaded threading = Threaded(config.batch_size, config.threads * 2,
		                                   config.threads, filter_, output)

		tp.Format.RunFilter(in_lm, threading, output)


cdef void RunContextFilter(
	const Config & config,
	util.FilePiece & in_lm,
	tp.Filter filter_,
	tp.Output & output,
	dtype=[Format, OutputBuffer]
):
	if config.context:
		ContextFilter[Filter]
		context_filter = ContextFilter[Filter](filter_)
		RunThreadedFilter(
			config, in_lm, context_filter, output, dtype=dtype
		)
	else:
		RunThreadedFilter(
			config, in_lm, filter_, output, dtype=dtype
		)

cpdef void DispatchBinaryFilter(
	const Config & config,
	util.FilePiece & in_lm,
	const tp.Binary & binary,
	tp.Output & out,
	dtype=[Format, Binary]
):
	ctypedef BinaryFilter[Binary] Filter
	RunContextFilter(config, in_lm, Filter(binary), out, dtype=[Format, BinaryOutputBuffer])

cpdef void DispatchFilterModes(
	const Py_FilterConfig & config,
	util.istream & in_vocab,
	Py_FilePiece & in_lm,
	const char *out_name,
	ct.Format format_type=None
):
	# ctypedef unordered_map[string, vector[U_int]] Words
	cdef:
		Substrings substrings
		unordered_map[string, vector[uint64_t]] words

	if config.mode == ct.FilterMode.MODE_MULTIPLE:
		out = (out_name, ReadMultiple(in_vocab, substrings))

		if config.phrase:
			RunContextFilter(
				config, in_lm, PhraseFilter(substrings), out,
			    dtype=[format_type, MultipleOutputBuffer]
			)
		else:
			RunContextFilter(
				config, in_lm, VocabFilter(words), out,
			    dtype=[Format, MultipleOutputBuffer]
			)

	out(out_name)

	if config.mode == FilterMode.MODE_COPY:
		tp.Format.Copy(in_lm, out)
		return

	if config.mode == FilterMode.MODE_SINGLE:
		cdef Words words
		ReadSingle(in_vocab, words)
		DispatchBinaryFilter[Format, Single](config, in_lm, Single(words), out)
		return

	if config.mode == FilterMode.MODE_UNION:
		if config.phrase:
			cdef Substrings substrings
			ReadMultiple(in_vocab, substrings)
			DispatchBinaryFilter[Format, Union](config, in_lm, Union(substrings), out)
		else:
			cdef Words words
			ReadMultiple(in_vocab, words)
			DispatchBinaryFilter[Format, Union](config, in_lm, Union(words), out)

