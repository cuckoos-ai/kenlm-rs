// cimport util
// cimport ngram
// from libcpp cimport bool
// from libcpp.string cimport string
// from libcpp.vector cimport vector
// from libc.stdint cimport  uintptr_t, uint64_t


// cdef extern from "lm/word_index.hh" namespace "lm":
//     ctypedef unsigned WordIndex

// cdef extern from "lm/return.hh" namespace "lm":
//     cdef struct FullScoreReturn:
//         float prob
//         unsigned char ngram_length


// cdef extern from "lm/virtual_interface.hh" namespace "lm::base" nogil:
//     cdef cppclass Vocabulary:
//         WordIndex Index(char *) except +
//         WordIndex BeginSentence() except +
//         WordIndex EndSentence() except +
//         WordIndex NotFound() except +

//     ctypedef Vocabulary const_Vocabulary

//     cdef cppclass Model:
//         void BeginSentenceWrite(void *)

//         void NullContextWrite(void *)

//         unsigned int Order()

//         const_Vocabulary& BaseVocabulary()

//         float BaseScore(void *in_state, WordIndex new_word, void *out_state)

//         FullScoreReturn BaseFullScore(void *in_state, WordIndex new_word, void *out_state)




// cdef extern from "lm/return.hh" namespace "lm":
//     cdef struct FullScoreReturn:
//         float prob  # log10 probability

//         # The length of n-gram matched.  Do not use this for recombination.
//         # Consider a model containing only the following n-grams:
//         # * -1 foo
//         # * -3.14  bar
//         # * -2.718 baz -5
//         # * -6 foo bar
//         #
//         # If you score ``bar'' then ngram_length is 1 and recombination state is the
//         # empty string because bar has zero backoff and does not extend to the right.
//         # If you score ``foo'' then ngram_length is 1 and recombination state is ``foo''.
//         # 
//         # Ideally, keep output states around and compare them.  Failing that,
//         # get out_state.ValidLength() and use that length for recombination.
//         unsigned char ngram_length

//         # Left extension information.  If independent_left is set, then prob is
//         # independent of words to the left (up to additional backoff).  Otherwise,
//         # extend_left indicates how to efficiently extend further to the left.
//         bool independent_left
//         uint64_t extend_left  # Defined only if independent_left

//         # Rest cost for extension to the left.
//         float rest


// cdef extern from "lm/wrappers/nplm.hh" namespace "lm::np":
//     cdef cppclass Model:
//         # Does this look like an NPLM
//         @staticmethod
//         bool Recognize(const string & file) except +

//         FullScoreReturn FullScore(const State &, const WordIndex &, State &) except +

//         FullScoreReturn FullScoreForgotState(
//             const WordIndex *context_rbegin, const WordIndex *context_rend,
//             const WordIndex new_word, State & out_state ) except +


// cdef extern  from "lm/weights.hh" namespace "lm":
//     ctypedef struct ProbBackoff:
//         float prob
//         float backoff

// kArrayAdd = None

// cdef extern from "python/score_sentence.hh" namespace "lm::base":
//     cdef float ScoreSentence(const Model *model, const char *sentence) except +


// cdef extern from "lm/enumerate_vocab.hh" namespace "lm":
//     cdef cppclass EnumerateVocab:
//         void Add(WordIndex index, const util.StringPiece & string_) except +


// cdef extern from "lm/return.hh" namespace "lm":
//     cdef struct FullScoreReturn:
//         # log10 probability
//         float prob

//         # * The length of n-gram matched.  Do not use this for recombination.
//         # * Consider a model containing only the following n-grams:
//         # * -1 foo
//         # * -3.14  bar
//         # * -2.718 baz -5
//         # * -6 foo bar
//         # *
//         # * If you score ``bar'' then ngram_length is 1 and recombination state is the
//         # * empty string because bar has zero backoff and does not extend to the
//         # * right.
//         # * If you score ``foo'' then ngram_length is 1 and recombination state is
//         # * ``foo''.
//         # *
//         # * Ideally, keep output states around and compare them.  Failing that,
//         # * get out_state.ValidLength() and use that length for recombination.
//         unsigned char ngram_length

//         # * Left extension information.  If independent_left is set, then prob is
//         # * independent of words to the left (up to additional backoff).  Otherwise,
//         # * extend_left indicates how to efficiently extend further to the left.
//         bool independent_left

//         # Defined only if independent_left
//         uint64_t extend_left

//         # Rest cost for extension to the left.
//         float rest


// cdef extern from "<Eigen/Core>" namespace "Eigen":
//     cdef void initParallel() except +


// cdef extern from "lm/common/model_buffer.hh" namespace "lm":
//     cdef cppclass ModelBuffer:
//         # Construct for writing.  Must call VocabFile() and fill it with null-delimited vocab words.
//         ModelBuffer(util.StringPiece file_base, bool keep_buffer, bool output_q) except +

//         # Load from file.
//         void ModelBuffer(util.StringPiece file_base) except +

//         # Must call VocabFile and populate before calling this function.
//         void Sink(util.Chains & chains, const vector[uint64_t] & counts) except +

//         # Read files and write to the given chains. If fewer chains are provided, only do the lower orders.
//         void Source(util.Chains & chains) except +

//         void Source(size_t order_minus_1, Chain & chain) except +

//         # The order of the n-gram model that is associated with the model buffer.
//         size_t Order() const
//         # Requires Sink or load from file.
//         const vector[uint64_t] & Counts() const

//         int VocabFile() const

//         int RawFile(size_t order_minus_1) const

//         bool Keep() const

//         # Slowly execute a language model query with binary search.
//         # This is used by interpolation to gather tuning probabilities rather than scanning the files.
//         float SlowQuery(const State & context, WordIndex word, State & out) const


// cdef extern from "lm/lm_exception.hh" namespace "lm":

//     cdef cppclass ConfigException:
//         ConfigException() except +

//     cdef cppclass LoadException:
//         LoadException() except +

//     cdef cppclass FormatLoadException:
//         FormatLoadException() except +

//     cdef cppclass VocabLoadException:
//         VocabLoadException() except +

//     cdef cppclass SpecialWordMissingException:
//         SpecialWordMissingException() except +


// cdef extern from "lm/builder/payload.hh" namespace "lm::builder":
//     ctypedef struct Uninterpolated:
//         # Uninterpolated probability.
//         float prob
//         # Interpolation weight for lower order.
//         float gamma

//     ctypedef struct BuildingPayload:
//         uint64_t count
//         Uninterpolated uninterp
//         ProbBackoff complete

//         bool IsMarked() const

//         void Mark() except +

//         void Unmark() except +

//         uint64_t UnmarkedCount() const

//         uint64_t CutoffCount() const


// cdef extern from "lm/kenlm_benchmark_main.cc" namespace "lm":
//     cdef QueryFromBytes(const Model & model, const Config & config) except +
//     cdef ConvertToBytes(const Model & model, int fd_in) except +


// cdef extern from "lm/common/ngram.hh" namespace "lm":
//     cdef cppclass NGram:

//         @staticmethod
//         size_t TotalSize(size_t order) except +

//         @staticmethod
//         size_t OrderFromSize(size_t size) except +


// cdef extern from "<boost/range/iterator_range.hpp>" namespace "boost":
//     cdef cppclass iterator_range[IteratorT]:
//         pass


// cdef extern from "lm/filter/arpa_io.hh" namespace "lm":
//     cdef cppclass ARPAOutput:
//         pass

// cdef extern from "lm/filter/count_io.hh" namespace "lm":
//     cdef cppclass CountOutput:
//         pass

// cdef extern from "lm/filter/format.hh" namespace "lm":

//     cdef cppclass MultipleOutput[Single]:
//         pass

//     cdef struct ARPAFormat:
//         @staticmethod
//         void Copy(util.FilePiece &in_, ARPAOutput &out) except +

//         @staticmethod
//         void RunFilter[Filter, Out](util.FilePiece &in_, Filter &filter_, Out &output) except +

//     cdef struct CountFormat:
//         @staticmethod
//         void Copy(util.FilePiece &in_, Output &out_) except +

//         @staticmethod
//         void RunFilter[Filter, Out](util.FilePiece &in_, Filter &filter_, Out &output) except +


// cdef extern from "lm/filter/wrapper.hh" namespace "lm":

//     cdef cppclass BinaryFilter[Binary]:
//         pass

// cdef extern from "lm/filter/format.hh" namespace "lm":
//     cdef cppclass BinaryOutputBuffer:
//         void Reserve(size_t size) except +

//         void AddNGram(const util.StringPiece & line) except +

//     cdef cppclass MultipleOutputBuffer:
//         MultipleOutputBuffer() except +

//         void Reserve(size_t size) except +

//         void AddNGram(const util.StringPiece &line) except +

//         void SingleAddNGram(size_t offset, const util.util.StringPiece &line) except +


// cdef extern from "lm/filter/thread.hh" namespace "lm":
//     cdef cppclass Controller[Filter, OutputBuffer, RealOutput]:
//         pass

// cpdef class Py_QueryPrinter:
//     cdef:
//         ngram.QueryPrinter _query_p
//         int fd
//         bool print_word
//         bool print_line
//         bool print_summary