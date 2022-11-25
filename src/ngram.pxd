# namespace "lm::ngram"

cimport kenlm
cimport util
cimport ngram
cimport constant
cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t, uintptr_t, uint64_t
ctypedef uintptr_t size_t


cdef extern from "lm/state.hh" namespace "lm::ngram":
    cdef cppclass State:
        int Compare(const State & other) const

    int hash_value(const State & state) except +


cdef extern from "lm/config.hh" namespace "lm::ngram":

    cdef struct Config:
        # Config()
        float probing_multiplier
        util.LoadMethod load_method
        bool show_progress
        # Level of complaining to do when loading from ARPA instead of binary format.
        constant.ARPALoadComplain arpa_complain

        # Where to log messages including the progress bar.  Set to NULL for silence.
        util.ostream *messages

        util.ostream *ProgressMessages() const

        # This will be called with every string in the vocabulary by the constructor; it need
        # only exist for the lifetime of the constructor. See enumerate_vocab.hh for more detail.
        # Config does not take ownership; just delete/let it go out of scope after the constructor exits.
        kenlm.EnumerateVocab *enumerate_vocab

        # ONLY EFFECTIVE WHEN READING ARPA
        # What to do when <unk> isn't in the provided model.
        constant.WarningAction unknown_missing
        # What to do when <s> or </s> is missing from the model.
        # If THROW_UP, the exception will be of type util::SpecialWordMissingException.
        constant.WarningAction sentence_marker_missing

        # What to do with a positive log probability. For COMPLAIN and SILENT, map to 0.
        constant.WarningAction positive_log_probability

        # The probability to substitute for <unk> if it's missing from the model.
        # No effect if the model has <unk> or unknown_missing == THROW_UP.
        float unknown_missing_logprob

        # Size multiplier for probing hash table. Must be > 1. Space is linear in
        # this. Time is probing_multiplier / (probing_multiplier - 1). No effect for sorted variant.
        # If you find yourself setting this to a low number, consider using the TrieModel which has lower memory consumption.
        float probing_multiplier

        #  Amount of memory to use for building.  The actual memory usage will be higher since this
        # just sets sort buffer size.  Only applies to trie models.
        size_t building_memory

        # Template for temporary directory appropriate for passing to mkdtemp.
        # The characters XXXXXX are appended before passing to mkdtemp. Only applies to trie.
        # If empty, defaults to write_mmap. If that's NULL, defaults to input file name.
        string temporary_directory_prefix

        #  While loading an ARPA file, also write out this binary format file. Set to NULL to disable.
        const char *write_mmap

        constant.WriteMethod write_method

        # Include the vocab in the binary file?  Only effective if write_mmap != NULL.
        bool include_vocab

        constant.RestFunction rest_function  # Only used for REST_LOWER.

        vector[string] rest_lower_files

        # Quantization options.  Only effective for QuantTrieModel.  One value is
        # reserved for each of prob and backoff, so 2^bits - 1 buckets will be used
        # to quantize (and one of the remaining backoffs will be 0).
        uint8_t prob_bits
        uint8_t backoff_bits

        # Bhiksha compression (simple form).  Only works with trie.
        uint8_t pointer_bhiksha_bits


cdef extern from "lm/model.hh" namespace "lm::ngram":
    cdef Model *LoadVirtual(char *, Config & config) except +
    #default constructor
    cdef Model *LoadVirtual(char *) except +

    cdef cppclass GenericModel[Search, VocabularyT]:
        # This is the model type returned by RecognizeBinary.
        const ModelType kModelType

        const unsigned int kVersion = Search.kVersion
        GenericModel(const char *file, const Config & init_config) except +

        # Get the size of memory that will be mapped given ngram counts. This
        # does not includes small non-mapped control structures, such as this class itself.
        @staticmethod
        uint64_t Size(const vector[uint64_t] & counts, const Config & config = Config()) except +

        # Get the state for a context. Don't use this if you can avoid it. Use
        # BeginSentenceState or NullContextState and extend from those. If
        # you're only going to use this state to call FullScore once, use FullScoreForgotState.
        # To use this function, make an array of WordIndex containing the context
        # vocabulary ids in reverse order.  Then, pass the bounds of the array:
        # [context_rbegin, context_rend).
        void GetState(const kenlm.WordIndex *context_rbegin, const kenlm.WordIndex *context_rend, State & out_state) except +

        # Score p(new_word | in_state) and incorporate new_word into out_state.
        # Note that in_state and out_state must be different references:
        # &in_state != &out_state.
        kenlm.FullScoreReturn FullScore(
            const State & in_state,
            const kenlm.WordIndex new_word,
            State & out_state) except +

        # Slower call without in_state. Try to remember state, but sometimes it
        # would cost too much memory or your decoder isn't setup properly.
        # To use this function, make an array of WordIndex containing the context
        # vocabulary ids in reverse order.  Then, pass the bounds of the array:
        # [context_rbegin, context_rend).  The new_word is not part of the context
        # array unless you intend to repeat words.
        kenlm.FullScoreReturn FullScoreForgotState(
            const kenlm.WordIndex *context_rbegin, const kenlm.WordIndex *context_rend,
            const kenlm.WordIndex new_word, State & out_state) except +



cdef extern from "lm/quantize.hh" namespace "lm::ngram":

    cdef cppclass MiddlePointer:
        bool Found() except +
        float Prob() except +
        float Backoff() except +
        float Rest() except +
        void Write(float prob, float backoff) except +

    cppclass LongestPointer:
        pass

    cdef cppclass BinaryFormat:
        pass

    cdef cppclass DontQuantize:
        DontQuantize() except +

        @staticmethod
        void UpdateConfigFromBinary(const BinaryFormat &, uint64_t, Config &) except +

        @staticmethod
        uint64_t Size(uint8_t, const Config &) except +

        @staticmethod
        uint8_t MiddleBits(const Config &) except +

        @staticmethod
        uint8_t LongestBits(const Config &) except +

        void SetupMemory(void *, unsigned char, const Config &) except +

        void Train(uint8_t, vector[float] &, vector[float] &) except +

        void TrainProb(uint8_t, vector[float] &) except +

        void FinishedLoading(const Config &) except +

    cdef cppclass Bins:
        pass

    cdef cppclass SeparatelyQuantize:

        SeparatelyQuantize() except +

        void UpdateConfigFromBinary(const BinaryFormat &, uint64_t, Config &) except +

        @staticmethod
        uint64_t Size(uint8_t, const Config &) except +

        @staticmethod
        uint8_t MiddleBits(const Config &) except +

        @staticmethod
        uint8_t LongestBits(const Config &) except +

        void SetupMemory(void *, unsigned char, const Config &) except +

        # Assumes 0.0 is removed from backoff.
        void Train(uint8_t order, vector[float] & prob, vector[float] & backoff) except +
        # Train just probabilities (for longest order).
        void TrainProb(uint8_t order, vector[float] & prob) except +

        void FinishedLoading(const Config & config) except +

        const Bins *GetTables(unsigned char order_minus_2) except +

        Bins & LongestTable() except +


cdef extern from "lm/binary_format.cc" namespace "lm::ngram":
    cdef bool RecognizeBinary(const char *file, ModelType & recognized)
    cdef const char *kModelNames[6]


cdef extern from "lm/ngram_query.hh" namespace "lm::ngram":

    cdef cppclass QueryPrinter:
        QueryPrinter(int fd, bool print_word, bool print_line, bool print_summary, bool flush)

        void Word(util.StringPiece surface, kenlm.WordIndex vocab, const kenlm.FullScoreReturn & ret) except +

        void Line(uint64_t oov, float total) except +

        void Summary(double ppl_including_oov, double ppl_excluding_oov, uint64_t corpus_oov,
                     uint64_t corpus_tokens) except +

    cdef void Query[Model](const char *file, const Config & config, bool sentence_context,
                           QueryPrinter & printer) except +

    cdef void Query[Model, Printer](const Model & model, bool sentence_context, Printer & printer) except +


cdef extern from "lm/model_type.hh" namespace "lm::ngram":
    ctypedef enum ModelType:
        PROBING = 0
        REST_PROBING = 1
        TRIE = 2
        QUANT_TRIE = 3
        ARRAY_TRIE = 4
        QUANT_ARRAY_TRIE = 5

cdef extern from "lm/build_binary_main.cc" namespace "lm::ngram":
    ctypedef unsigned long int ulong

    cdef uint8_t ParseBitCount(const char *) except +
    cdef float ParseFloat(const char *) except +
    cdef np.npy_ulong ParseUInt(const char *) except +
    cdef void ParseFileList(const char *, vector[string] & to) except +


cdef extern from "lm/value.hh" namespace "lm::ngram":
    ctypedef struct RestValue:
        pass

    ctypedef struct BackoffValue:
        pass

cdef extern from "lm/search_hashed.hh" namespace "lm::ngram::detail":
    cdef cppclass HashedSearch[Value]:

        @staticmethod
        void UpdateConfigFromBinary(const BinaryFormat &, const vector[uint64_t] &, uint64_t, Config &) except +

        @staticmethod
        uint64_t Size(const vector[uint64_t] & counts, const Config & config) except +

        uint8_t *SetupMemory(uint8_t *start, const vector[uint64_t] & counts, const Config & config) except +

        void InitializeFromARPA(
            const char *file, util.FilePiece & f, vector[uint64_t] & counts, const Config & config,
            SortedVocabulary & vocab, BinaryFormat & backing) except +

        unsigned char Order() except +

        kenlm.ProbBackoff & UnknownUnigram() except +

        UnigramPointer LookupUnigram(kenlm.WordIndex word, Node &node, bool &independent_left,
                                     uint64_t &extend_left) except +

        MiddlePointer Unpack(uint64_t extend_pointer, unsigned char extend_length, Node & node) except +

        MiddlePointer LookupMiddle(
            unsigned char order_minus_2, kenlm.WordIndex word, Node & node,
            bool & independent_left, uint64_t & extend_left) except +

        LongestPointer LookupLongest(kenlm.WordIndex word, const Node & node) except +
        bool FastMakeNode(const kenlm.WordIndex *begin, const kenlm.WordIndex *end, Node & node) except +


cdef extern  from "lm/vocab.hh" namespace "lm::ngram":
    cdef cppclass SortedVocabulary:
        SortedVocabulary() except +
        kenlm.WordIndex Index(const util.StringPiece & str_) except +

    cdef cppclass ProbingVocabulary:
        ProbingVocabulary() except +
        kenlm.WordIndex Index(const util.StringPiece & str_) except +


cdef extern from "lm/trie.hh" namespace "lm::ngram::trie":
    cdef struct NodeRange:
        uint64_t begin, end

    cdef cppclass UnigramPointer:
        pass

    cdef cppclass Unigram:
        pass

    cdef cppclass BitPacked:
        pass

    cdef cppclass BitPackedMiddle[Bhiksha]:
        pass

    cdef cppclass BitPackedLongest:
        pass

cdef extern from "lm/bhiksha.hh" namespace "lm::ngram::trie":
    cdef class BhikshaBase:
        @staticmethod
        cdef void UpdateConfigFromBinary(BinaryFormat &file, uint64_t offset, Config &config):
            pass

        @staticmethod
        cdef uint64_t Size(uint64_t max_offset, uint64_t max_next, const Config &config):
            pass

        @staticmethod
        cdef uint8_t InlineBits(uint64_t max_offset, uint64_t max_next, const Config &config):
            pass

        cdef void ReadNext(self, void *base, uint64_t bit_offset, uint64_t index, uint8_t total_bits, NodeRange &out):
            pass

    cdef cppclass ArrayBhiksha(BhikshaBase):
        @staticmethod
        const ModelType kModelTypeAdd = kArrayAdd
        ArrayBhiksha(void *base, uint64_t max_offset, uint64_t max_value, const Config &config)

    cdef cppclass DontBhiksha(BhikshaBase):
        @staticmethod
        const ModelType kModelTypeAdd = kArrayAdd

        DontBhiksha(const void *base, uint64_t max_offset, uint64_t max_next, const Config &config)


cdef extern from "lm/search_trie.hh" namespace "lm::ngram::trie":

    ctypedef NodeRange Node
    cdef cppclass TrieSearch[Quant, Bhiksha]:

        @staticmethod
        void UpdateConfigFromBinary(const BinaryFormat & file, const vector[uint64_t] & counts,
                                    uint64_t offset, Config & config) except +

        @staticmethod
        uint64_t Size(const vector[uint64_t] & counts, const Config & config) except +

        uint8_t *SetupMemory(uint8_t *start, const vector[uint64_t] & counts, const Config & config) except +

        void InitializeFromARPA(
            const char *file, util.FilePiece & f, vector[uint64_t] & counts,
            const Config & config, SortedVocabulary & vocab, BinaryFormat & backing) except +

        UnigramPointer LookupUnigram(kenlm.WordIndex word, Node &node, bool &independent_left,
                                     uint64_t &extend_left) except +

        kenlm.ProbBackoff & UnknownUnigram() except +

        MiddlePointer Unpack(uint64_t extend_pointer, unsigned char extend_length, Node & node) except +

        MiddlePointer LookupMiddle(
			unsigned char order_minus_2, kenlm.WordIndex word, Node & node,
	        bool & independent_left, uint64_t & extend_left) except +

    LongestPointer LookupLongest(kenlm.WordIndex word, const Node & node) except +
    bool FastMakeNode(const kenlm.WordIndex *begin, const kenlm.WordIndex *end, Node & node) except +


cdef extern from "lm/sizes.cc" namespace "lm::ngram" nogil:
    cdef void ShowSizes(const char *file, const Config & config) except +