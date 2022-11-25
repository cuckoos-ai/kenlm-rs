cimport kenlm
from util import Py_ChainPosition
from interpolate import Py_SortConfig
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t


cpdef class Py_Discount:
    cdef Discount _discount
    def __init__(self, int[:] amount):
        self._discount = Discount(amount)

    def __del__(self):
        del self._discount

    def Get(self, int count) -> float:
        return self._discount.Get(count)

    def Apply(self, int count) -> float:
        return self._discount.Apply(count)


cpdef class Py_CorpusCount:
    cdef CorpusCount *_counter

    def __cinit__(
            self,
            FilePiece & _from,
            int vocab_write,
            bool dynamic_vocab,
            uint64_t & token_count,
            kenlm.WordIndex & type_count,
            vector[bool] & prune_words,
            const string& prune_vocab_filename,
            size_t entries_per_block):
        self._counter = CorpusCount(
            _from, vocab_write, dynamic_vocab, token_count, type_count,
            prune_vocab_filename, entries_per_block)

    cpdef Run(self, Py_ChainPosition *pos):
        self._counter.Run(pos.c_pos_)

    cpdef size_t VocabUsage(self, int vocab_estimate):
        return self._counter.VocabUsage(vocab_estimate)

    cpdef float DedupeMultiplier(self, int order):
        return self._counter.DedupeMultiplier(order)

    def __delloc__(self):
        del self._counter

class Py_InitialProbabilitiesConfig:
    cdef:
        ChainConfig adder_in
        ChainConfig adder_out
        # SRILM doesn't normally interpolate unigrams.
        bool _interpolate_unigrams
        bool _adder_in
        bool _adder_out

    def __init__(self, bool interpolate_unigrams = False):
        self._adder_out = None
        self._adder_in = None
        self._interpolate_unigrams = interpolate_unigrams

    @property
    def interpolate_unigrams(self):
        return self._interpolate_unigrams
    @interpolate_unigrams.setter
    def interpolate_unigrams(self, v):
        self._interpolate_unigrams = v

    @property
    def adder_in(self):
        return self._adder_in
    @adder_in.setter
    def adder_in(self, v):
        self._adder_in = v

    @property
    def adder_out(self):
        return self._adder_out
    @adder_out.setter
    def adder_out(self, v):
        self._adder_out = v


class Py_PipelineConfig:
    cdef:
        PipelineConfig __p_config
        size_t order
        Py_SortConfig sort_cfg
        Py_InitialProbabilitiesConfig initial_probs
        ChainConfig read_backoffs
        kenlm.WordIndex vocab_estimate
        uint64_t vocab_size_for_unk
        size_t minimum_block
        size_t block_count

        vector[uint64_t] prune_thresholds  # mjd
        bool prune_vocab
        string prune_vocab_file
        bool renumber_vocabulary
        DiscountConfig discount
        bool output_q

    def __init__(
            self,
            Py_InitialProbabilitiesConfig initial_probs,
            Py_SortConfig sort_cfg,
            size_t minimum_block,
            size_t block_count,
            int vocab_estimate,
            uint64_t vocab_size_for_unk,
            bool renumber_vocabulary,
            bool prune_vocab_file,
            bool output_q,
            # disallowed_symbol_action=catch_error
    ):
        self.initial_probs = initial_probs
        self.sort_cfg = sort_cfg
        self.minimum_block = minimum_block
        self.block_count = block_count
        self.vocab_estimate = vocab_estimate
        self.vocab_size_for_unk = vocab_size_for_unk
        self.renumber_vocabulary = renumber_vocabulary
        self.prune_vocab_file = prune_vocab_file
        self.output_q = output_q

    def __getattr__(self, str key):
        # assert key in self
        return self.__p_config[key]

    def __setattr__(self, str key, value):
        # TODO: Check if field present
        self.__p_config[key] = value

    @property
    def config(self):
        return self.__p_config

cpdef list Py_ParsePruning(str param, Py_ssize_t order):
    cdef vector[uint64_t] res
    res = ParsePruning(param, order)
    return res

cpdef Py_Discount Py_ParseDiscountFallback(str param):
    cdef Discount discount = ParseDiscountFallback(param)
    return discount
