
cimport kenlm
cimport constant
from libc.stdint cimport uint64_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from util cimport StringPiece, Chains, FilePiece, SortConfig, ChainConfig, ChainPosition


cdef extern from "lm/builder/discount.hh" namespace "lm::builder":
    cdef struct Discount:
        float amount[4]
        float Get(uint64_t count) const
        float Apply(uint64_t count) const


cdef extern from "lm/builder/header_info.hh" namespace "lm::builder":
    cdef struct HeaderInfo:
        string input_file
        uint64_t token_count
        vector[uint64_t] counts_pruned
        HeaderInfo() except +
        HeaderInfo(const string& input_file_in, uint64_t token_count_in,
                   const vector[uint64_t] & counts_pruned_in) except +


cdef extern from "lm/builder/output.hh" namespace "lm::builder":
    cdef enum HookType:
        # Probability and backoff (or just q). Output must process the orders in
        # parallel or there will be a deadlock.
        PROB_PARALLEL_HOOK,
        # Probability and backoff (or just q). Output can process orders any way it likes.
        # This requires writing the data to disk then reading.  Useful for ARPA files, which put unigrams first etc.
        PROB_SEQUENTIAL_HOOK,
        # Keep this last so we know how many values there are.
        NUMBER_OF_HOOKS

    cdef cppclass OutputHook:
        OutputHook(HookType hook_type) except +
        # void Sink(const HeaderInfo &info, int vocab_file, Chains &chains) = 0
        # HookType Type() const

    cdef cppclass Output:
        Output(StringPiece file_base, bool keep_buffer, bool output_q) except +
        void Add(OutputHook *) except +

        bool Have(HookType hook_type) except +

        int VocabFile() except +

        void SetHeader(const HeaderInfo & header) except +
        const HeaderInfo & GetHeader() except +

        # This is called by the pipeline.
        void SinkProbs(Chains & chains) except +
        unsigned int Steps() except +


cdef extern from "lm/builder/corpus_count.hh" namespace "lm::builder":
    cdef cppclass CorpusCount:
        # Memory usage will be DedupeMultipler(order) * block_size + total_chain_size + unknown vocab_hash_size
        @staticmethod
        float DedupeMultiplier(size_t order) except +

        # How much memory vocabulary will use based on estimated size of the vocab.
        @staticmethod
        size_t VocabUsage(size_t vocab_estimate) except +

        # token_count: out.
        # type_count aka vocabulary size.  Initialize to an estimate.  It is set to the exact value.
        CorpusCount(
            FilePiece & from_,
            int vocab_write,
            bool dynamic_vocab,
            uint64_t &token_count,
            kenlm.WordIndex &type_count,
            vector[bool] &prune_words,
            const string &prune_vocab_filename,
            size_t entries_per_block,
            constant.WarningAction disallowed_symbol) except +

        void Run(const ChainPosition &position) except +


cdef extern from "lm/builder/adjust_counts.hh" namespace "lm::builder":
    cdef struct DiscountConfig:
        # Overrides discounts for orders [1,discount_override.size()].
        vector[Discount] overwrite
        # If discounting fails for an order, copy them from here.
        Discount fallback
        # What to do when discounts are out of range or would trigger divison by
        # zero. It does something other than THROW_UP, use fallback_discount.
        constant.WarningAction bad_action;


cdef extern from "lm/builder/pipeline.hh" namespace "lm::builder":
    cdef struct PipelineConfig:
        size_t order
        SortConfig sort
        InitialProbabilitiesConfig initial_probs
        ChainConfig read_backoffs

        # Estimated vocabulary size.  Used for sizing CorpusCount memory and initial probing hash table sizing, also in CorpusCount.
        kenlm.WordIndex vocab_estimate

        # Minimum block size to tolerate.
        size_t minimum_block

        # Number of blocks to use.  This will be overridden to 1 if everything fits.
        size_t block_count

        # n-gram count thresholds for pruning. 0 values means no pruning for corresponding n-gram order
        vector[uint64_t] prune_thresholds;  # mjd
        bool prune_vocab
        string prune_vocab_file

        # Renumber the vocabulary the way the trie likes it?
        bool renumber_vocabulary

        # What to do with discount failures.
        DiscountConfig discount

        # Compute collapsed q values instead of probability and backoff
        bool output_q

        # * Computing the perplexity of LMs with different vocabularies is hard.  For
        # * example, the lowest perplexity is attained by a unigram model that
        # * predicts p(<unk>) = 1 and has no other vocabulary.  Also, linearly
        # * interpolated models will sum to more than 1 because <unk> is duplicated
        # * (SRI just pretends p(<unk>) = 0 for these purposes, which makes it sum to
        # * 1 but comes with its own problems).  This option will make the vocabulary
        # * a particular size by replicating <unk> multiple times for purposes of
        # * computing vocabulary size.  It has no effect if the actual vocabulary is
        # * larger.  This parameter serves the same purpose as IRSTLM's "dub".

        uint64_t vocab_size_for_unk

        # What to do the first time <s>, </s>, or <unk> appears in the input. If this is
        # anything but THROW_UP, then the symbol will always be treated as whitespace.
        constant.WarningAction disallowed_symbol_action

        const string & TempPrefix() const

        size_t TotalMemory() const

    cdef void Pipeline(PipelineConfig & config, int text_file, Output & output) except +


cdef extern from "lm/builder/lmplz_main.cc" namespace "lm::builder":
    cdef Discount ParseDiscountFallback(const vector[string] & param) except +

    cdef vector[uint64_t] ParsePruning(const vector[string] & param, size_t order) except +


cdef extern from "lm/builder/initial_probabilities.hh" namespace "lm::builder":
    cdef struct InitialProbabilitiesConfig:
        # These should be small buffers to keep the adder from getting too far ahead
        ChainConfig adder_in
        ChainConfig adder_out
        # SRILM doesn't normally interpolate unigrams.
        bool interpolate_unigrams

