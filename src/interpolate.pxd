cimport util
cimport kenlm
from libcpp.vector cimport vector


cdef extern from "lm/interpolate/tune_weights.cc" namespace "lm::interpolate":
    cdef void TuneWeights(int tune_file, const vector[util.StringPiece] & model_names,
        const InstancesConfig & config, vector[float] & weights_out) except +

cdef extern from "lm/interpolate/pipeline.hh" namespace "lm::interpolate":
    cdef struct Config:
        vector[float] lambdas
        util.SortConfig sort
        size_t BufferSize() const

cdef extern from "lm/interpolate/tune_instances.hh" namespace "lm::interpolate":
    cdef struct InstancesConfig:
        # For batching the model reads.  This is per order.
        size_t model_read_chain_mem
        # This is being sorted, make it larger.
        size_t extension_write_chain_mem
        size_t lazy_memory
        util.SortConfig sort

cdef extern from "lm/interpolate/pipeline.cc" namespace "lm::interpolate":
    cdef void Pipeline(util.FixedArray[kenlm.ModelBuffer] & models, const Config & config, int write_file) except +

cdef extern from "lm/interpolate/split_worker.hh" namespace "lm::interpolate":
    cdef cppclass SplitWorker:
        # Constructs a split worker for a particular order. It writes the
        # split-off backoff values to the backoff chain and the ngram id and
        # probability to the sort chain for each ngram in the input.
        SplitWorker(size_t order, util.Chain &backoff_chain, util.Chain &sort_chain)

        # The callback invoked to handle the input from the ngram intermediate files.
        cdef void Run(const util.ChainPosition& position)