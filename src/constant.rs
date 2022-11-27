use core::option::Option;
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub(crate) enum ARPALoadComplain {
    ALL = 1,
    EXPENSIVE = 2,
    NONE = 3,
}

#[derive(Debug)]
pub enum WriteMethod {
    WRITE_MMAP = 1,  // Map the file directly.
    WRITE_AFTER = 2, // Write after we're done.
}

// Left rest options. Only used when the model includes rest costs.
#[derive(Debug)]
pub enum RestFunction {
    REST_MAX,   // Maximum of any score to the left
    REST_LOWER, // Use lower-order files given below.
}

pub(crate) enum FilterMode {
    MODE_COPY = 1,
    MODE_SINGLE = 2,
    MODE_MULTIPLE = 3,
    MODE_UNION = 4,
    MODE_UNSET = 5,
}

#[derive(Debug)]
pub(crate) enum WarningAction {
    THROW_UP = 1,
    COMPLAIN = 2,
    SILENT = 3,
}

// ------------------
// Constant PYX below
// ------------------
#[derive(Debug)]
pub(crate) enum ModelType {
    PROBING,
    REST_PROBING,
    TRIE,
    QUANT_TRIE,
    ARRAY_TRIE,
    QUANT_ARRAY_TRIE,
}

#[derive(Debug)]
pub(crate) enum Format {
    FORMAT_ARPA,
    FORMAT_COUNT,
}

#[derive(Debug)]
pub(crate) enum LoadMethod {
    LAZY = 1,
    POPULATE_OR_LAZY = 2,
    POPULATE_OR_READ = 3,
    READ = 4,
    PARALLEL_READ = 5,
}

#[derive(Debug)]
pub(crate) enum FormatEnum {
    FORMAT_ARPA,
    FORMAT_COUNT,
}

#[derive(Debug)]
pub(crate) enum HookType {
    // Probability and backoff (or just q). Output must process the orders in
    // parallel or there will be a deadlock.
    PROB_PARALLEL_HOOK,
    // Probability and backoff (or just q). Output can process orders any way it likes.
    // This requires writing the data to disk then reading.  Useful for ARPA files, which put unigrams first etc.
    PROB_SEQUENTIAL_HOOK,
    // Keep this last so we know how many values there are.
    NUMBER_OF_HOOKS,
}
