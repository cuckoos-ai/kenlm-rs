use std::collections::HashMap;
use std::fs::{read_to_string, File};
use std::io::prelude::*;
use std::io::BufReader;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

use core::option::Option;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy)]
pub struct WordIndex;

#[derive(Debug, Clone, Copy)]
pub struct StringPiece {
    size_type: i8
}

impl StringPiece {
    pub fn new(inp_string: &str) -> Self;
}

#[derive(Debug, Clone, Copy)]
pub struct Model;

#[derive(Debug, Clone, Copy)]
pub struct Vocabulary;

#[derive(Debug, Clone, Copy)]
pub struct ModelBuffer;

#[derive(Debug)]
pub struct ProbBackoff {
    prob: f64,
    backoff: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NGram;

#[derive(Debug, Clone, Copy)]
pub struct EnumerateVocab;

#[derive(Debug, Clone, Copy)]
pub struct FullScoreReturn;

#[derive(Debug, Clone, Copy)]
pub struct Uninterpolated {
    // Uninterpolated probability.
    prob: f64,
    // Interpolation weight for lower order.
    gamma: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BuildingPayload {
    count: u64,

    uninterp: Uninterpolated,

    complete: ProbBackoff,
}

#[derive(Debug, Clone, Copy)]
pub struct ARPAOutput;

#[derive(Debug, Clone, Copy)]
pub struct CountOutput;

#[derive(Debug, Clone, Copy)]
pub struct MultipleOutput<Single>;

#[derive(Debug, Clone, Copy)]
pub struct ARPAFormat;

#[derive(Debug)]
pub struct CountFormat;

#[derive(Debug)]
pub struct Controller<Filter, OutputBuffer, RealOutput>;

#[derive(Debug)]
pub struct BinaryFilter<Binary>;

trait OutputBuffer {}
#[derive(Debug, Clone, Copy)]
pub struct BinaryOutputBuffer;

#[derive(Debug, Clone, Copy)]
pub struct MultipleOutputBuffer;

// Exceptions
#[derive(Debug)]
pub struct SpecialWordMissingException;

#[derive(Debug)]
struct ConfigException;

#[derive(Debug)]
struct LoadException;

#[derive(Debug)]
struct FormatLoadException;

#[derive(Debug)]
struct VocabLoadException;

impl OutputBuffer for BinaryOutputBuffer {}

impl OutputBuffer for MultipleOutputBuffer {}

trait Buffer {
    pub fn new(self) -> Self {}

    /// .
    pub fn Reserve(self, size: i64);

    /// .
    pub fn AddNGram(self, line: &StringPiece);
}

impl Buffer for MultipleOutputBuffer {
    fn single_add_ngram(self, offset: i64, line: &StringPiece) {}

    fn new(self) -> Self {}
}

trait Format<Filter, Out> {
    pub fn run_filter<Filter, Out>(self, in_fp: &FilePiece, filter_: &Filter, output: &Out);

    pub fn copy(self, in_fp: &FilePiece, out: &ARPAOutput);
}

impl Format for CountFormat {}

impl Format for ARPAFormat {}

impl Model {
    pub fn BeginSentenceWrite(self);

    pub fn NullContextWrite(self);

    pub fn Order(self) -> u64;

    pub fn BaseVocabulary(self) -> Vocabulary;

    pub fn BaseScore(self, in_state: T, new_word: WordIndex, out_state: T) -> f64;

    pub fn BaseFullScore(self, in_state: T, new_word: WordIndex, out_state: T) -> FullScoreReturn;
}

impl ModelBuffer {
    // Construct for writing.  Must call VocabFile() and fill it with null-delimited vocab words.
    // Load from file.
    pub fn new(file_base: &StringPiece, keep_buffer: bool, output_q: bool) -> Self {}

    // Must call VocabFile and populate before calling this function.
    pub fn sink(self, chains: &Chains, counts: Vec<u64>) -> Result<()>;

    pub fn model_buffer(self, file_base: &StringPiece) -> Result<()>;

    // # Read files and write to the given chains. If fewer chains are provided, only do the lower orders.
    pub fn source(self, chains: &Chains) -> Option<()>;

    pub fn source(self, order_minus_1: i8, chain: &Chain) -> Option<()>;

    // The order of the n-gram model that is associated with the model buffer.
    fn order() -> i8;

    // Requires Sink or load from file.
    fn counts() -> Vec<u64>;

    fn vocab_file(self) -> i64;

    fn raw_file(self, order_minus_1: i8) -> i64;

    // fn SlowQuery(self, context: &State, word: &WordIndex, out: &State) -> Result<()>;
    fn keep(self) -> bool;

    // Slowly execute a language model query with binary search.
    // This is used by interpolation to gather tuning probabilities rather than scanning the files.
    fn slow_query(self, context: &State, word: &WordIndex, out: &State) -> f64;
}

impl Vocabulary {
    fn index(self, id: &str) -> WordIndex;

    fn begin_sentence(self) -> Result<()>;

    fn end_sentence(self) -> Result<()>;

    fn not_found(self) -> Option<()>;
}

#[derive(Clone, Copy, Debug, Default)]
struct FullScoreReturn {
    prob: f64, // log10 probability

    // The length of n-gram matched.  Do not use this for recombination.
    // Consider a model containing only the following n-grams:
    // * -1 foo
    // * -3.14  bar
    // * -2.718 baz -5
    // * -6 foo bar
    //
    // If you score ``bar'' then ngram_length is 1 and recombination state is the
    // empty string because bar has zero backoff and does not extend to the right.
    // If you score ``foo'' then ngram_length is 1 and recombination state is ``foo''.
    //
    // Ideally, keep output states around and compare them.  Failing that,
    // get out_state.ValidLength() and use that length for recombination.
    ngram_length: String,

    // Left extension information.  If independent_left is set, then prob is
    // independent of words to the left (up to additional backoff).  Otherwise,
    // extend_left indicates how to efficiently extend further to the left.
    independent_left: bool,

    extend_left: u64,

    // Rest cost for extension to the left.
    rest: f64,
}

impl Default for FullScoreReturn {
    fn default() -> Self {
        Self {}
    }
}

trait Model {
    fn Recognize(&self, file: &str) -> Option<()> {}

    fn FullScore(&self, state: &State, word_index: &WordIndex, state: &State) -> FullScoreReturn {}

    /// .
    fn FullScoreForgotState(
        &self,
        context_rbegin: &WordIndex,
        context_rend: &WordIndex,
        new_word: &WordIndex,
        out_state: &State,
    ) -> FullScoreReturn;
}

impl EnumerateVocab {
    fn Add(self, index: WordIndex, string: &StringPiece) -> Result<()>;
}

impl BuildingPayload {
    pub fn is_marked(self) -> bool;

    pub fn mark(self);

    pub fn unmark(self);

    pub fn unmarked_count(self) -> u64;

    pub fn cutoff_count(self) -> u64;
}

impl NGram {
    pub fn total_size(self, order: i64) -> i64;

    pub fn order_from_size(self, size: i64) -> i64;
}

pub fn score_sentence(model: Model, sentence: &str) -> f64 {}

pub fn query_from_bytes(model: &Model, config: &Config) {}

pub fn convert_to_bytes(model: &Model, fd_in: i64) {}
