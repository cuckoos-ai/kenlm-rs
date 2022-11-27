use crate::{kenlm::WordIndex};
use crate::constant::WarningAction;
use crate::util::{Chains, FilePiece, SortConfig, ChainConfig, ChainPosition, StringPiece};



#[derive(Debug, Default)]
pub struct Discount {
    amount: &[i64; 4],
}

impl Discount {
    fn new() -> Self {
        Discount { amount: () }
    }

    pub fn Get(&self, count: u64) -> f64;
    pub fn Apply(&self, count: u64) -> f64;
}


#[derive(Debug)]
pub struct HeaderInfo {
    input_file: &str,
    token_count: u64,
    counts_pruned: Vec<u64>
    
}

impl HeaderInfo {
    fn new(input_file_in: &str, token_count_in: u64, counts_pruned_in: &Vec<u64>) -> Self {
        HeaderInfo { input_file: input_file_in, token_count: token_count_in, counts_pruned: counts_pruned_in }
    }
}

#[derive(Debug, Clone)]
pub struct OutputHook;


impl OutputHook{
    pub fn new(hook_type: HookType) -> Self;
    pub fn Sink(&self, info: &HeaderInfo, vocab_file: i64, chains: &Chains);
    pub fn Type(&self) -> HookType;
}

#[derive(Debug, Clone)]
pub struct Output;

impl Output {
    pub fn new(file_base: StringPiece, keep_buffer: bool, output_q: bool ) -> Self;
    
    pub fn Add(&self, outhook: OutputHook);

    pub fn Have(&self, hook_type: HookType) -> bool;

    pub fn VocabFile(&self) -> i64;

    pub fn SetHeader(&self, header: &HeaderInfo);
    
    pub fn GetHeader(&self) -> &HeaderInfo;

    // This is called by the pipeline.
    pub fn SinkProbs(&self, chains: &Chains);
    
    pub fn Steps(&self) -> u64;
}


#[derive(Debug, Clone)]
pub struct CorpusCount {
    token_count: i64
}

impl CorpusCount {
    // Memory usage will be DedupeMultipler(order) * block_size + total_chain_size + unknown vocab_hash_size
    fn DedupeMultiplier(order: i8) -> f64;

    // How much memory vocabulary will use based on estimated size of the vocab.
    fn VocabUsage(&self, vocab_estimate: i8) -> i8;

    // type_count aka vocabulary size.  Initialize to an estimate.  It is set to the exact value.
    fn new(
        from_: &FilePiece,
        vocab_write: i64,
        dynamic_vocab: bool,
        token_count: &u64,
        type_count: &WordIndex,
        prune_words: &Vec<bool>,
        prune_vocab_filename: &str,
        entries_per_block: i8,
        disallowed_symbol: WarningAction
    ) -> Self;

    fn Run(&self, position: &ChainPosition);
}

#[derive(Debug, Clone)]
struct InitialProbabilitiesConfig{
    // These should be small buffers to keep the adder from getting too far ahead
    adder_in: ChainConfig ,
    adder_out: ChainConfig ,
    // SRILM doesn't normally interpolate unigrams.
    interpolate_unigrams: bool
}

#[derive(Debug, Clone)]
pub struct DiscountConfig {
    // Overrides discounts for orders [1,discount_override.size()].
    overwrite: Vec<Discount,
    // If discounting fails for an order, copy them from here.
    fallback: Discount,
    //  What to do when discounts are out of range or would trigger divison by
    //  zero. It does something other than THROW_UP, use fallback_discount.
    bad_action: WarningAction
}


#[derive(Debug, Default)]
pub struct PipelineConfig {
    order: i8,
    sort: SortConfig,
    initial_probs: InitialProbabilitiesConfig,
    read_backoffs: ChainConfig,

    // Estimated vocabulary size.  Used for sizing CorpusCount memory and initial probing hash table sizing, also in CorpusCount.
    vocab_estimate: WordIndex,

    // Minimum block size to tolerate.
    minimum_block: i8,

    // Number of blocks to use.  This will be overridden to 1 if everything fits.
    block_count: i8,

    // n-gram count thresholds for pruning. 0 values means no pruning for corresponding n-gram order
    prune_thresholds: Vec<i64>,  // mjd
    prune_vocab: bool,
    prune_vocab_file: &str,

    // Renumber the vocabulary the way the trie likes it?
    renumber_vocabulary: bool,

    // What to do with discount failures.
    discount: DiscountConfig,

    // Compute collapsed q values instead of probability and backoff
    output_q: bool

    // * Computing the perplexity of LMs with different vocabularies is hard.  For
    // * example, the lowest perplexity is attained by a unigram model that
    // * predicts p(<unk>) = 1 and has no other vocabulary.  Also, linearly
    // * interpolated models will sum to more than 1 because <unk> is duplicated
    // * (SRI just pretends p(<unk>) = 0 for these purposes, which makes it sum to
    // * 1 but comes with its own problems).  This option will make the vocabulary
    // * a particular size by replicating <unk> multiple times for purposes of
    // * computing vocabulary size.  It has no effect if the actual vocabulary is
    // * larger.  This parameter serves the same purpose as IRSTLM's "dub".

    vocab_size_for_unk: u64,

    // What to do the first time <s>, </s>, or <unk> appears in the input. If this is
    // anything but THROW_UP, then the symbol will always be treated as whitespace.
    disallowed_symbol_action: WarningAction,

}

impl PipelineConfig {
    fn new() -> Self;
    fn TempPrefix(self) -> &str;
    fn TotalMemory() -> i8;
}

pub fn Pipeline(config: &PipelineConfig, text_file: i64, output: &Output);

pub fn ParseDiscountFallback(param: &Vec<String>) -> Discount;

pub fn ParsePruning(param: &Vec<String>, order: i8) -> Vec<u64>;
