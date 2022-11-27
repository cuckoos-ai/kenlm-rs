
use numpy::ndarray::*;
use crate::builder::Output;
use crate::interpolate::SplitWorker;
use crate::ngram::*;
use crate::kenlm::{StringPiece};
use std::cell::RefCell;
use std::io::prelude::*;

pub(crate) struct FixedArray<T>;

type FixedArrayWorker = FixedArray<SplitWorker>;

impl FixedArray<T>{
    pub fn begin() -> T;

    pub fn end() -> T;
    
    pub fn back() -> T;
    
    pub fn size() -> u8;

    pub fn empty() -> bool;
    
    pub fn push_back(self);
    
    pub fn pop_back(self);
    
    pub fn clear(self);
}


#[derive(Debug, Clone)]
pub struct LineIterator;

#[derive(Debug, Clone)]
pub(crate) struct FilePiece;

impl Default for FilePiece {
    fn default() -> Self {
        
    }
}

#[derive(Debug, Clone)]
pub struct Model;

#[derive(Debug, Clone)]
pub struct Width;

#[derive(Debug)]
pub struct BenchmarkConfig{
    fd_in: i8,
    threads: i8,
    buf_per_thread: i64,
    query: bool
}

#[derive(Debug, Clone, Copy)]
pub struct IStream;

#[derive(Debug, Clone, Copy)]
pub struct OStream;

#[derive(Debug)]
pub struct SortConfig {
    
}

impl SortConfig {
    
}

impl FilePiece {
    // Takes ownership of fd.  name is used for messages.
    pub fn new(file: &str, stream: IStream, show_progress: OStream, min_buffer: &i8) -> Self;

    // Read from an istream. Don't use this if you can avoid it. Raw fd IO is
    // much faster. But sometimes you just have an istream like Boost's HTTP
    // server and want to parse it the same way. name is just used for messages and FileName().

    pub fn begin(self) -> LineIterator;
    pub fn end(self) -> LineIterator;
    pub fn peek(self) -> &str;

    // Leaves the delimiter, if any, to be returned by get().  Delimiters defined by isspace().
    pub fn ReadDelimited(self, delim: bool) -> StringPiece;

    // Read word until the line or file ends.
    pub fn ReadWordSameLine(self, to: &StringPiece, delim: bool) -> bool;

    // Read a line of text from the file.
    // Unlike ReadDelimited, this includes leading spaces and consumes the
    // delimiter. It is similar to getline in that way.

    // If strip_cr is true, any trailing carriate return (as would be found on
    // a file written on Windows) will be left out of the returned line.

    // Throws EndOfFileException if the end of the file is encountered. If the
    // file does not end in a newline, this could mean that the last line is never read.
    pub fn ReadLine(self, delim: &str, strip_cr: bool) -> StringPiece;

    // Read a line of text from the file, or return false on EOF.
    // This is like ReadLine, except it returns false where ReadLine throws
    // EndOfFileException.  Like ReadLine it may not read the last line in the
    // file if the file does not end in a newline.
    // If strip_cr is true, any trailing carriate return (as would be found on
    // a file written on Windows) will be left out of the returned line.
    pub fn ReadLineOrEOF(self, to: &StringPiece, delim: bool, strip_cr: bool) -> bool;
    pub fn ReadFloat(self) -> f64;
    pub fn ReadDouble(self) ->f64;
    pub fn ReadLong(self) -> i64;
    pub fn ReadULong(self) -> u64;

    // Skip spaces defined by isspace.
    pub fn SkipSpaces(self, delim: bool);

    fn Offset(self) -> u64;
    fn FileName(self) -> &str;
    // Force a progress update.
    fn UpdateProgress(self);
}

#[derive(Debug, Clone, Copy)]
pub struct MallocException;

#[derive(Debug)]
pub struct HandlerT;

#[derive(Debug)]
pub struct Construct;

#[derive(Debug)]
pub struct Worker<HandlerT>;

#[derive(Debug)]
pub struct RecyclingHandler<HandlerT>;

#[derive(Debug)]
pub struct Find;

#[derive(Debug)]
pub struct TokenIter;

#[derive(Debug)]
pub struct Compare;

#[derive(Debug)]
pub struct Combine;

#[derive(Debug)]
pub struct Sorts<FixedArray>;

impl Sorts<FixedArray> {
    pub fn Sorts(self, number: &i8);
    pub fn push_back(
        self, 
        chain: &Chain, config: &SortConfig, 
        compare: &Compare, combine: &Combine
    );
}

#[derive(Debug)]
pub struct Chain;

impl Chain {
    pub fn new(config: &ChainConfig) -> Self;
    pub fn EntrySize(self) -> i8;
    pub fn SetProgressTarget(self, target: u64);
    pub fn ActivateProgress(self);
    pub fn BlockCount(self) -> i8;
    pub fn Add(self) -> ChainPosition;
    pub fn Wait(self, release_memory: bool);
    pub fn Start(self);
    pub fn Running(self) -> bool;
}

#[derive(Debug)]
pub struct ChainPosition;

impl ChainPosition {
    pub fn GetChain(self) -> &Chain;
}
    
#[derive(Debug)]
pub struct DiscountConfig {
    overwrite: Vec<Discount>,
    // If discounting fails for an order, copy them from here.
    fallback: Discount,
    // What to do when discounts are out of range or would trigger divison by
    // zero. It does something other than THROW_UP, use fallback_discount.
    bad_action: WarningAction
}


#[derive(Debug)]
pub struct Chains<FixedArray>;

pub struct SortConfig {
    // Filename prefix where temporary files should be placed.
    temp_prefix: &str,
    // Size of each input/output buffer.
    buffer_size: i8,
    // Total memory to use when running alone.
    total_memory: i8
}


#[derive(Debug, Default)]
pub struct ChainConfig {
    entry_size: i8,  // Number of bytes in each record.

    block_count: i8,  // Number of blocks in the chain.
    // Total number of bytes available to the chain. This value will be divided amongst the blocks in the chain.
    // Chain's constructor will make this a multiple of entry_size.
    total_memory: i8
}

impl ChainConfig {
    pub fn new(in_entry_size: i8, in_block_count: i8, in_total_memory: i8) -> Self {
        ChainConfig { entry_size: in_entry_size, block_count: in_block_count, total_memory: in_total_memory }
    }
}

pub fn QueryFromBytes(model: &Model, config: BenchmarkConfig, Width: i8);
pub fn ConvertToBytes(model: &Model, fd_in: i64, width: i8);
// Determine how much physical memory there is.  Return 0 on failure.
pub fn GuessPhysicalMemory() -> u64;
// Parse a size like unix sort.  Sadly, this means the default multiplier is K.
pub fn ParseSize(arg: &str) -> u64;

// If it's a directory, add a /.  This lets users say -T /tmp without creating /tmpAAAAAA
pub fn NormalizeTempPrefix(base: &str);
pub fn CreateOrThrow(name: &str) -> i64;
pub fn OpenReadOrThrow(name: &str) -> i64;
pub fn PrintUsage(out: &OStream);

// Time in seconds since process started.  Zero on unsupported platforms.
pub fn WallTime() -> f64;

// User + system time, process-wide.
pub fn CPUTime() -> f64;

// User + system time, thread-specific.
pub fn ThreadTime() -> f64;

// Resident usage in bytes.
pub fn RSSMax() -> u64;

