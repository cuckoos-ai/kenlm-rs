// cimport kenlm
// cimport vocab
// cimport phrase
// cimport ngram
// cimport benchmark

// ctypedef fused Format:
// 	kenlm.ARPAFormat
// 	kenlm.CountFormat

// ctypedef fused Filter:
// 	phrase.PhraseFilter
// 	vocab.VocabFilter

// ctypedef fused OutputBuffer:
// 	kenlm.BinaryOutputBuffer
// 	kenlm.MultipleOutputBuffer

// ctypedef fused Output:
// 	pass

// ctypedef fused Quant:
// 	ngram.DontQuantize
// 	ngram.SeparatelyQuantize

// ctypedef fused Bhiksha:
// 	ngram.DontBhiksha
// 	ngram.ArrayBhiksha

// ctypedef fused VocabularyT:
// 	ngram.ProbingVocabulary
// 	ngram.SortedVocabulary

// ctypedef fused Search:
// 	ngram.HashedSearch
// 	ngram.TrieSearch

// ctypedef fused Value:
// 	ngram.RestValue
// 	ngram.BackoffValue

// ctypedef fused WORKER_TYPE:
//     benchmark.Model
//     benchmark.Width

// ctypedef fused QUERY_TYPE:
// 	kenlm.Model
// 	kenlm.Printer

// ctypedef fused Binary:
// 	pass

// # cdef cppclass MultipleARPAOutput:
// #     pass
// #
// # cdef cppclass Filter:
// #     pass
// #
// # cdef cppclass Out:
// #     pass

// # ctypedef MultipleARPAOutput Multiple
// # ctypedef MultipleOutput[CountOutput] Multiple
