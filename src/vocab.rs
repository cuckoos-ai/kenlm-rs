// cimport util
// cimport constant as ct
// from libcpp.string cimport string
// from libcpp.vector cimport vector
// from libcpp.unordered_set cimport unordered_set
// from libcpp.unordered_map cimport unordered_map


// ctypedef unordered_map[string, vector[ct.UINT64_t]] Words

// cdef extern from "lm/filter/vocab.hh" namespace "lm::vocab":
//     cdef cppclass Multiple:
//         Multiple(const Words &vocabs)

//     ctypedef Multiple VocabFilter

// cdef extern from "lm/filter/vocab.cc" namespace "lm::vocab":
//     cdef void ReadSingle(util.istream & in_, unordered_set [string] &out) except +


// cdef extern from "lm/filter/vocab.hh" namespace "lm::vocab":
//     cdef cppclass Union:
//         Union(const Words & vocabs) except +
