cimport kenlm
cimport util


cdef extern from "lm/filter/phrase.hh" namespace "lm::phrase":
    cdef cppclass Substrings:
        pass

    cdef cppclass Iterator:
        pass

    cdef cppclass Output:
        pass

    cdef cppclass Multiple:
        void AddNGram(const Iterator &begin, const Iterator &end,
                      const kenlm.StringPiece &line, Output &output) except +

        void AddNGram(const kenlm.StringPiece & ngram,
                      const kenlm.StringPiece & line, Output & output) except +

    ctypedef Multiple PhraseFilter

    unsigned int ReadMultiple(util.istream &in_, Substrings &out_) except +


cdef extern from "lm/filter/phrase.hh" namespace "lm::phrase":
    cdef cppclass Union:
        Union(const Substrings & substrings) except +
