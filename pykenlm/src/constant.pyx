from libc.stdint cimport  uintptr_t, uint64_t
from enum import Enum
cimport kenlm

ctypedef uintptr_t size_t


class Py_ARPALoadComplain(Enum):
	ALL = ARPALoadComplain.ALL
	EXPENSIVE = ARPALoadComplain.EXPENSIVE
	NONE = ARPALoadComplain.NONE


class Py_ModelType(Enum):
	PROBING = ModelType.PROBING
	REST_PROBING = ModelType.REST_PROBING
	TRIE = ModelType.TRIE
	QUANT_TRIE = ModelType.QUANT_TRIE
	ARRAY_TRIE = ModelType.ARRAY_TRIE
	QUANT_ARRAY_TRIE = ModelType.QUANT_ARRAY_TRIE

	def __iter__(self):
		for mem in self.__members__:
			yield mem
	def __contains__(self, item):
		return item in self.__members__


class Py_FilterMode(Enum):
	MODE_COPY = 1
	MODE_SINGLE = 2
	MODE_MULTIPLE = 3
	MODE_UNION = 4
	MODE_UNSET = 5


class Py_Format(Enum):
	FORMAT_ARPA = 1
	FORMAT_COUNT = 2
