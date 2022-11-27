// cimport kenlm
// cimport constant as ct

// class Py_NgramConfig:
// 	"""
// 	Wrapper around lm::ngram::Config.
// 	Pass this to Model's constructor to set configuration options.
// 	"""
// 	cdef Config _c_config

// 	def __init__(
// 		self,
// 		bool prob_bits,
// 		bool backoff_bits,
// 		bool pointer_bhiksha_bits,
// 		float unknown_missing_logprob,
// 		float probing_multiplier,
// 		string temporary_directory_prefix,
// 		size_t building_memory,
// 		ct.WriteMethod write_method,
// 		ct.WarningAction sentence_marker_missing,
// 		ct.WarningAction positive_log_probability,
// 		ct.RestFunction rest_function,
// 		bool include_vocab,
// 	):
// 		self._c_config.rest_function = rest_function
// 		self._c_config.include_vocab = include_vocab
// 		self._c_config.write_method = write_method
// 		self._c_config.sentence_marker_missing = sentence_marker_missing
// 		self._c_config.positive_log_probability = positive_log_probability
// 		self._c_config.building_memory = building_memory
// 		self._c_config.temporary_directory_prefix = temporary_directory_prefix
// 		self._c_config.prob_bits = prob_bits
// 		self._c_config.backoff_bits = backoff_bits
// 		self._c_config.probing_multiplier = probing_multiplier
// 		self._c_config.unknown_missing_logprob = unknown_missing_logprob
// 		self._c_config.pointer_bhiksha_bits = pointer_bhiksha_bits

// 	@property
// 	def load_method(self):
// 		return self._c_config.load_method

// 	@load_method.setter
// 	def load_method(self, to):
// 		self._c_config.load_method = to

// 	@property
// 	def show_progress(self):
// 		return self._c_config.show_progress
// 	@show_progress.setter
// 	def show_progress(self, to):
// 		self._c_config.show_progress = to

// 	@property
// 	def arpa_complain(self):
// 		return self._c_config.arpa_complain
// 	@arpa_complain.setter
// 	def arpa_complain(self, to):
// 		self._c_config.arpa_complain = to

// 	def __del__(self):
// 		del self._c_config





// cdef class GenericModel__template:
// 	cdef:
// 		GenericModel[Search, VocabularyT] model_
// 		const ModelType kModelType
// 		const unsigned int kVersion = Search.kVersion

// 	def __cinit__(self, char *file_path, Config & init_config):
// 		pass

// 	def __dealloc__(self):
// 		del self.model_

// cpdef class ProbingModel(GenericModel__template):
// 	def __cinit__(self, char *file_path, Config & init_config):
// 		self.model_ = GenericModel[HashedSearch[BackoffValue], ProbingVocabulary](
// 			file_path, init_config
// 		)

// cpdef class RestProbingModel(GenericModel__template):
// 	def __cinit__(self, char *file_path, Config & init_config):
// 		self.model_ = GenericModel[HashedSearch[RestValue], ProbingVocabulary](
// 			file_path, init_config
// 		)

// cpdef class TrieModel(GenericModel__template):
// 	def __cinit__(self, char *file_path, Config & init_config):
// 		self.model_ = GenericModel[TrieSearch[DontQuantize, DontBhiksha], SortedVocabulary](
// 			file_path, init_config
// 		)

// cpdef class ArrayTrieModel(GenericModel__template):
// 	def __cinit__(self, char *file_path, Config & init_config):
// 		self.model_ = GenericModel[HashedSearch[DontQuantize, ArrayBhiksha], SortedVocabulary](
// 			file_path, init_config
// 		)

// cpdef class QuantTrieModel(GenericModel__template):
// 	def __cinit__(self, char *file_path, Config & init_config):
// 		self.model_ = GenericModel[HashedSearch[SeparatelyQuantize, DontBhiksha], SortedVocabulary](
// 			file_path, init_config
// 		)

// cpdef class QuantArrayTrieModel(GenericModel__template):
// 	def __cinit__(self, char *file_path, Config & init_config):
// 		self.model_ = GenericModel[TrieSearch[SeparatelyQuantize, ArrayBhiksha], SortedVocabulary](
// 			file_path, init_config
// 		)
