// from libcpp.vector cimport vector
// from libcpp.string cimport string

// class Py_InterpolateConfig:
//     cdef Config in_config_

//     def __cinit__(self, float[:] lambdas, Py_SortConfig sort_):
//         self.in_config_.lambdas = lambdas
//         self.in_config_.sort = sort_.sort_cfg

//     @property
//     def lambdas(self):
//         return self.in_config_.lambdas

//     @lambdas.setter
//     def lambdas(self, v):
//         self.in_config_.lambdas = v

//     def __dealloc__(self):
//         del self.in_config_

// class Py_SortConfig:
//     cdef:
//         util.SortConfig sort_cfg
//         # Filename prefix where temporary files should be placed.
//         string temp_prefix
//         # Size of each input/output buffer.
//         size_t buffer_size
//         # Total memory to use when running alone.
//         size_t total_memory

//     def __init__(self, str temp_prefix, int buffer_size, int total_memory):
//         self.sort_cfg.temp_prefix = temp_prefix
//         self.sort_cfg.buffer_size = buffer_size
//         self.sort_cfg.total_memory = total_memory

//     def __del__(self):
//         del self.sort_cfg


// class Py_SplitWorker:
//     cdef SplitWorker sw
//     cpdef __cinit__(self):
//         pass
//     cpdef __dealloc__(self):
//         del self.sw
//     cpdef void Run(self, util.Py_ChainPosition position):
//         pass

// class Py_InstancesConfig:
//     cdef InstancesConfig _in_cfg

//     def __cinit__(self, Py_SortConfig sort_):
//         cdef InstancesConfig instance_cfg
//         # TODO: FixMe: Cast to Sortconfig
//         instance_cfg.sort = sort_.sort_cfg
//         self._in_cfg.model_read_chain_mem = instance_cfg.sort.buffer_size
//         self._in_cfg.extension_write_chain_mem = instance_cfg.sort.extension_write_chain_mem
//         self._in_cfg.lazy_memory = instance_cfg.sort.total_memory
//         self._in_cfg = instance_cfg

//     def __dealloc__(self):
//         del self._in_cfg


// def Py_Pipeline(util.FixedArray &models, Config & config, int write_file) -> None:
//     # TODO: FixMe - Cast to ModelBuffer
//     # FixedArray[ModelBuffer]
//     Pipeline(models, config, write_file)


// def Py_TuneWeights(
//     str tune_file,
//     list model_names,
//     Py_InstancesConfig & config,
//     float[:] weights_out):
//     """

//     Parameters
//     ----------
//     tune_file
//     model_names
//     config
//     weights_out

//     Returns
//     -------

//     """
//     cdef vector[util.StringPiece] model_names_ = model_names
//     # TODO: FixMe
//     TuneWeights(tune_file, model_names_, config, weights_out)
