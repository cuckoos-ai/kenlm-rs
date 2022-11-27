

def GuessPhysicalMemory() -> int:
    """_summary_

    :return _type_: _description_
    """

def ParseSize(arg: str) -> int:
    """_summary_

    :param _type_ arg: _description_
    :return _type_: _description_
    """
    

def ParseBitCount(inp_: str) -> int:
    """_summary_

    :param _type_ inp_: _description_
    :return _type_: _description_
    """


class StringPiece:
    sp: 'StringPiece'

    def __init__(self, string_: str):
        self.sp = StringPiece(string_)

    def __dealloc__(self):
        del self.sp

    def __iter__(self):
        """
        An string iterator to iterate over string
        Returns
        -------

        """
        yield


class ChainConfig:
    chain_cfg_: 'ChainConfig'

    def __init__(self, size_t in_entry_size_, size_t in_block_count, size_t in_total_memory):
        self.chain_cfg_.entry_size = in_entry_size_
        self.chain_cfg_.block_count = in_block_count
        self.chain_cfg_.total_memory = in_total_memory

    @property
    def in_entry_size(self):
        return self.chain_cfg_.entry_size
    
    @property
    def in_block_count(self):
        return self.chain_cfg_.block_count
    
    @property
    def in_total_memory(self):
        return self.chain_cfg_.total_memory

class Chain:
     Chain * __chainer

    def __init__(self, ChainConfig config):
        # TODO: FixMe - Do type conversion
        self.__chainer = Chain(config.chain_cfg_)

    def __dealloc__(self):
        del self.__chainer

    def __len__(self):
        pass

class FixedArrayMixin:
    def begin(self):
        pass
    def end(self):
        pass
    def back(self):
        pass
    def size(self) -> int:
        pass
    def empty(self) -> bool:
        pass
    def __getitem__(self, i: int):
        pass
    def push_back(self):
        pass
    def pop_back(self):
        pass
    def clear(self):
        pass

class Chains(FixedArrayMixin):
     Chains chains
    def __init__(self):
        pass
    def __dealloc__(self):
        del self.chains


def class ChainPosition:
     ChainPosition c_pos_

    def __init__(self, **kwds):
        self.c_pos_ = ChainPosition(kwds)
    def __dealloc__(self):
        del self.c_pos_

    def Chain GetChain(self):
        pass


def class PySorts(FixedArrayMixin):
     Sorts _sorted
    def __init__(self, size_t number):
        # Sorts[ContextOrder]
        self._sorted = Sorts()
    def __dealloc__(self):
        del self._sorted

class FilePiece:
     FilePiece file_piece
     size_t min_buffer

    def __init__(self, io.FileIO fp, bool show_progress = False, size_t min_buffer = 1048576):
        self.fp = fp
        self.show_progress = show_progress
        self.min_buffer = min_buffer

    def begin(self) -> Iterator:
        pass

    def end(self) -> Iterator:
        pass

    def peek(self) -> str:
        pass

    def ReadDelimited(self, bool[:] delim = np.zeros((256,), dtype=bool)) -> StringPiece:
        pass

    def ReadWordSameLine(self, StringPiece to, bool[:] delim = np.zeros((256,), dtype=bool)) -> bool:
        pass

    def ReadLine(self, str delim = '\n', bool strip_cr = True) -> StringPiece:
        pass

    def ReadLineOrEOF(self, StringPiece to, char delim = '\n', bool strip_cr = True) -> bool:
        pass

    def ReadFloat(self) -> np.float:
        pass

    def ReadDouble(self) -> np.double:
        pass

    def ReadLong(self) -> np.long:
        pass

    def ReadULong(self) -> np.uint:
        pass

    def SkipSpaces(self, bool[:] delim = np.zeros((256,), dtype=bool)) -> None:
        # Skip spaces defined by isspace.
        pass

    def Offset(self) -> np.intc:
        pass

    def FileName(self) -> str:
        pass

    def UpdateProgress(self) -> None:
        # Force a progress update.
        pass

    def __dealloc__(self):
        del self.file_piece

def class RecyclingThreadPool:
     :
        size_t queue
        size_t workers
        Construct handler_construct
        Request poison

    def __init__(self, size_t queue, size_t workers, Construct handler_construct,
                    Request poison, HandlerType HandlerT = None):
        self._pool = RecyclingThreadPool[HandlerT](queue, workers, handler_construct, poison)
    def __dealloc__(self):
        del self._pool
    def PopulateRecycling(self, Request request):
        self._pool.PopulateRecycling(request)

    def Request Consume(self):
        self._pool.Consume()

    def Produce(self, const Request &request):
        self._pool.Produce(request)
