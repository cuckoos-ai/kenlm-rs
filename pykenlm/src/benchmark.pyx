cimport util
cimport kenlm
from libc.string cimport memmove
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc

cpdef QueryFromBytes(Model model, BenchmarkConfig config, uint8_t Width):
    # TODO: Python api for filestream
    cdef FileStream out
    cdef double total

    out.write(config.threads)
    cdef const Width kEOS = model.GetVocabulary().EndSentence()
    cdef double total = 0.0
    # Number of items to have in queue in addition to everything in flight.
    cdef const size_t kInQueue = 3
    cdef size_t total_queue = config.threads + kInQueue
    cdef vector[Width] backing(config.buf_per_thread * total_queue)
    cdef double loaded_cpu
    cdef double loaded_wall
    cdef uint64_t queries = 0
    cdef size_t i = 0

    pool = kenlm.RecyclingThreadPool[Worker[Model, Width]](
        total_queue,
        config.threads,
        Worker(model, total), kenlm.iterator_range[Width *]((Width *)0, (Width *)0)
    )


    for i in range(total_queue):
        pool.PopulateRecycling(kenlm.iterator_range[Width *](
            &backing[i * config.buf_per_thread],
            &backing[i * config.buf_per_thread]))

    print(f"To Load, CPU: {loaded_cpu} Wall: {loaded_wall}")

    overhang = kenlm.iterator_range[Width *]((Width *)0, (Width *)0)

    while 1:
        kenlm.iterator_range[Width *] buf = pool.Consume()
        memmove(buf.begin(), overhang.begin(), overhang.size() * sizeof(Width))
        size_t got = kenlm.ReadOrEOF(config.fd_in, buf.begin() + overhang.size(), (config.buf_per_thread - overhang.size()) * sizeof(Width))
        if not got and overhang.empty():
            break
        assert !(got % sizeof(Width), f"File size not a multiple of vocab id size {sizeof(Width)}")

        cdef Width *read_end = buf.begin() + overhang.size() + got / sizeof(Width)
        cdef Width *last_eos
        # for (; ; --last_eos) {
        last_eos = read_end - 1
        while True:
            last_eos = -1
            if (*last_eos == kEOS):
                break

        buf = iterator_range[Width*](buf.begin(), last_eos + 1)
        overhang = iterator_range[Width*](last_eos + 1, read_end)
        queries += buf.size()
        pool.Produce(buf)


    # util::FileStream(2, 70) << "Probability sum: " << total << '\n';
    print(f"Queries: {queries}")
    print(f"Excluding load, CPU: {(after_cpu - loaded_cpu)} Wall: {(after_wall - loaded_wall)}")

    print(f"Seconds per query excluding load, CPU: {cpu_per_entry} Wall: {wall_per_entry}")
    print(f"Queries per second excluding load, CPU: {1.0/cpu_per_entry} Wall: {(1.0/wall_per_entry)}")
    print(f"RSSMax: {kenlm.RSSMax()}")


cpdef ConvertToBytes(Model &model, int fd_in, Width=uint8_t):
    cdef:
        kenlm.FilePiece in_(fd_in)
        kenlm.FileStream out(1)
        Width width
        kenlm.StringPiece word
        const Width end_sentence = (Width) model.GetVocabulary().EndSentence()

    while True:
        while in_.ReadWordSameLine(word):
            width = (Width)model.GetVocabulary().Index(word)
            out.write(&width, sizeof(Width))

        if not in_.ReadLineOrEOF(word):
            break
        out.write(&end_sentence, sizeof(Width))


cpdef class Worker:
    cdef:
        const Model &model_
        double total_
        double &add_total_
        kenlm.State state_[3]

    ctypedef kenlm.iterator_range[Width *] Request

    cdef __cinit__(self, const Model &model, double &add_total):
        self.model_ = model
        self.total_ = 0.0
        self.add_total_ = add_total

    #  Destructors happen in the main thread, so there's no race for add_total_.
    cdef __dealloc__(self):
        self.add_total_ += self.total_

    cpdef void operator(Request request):
        # TODO: FixMe Look for how to operator overloading in cython
        # - Also fix the subroutine

        cdef:
            const State *const begin_state = &model_.BeginSentenceState()
            const State *next_state = begin_state
            const Width kEOS = model_.GetVocabulary().EndSentence()
            float summ = 0.0
            # Do even stuff first.
            const Width *even_end = request.begin() + (request.size() & ~1);
            # Alternating states
            const Width *i
        i = request.begin()

        while i != even_end:
            summ += model_.FullScore(*next_state, deref(i), state_[1]).prob
            # TODO: Pointer increment here
            next_state = begin_state if (deref(inc(i)) == kEOS) else &state_[1]
            summ += model_.FullScore(deref(next_state), deref(i), state_[0]).prob
            next_state = begin_state if (deref(inc(i)) == kEOS) else &state_[0]

        # Odd corner case.
        if request.size() & 1:
            summ += model_.FullScore(*next_state, *i, state_[2]).prob
            next_state = begin_state if deref(inc(i)) == kEOS) else &state_[2]

        total_ += summ
