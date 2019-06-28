# Code Generation Notes

## Term Representation

Terms will be broken up into two primary types:

  - Immediates, these fit in a single word, and so use a word
  - Aggregates/boxes, these will be represented as a struct of
    two fields, the first being the type metadata such as "tuple of 2 elements",
    and the second being either a pointer to somewhere, or the
    header for data which immediately follows it. In other words,
    a tuple of 2 immediates would be something like this:

        { i8*, i8* }

    Where in C-style the meaning is more like:

        typedef unsigned long Term;
        struct { Term tag, Term data[0] };

    The nature of the `data` field is that it always
    has reserved space for at least one word, but one
    may allocate extra space for it and index off the
    end to access additional words.

## Functions


The current approach until we have an actual type system is to
treat all functions as returning Term, and taking N terms as arguments.
In cases where some of those arguments are continuations, they are the
exception to the rule, and all functions will have a final parameter that
is the out value for errors.

## Threading

- The creation of a process must allocate a stack where the first frame which
will execute the call to `exit`, the call to the start function of that process
is then pushed on the stack as a new frame, and is where execution will begin
- As a result, all functions


### Coroutines

Additionally, every function will be structured as a coroutine, where the first
resumption performs the reduction/gc checks and either yields back to the scheduler,
or jumps to the "real" entry of the function. If such checks are required later in
the function, then it will yield again. When the function returns for the final time,
the coroutine frame will be destroyed.

The coroutine system needs two allocation functions to store a continuation
frame on the process heap:

    coroutine_alloc(size: usize) -> *mut u8;
    coroutine_free(ptr: *mut u8);
    
The _only_ thing the runtime should do with these functions is allocate the
requisite amount of memory, and correspondingly, free it, but the memory is
otherwise managed by the generated code, not the runtime itself. These functions
cannot fail, so if memory cannot be allocated for the coroutine, then the system
must abort, or the behaviour is undefined, and probably horrible.

LLVM handles storing allocas in the coroutine frame which are live across
suspension points, and generates code to unpack those values from the frame when
resuming the coroutine; as a result, we will not store any stack allocations in
the process itself. Since we manage the memory for the frame, we do have the
freedom to treat values in the frame as roots for GC, and update them
accordingly, but we will need to generate layout metadata for the frame so that
we can ensure that scanning the frame for roots is done correctly.

### Regarding Scheduling

The driver for the system will be the scheduler, which will call a shim function in
the generated code that handles resuming the current continuation if applicable, calling
it with the requisite arguments. The scheduler itself will only be given back an opaque
pointer for the continuation, to be stored in the process control block. When yielding,
control will appear to return from the shim function (from the perspective of the scheduler),
to the scheduler, which can then do whatever it needs to do, such as popping a different process
off the run queue and resuming its continuation. Each time a process yields, it will return
a pointer to its continuation, so the scheduler must store it.
