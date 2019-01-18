///! In Erlang, memory allocation consists of a handful of use-case specific allocators,
///! and is managed and cleaned up by two separate collection strategies depending on what
///! type of data it is, and where it is allocated:
///!
///! Let's look at the collectors first, there are two:
///!
///! * Generational, copying garbage collector for process-local heaps and message areas
///!   * The copy algorithm is Cheney-style
///!   * Stop-and-copy, but only affects the process being collected
///!   * Does not need a remembered set, as pointers are unidirectional (new-to-old, but never old-to-new)
///!   * Data must survive two generations before being promoted to the old generation
///!   * In Lumen (as in HiPE), we use stack maps to guide the collector (identify roots)
///!   * Generational stack scanning is used to further reduce the number of roots which need to be
///!     scanned during collection (by using information from previous scans)
///!   * Generational process scanning (basically like stack scanning, but applied to processes) is
///!     used to reduce the root set for the message area, so only memory from active processes need
///!     be considered (i.e. processes which have sent/received messages since the last collection).
///!     Such processes are stored in a structure called the _dirty process set_
///! * Reference counting for objects on a shared heap
///! * And technically, there is a third, which could be considered region-based collection,
///!   which occurs when a process exits and all of its owned data is reclaimed, this includes
///!   its Process Control Block (PCB), stack, and heap.
///!
///! The latter two are not particularly interesting, suffice to say that they work the same
///! as you'd expect them to work. What is interesting is how the generational GC works.
///!
///! First, some fundamental properties that enable the GC to be performant:
///!
///! * Every process has its own heap, and collection only needs to consider that heap, this
///!   means that unlike typical GCs, which must examine all roots globally, in Erlang, the
///!   GC only need to consider roots in a small subset of the heap, and only when that heap
///!   grows past the initial heap size.
///!   In order to ensure this property holds, there are invariants which must not be violated,
///!   and are maintained by the allocator:
///!     * No pointers from shared heaps to local heaps
///!     * No pointers from one local heap to another local heap
///!     * No cyclical references
///! * Not every process has to go through a GC, short-lived processes will almost certainly
///!   not incur any collection at all, and their memory will be reclaimed when they exit,
///!   similar to how Rust data types are dropped at the end of their scope.
///! * Data is allocated on the process-local heap by default, unless it is known to be data
///!   which will be shared, in which case it is allocated on a shared heap, the most obvious
///!   case of which is data which is used in message sends
///! * Data, in general, is copied when sent via messages, but there are techniques to make
///!   this much more important than the naive approach:
///!     * As mentioned above, the compiler will speculatively allocate data on the shared heap
///!       if it knows that the data will be used in a message, which means the data does not
///!       need to be copied, it only needs a reference
///!     * In addition, all data involved in a message send is wrapped on a copy-on-demand operation,
///!       which will copy locally-allocated data to the shared heap when it is actually needed, but
///!       this check is eliminated if, as in the first point, the compiler allocates it on the shared
///!       heap in advance.
///!     * Large binaries will be allocated on a shared heap, to avoid copying data when sending
///!       it between processes
///!     * When data is sent back and forth between two processes, and is not modified, it is copied
///!       at most once (to the shared heap), as it is shared by reference
///!
///! From the allocator's perspective, there are two types of objects:
///!
///! * Cons cells (list objects with a head and a tail), size is only two words
///! * Boxed objects (consisting of a header word, and either contains data directly, or is a pointer to data)
///!   * Boxed objects which are pointers to the data are generally pointers to another header, containing size
///!     information about that data
///!   * Consists of tuples, maps, arbitrary precision integers, floats, binaries, and closures
///!
///! Likewise, the allocator (and the rest of the system) needs to know which type of reference the data is:
///!
///! * An owned reference to the local heap
///! * A shared reference
///!
///! ## Incremental collector for the message area:
///!
///! ### Definitions
///!
///! * Mutator: a thread which is doing work which interacts with the allocator
///! * Collection stage: contiguous period of time during which garbage collection takes place
///! * Minor collection: complete collection of the young generation
///! * Major collection: complete collection of both young and old generations
///!
///! ### Design
///!
///! * Runs in a dedicated thread
///! * Uses a tri-color abstraction; objects are assigned one of three colors: white, gray, or black
///!   * White (unprocessed) is the default color of all objects at the beginning of a cycle
///!   * Gray (visited) is the color of objects visited, but only partially processed
///!   * Black (completed) is the color of objects which have been fully processed, only given to gray objects
///!   * At the end of a collection, all gray objects have been turned black, and any remaining white objects
///!     are collected
///! * Young generation is managed by a copying collector, with two evenly-sized spaces:
///!   * The nursery is used for allocations by the mutator during a cycle
///!   * The _from space_ is used in the copying collection,
///!   * The _to space_ is the old generation
///! * The old generation is managed by a mark-and-sweep collector
///!   * Consists of `n` pages in a linked list
///!   * Allocation uses a free-list, but the algorithm used can be one of many options:
///!     * First-fit
///!     * Divide the free-list into sublists for objects of different sizes
///! * Forwarding area, to allow the mutator to access objects in the from space between collection stages, i.e. during a cycle
///!   * Is no larger than the size of the from space
///! * To mark an object in the old generation as live, a bit vector is used, called a black map; we cannot mark the objects
///!   themselves because we already use all the bits in headers for type information
///! * There is a pointer into the nursery, called the allocation limit
mod common;
pub mod gc;

pub enum AllocatorType {
    System, // sys_alloc
    Temporary, // temp_alloc
    ShortLived, // sl_alloc
    Standard, // std_alloc
    LongLived, // ll_alloc
    EHeap, // eheap_alloc
    ETS, // ets_alloc
    FixedSize, // fix_alloc
    Literal, // literal_alloc
    Exec, // exec_alloc
    Binary, // binary_alloc
    Driver, // driver_alloc
    Test // test_alloc
}

pub enum AllocatorClass {
    Processes, // process_data
    Atom, // atom_data
    Code, // code_data
    ETS, // ets_data
    Binaries, // binary_data
    System // system_data
}

/;type NULL = *const u8

//trait Gc {
    // (p, unsigned_val(Eterm), NULL, 0)
    //fn erts_garbage_collect(p: &Process, need: usize, NULL, na: usize)
//}
