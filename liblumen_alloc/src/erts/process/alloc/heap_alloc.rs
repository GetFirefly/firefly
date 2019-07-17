use core::alloc::{AllocErr, Layout};
use core::ops::DerefMut;
use core::ptr::{self, NonNull};
use core::str::Chars;

use alloc::sync::Arc;
use alloc::vec::Vec;

use liblumen_core::util::reference::bytes;
use liblumen_core::util::reference::str::inherit_lifetime as inherit_str_lifetime;

use crate::borrow::CloneToProcess;
use crate::erts::process::code::Code;
use crate::erts::term::reference::{self, Reference};
use crate::erts::term::{
    make_pid, pid, AsTerm, Bitstring, BytesFromBinaryError, Closure, Cons, ExternalPid, Float,
    HeapBin, Integer, Map, ProcBin, StrFromBinaryError, SubBinary, Term, Tuple, TypedTerm,
};
use crate::{erts, ModuleFunctionArity};
use crate::{scheduler, VirtualAlloc};

/// A trait, like `Alloc`, specifically for allocation of terms on a process heap
pub trait HeapAlloc {
    /// Perform a heap allocation.
    ///
    /// If space on the process heap is not immediately available, then the allocation
    /// will be pushed into a heap fragment which will then be later moved on to the
    /// process heap during garbage collection
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr>;

    /// Same as `alloc`, but takes a `Layout` rather than the size in words
    unsafe fn alloc_layout(&mut self, layout: Layout) -> Result<NonNull<Term>, AllocErr> {
        let need = erts::to_word_size(layout.size());
        self.alloc(need)
    }

    /// Returns true if the given pointer is owned by this process/heap
    fn is_owner<T>(&mut self, ptr: *const T) -> bool;

    /// Constructs a binary from the given byte slice, and associated with the given process
    ///
    /// For inputs greater than 64 bytes in size, the resulting binary data is allocated
    /// on the global shared heap, and reference counted (a `ProcBin`), the header to that
    /// binary is allocated on the process heap, and the data is placed in the processes'
    /// virtual binary heap, and a boxed term is returned which can then be placed on the stack,
    /// or as part of a larger structure if desired.
    ///
    /// For inputs less than or equal to 64 bytes, both the header and data are allocated
    /// on the process heap, and a boxed term is returned as described above.
    ///
    /// NOTE: If allocation fails for some reason, `Err(AllocErr)` is returned, this usually
    /// indicates that a process needs to be garbage collected, but in some cases may indicate
    /// that the global heap is out of space.
    fn binary_from_bytes(&mut self, bytes: &[u8]) -> Result<Term, AllocErr>
    where
        Self: VirtualAlloc,
    {
        let len = bytes.len();

        // Allocate ProcBins for sizes greater than 64 bytes
        if len > 64 {
            match self.procbin_from_bytes(bytes) {
                Err(_) => Err(AllocErr),
                Ok(term) => {
                    // Add the binary to the process's virtual binary heap
                    let bin_ptr = term.boxed_val() as *mut ProcBin;
                    let bin = unsafe { &*bin_ptr };
                    self.virtual_alloc(bin);

                    Ok(term)
                }
            }
        } else {
            self.heapbin_from_bytes(bytes)
        }
    }

    /// Either returns a `&[u8]` to the pre-existing bytes in the heap binary, process binary, or
    /// aligned subbinary or creates a new aligned binary and returns the bytes from that new
    /// binary.
    fn bytes_from_binary<'heap>(
        &'heap mut self,
        binary: Term,
    ) -> Result<&'heap [u8], BytesFromBinaryError>
    where
        Self: VirtualAlloc,
    {
        match binary.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::HeapBinary(heap_binary) => {
                    Ok(unsafe { bytes::inherit_lifetime(heap_binary.as_bytes()) })
                }
                TypedTerm::ProcBin(process_binary) => {
                    Ok(unsafe { bytes::inherit_lifetime(process_binary.as_bytes()) })
                }
                TypedTerm::SubBinary(subbinary) => {
                    if subbinary.is_binary() {
                        if subbinary.bit_offset() == 0 {
                            Ok(unsafe { bytes::inherit_lifetime(subbinary.as_bytes()) })
                        } else {
                            let aligned_byte_vec: Vec<u8> = subbinary.byte_iter().collect();
                            let aligned = self
                                .binary_from_bytes(&aligned_byte_vec)
                                .map_err(|error| BytesFromBinaryError::Alloc(error))?;

                            self.bytes_from_binary(aligned)
                        }
                    } else {
                        Err(BytesFromBinaryError::NotABinary)
                    }
                }
                _ => Err(BytesFromBinaryError::Type),
            },
            _ => Err(BytesFromBinaryError::Type),
        }
    }

    /// Constructs a binary from the given string, and associated with the given process
    ///
    /// For inputs greater than 64 bytes in size, the resulting binary data is allocated
    /// on the global shared heap, and reference counted (a `ProcBin`), the header to that
    /// binary is allocated on the process heap, and the data is placed in the processes'
    /// virtual binary heap, and a boxed term is returned which can then be placed on the stack,
    /// or as part of a larger structure if desired.
    ///
    /// For inputs less than or equal to 64 bytes, both the header and data are allocated
    /// on the process heap, and a boxed term is returned as described above.
    ///
    /// NOTE: If allocation fails for some reason, `Err(AllocErr)` is returned, this usually
    /// indicates that a process needs to be garbage collected, but in some cases may indicate
    /// that the global heap is out of space.
    fn binary_from_str(&mut self, s: &str) -> Result<Term, AllocErr>
    where
        Self: VirtualAlloc,
    {
        let len = s.len();
        // Allocate ProcBins for sizes greater than 64 bytes
        if len > HeapBin::MAX_SIZE {
            match self.procbin_from_str(s) {
                Err(_) => Err(AllocErr),
                Ok(term) => {
                    // Add the binary to the process's virtual binary heap
                    let bin_ptr = term.boxed_val() as *mut ProcBin;
                    let bin = unsafe { &*bin_ptr };
                    self.virtual_alloc(bin);
                    Ok(term)
                }
            }
        } else {
            self.heapbin_from_str(s)
        }
    }

    fn closure(
        &mut self,
        creator: Term,
        module_function_arity: Arc<ModuleFunctionArity>,
        code: Code,
    ) -> Result<Term, AllocErr> {
        let closure = Closure::new(module_function_arity, code, creator);

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<Closure>())?.as_ptr() as *mut Closure;
            ptr::write(ptr, closure);
            let heap_closure = &*ptr;

            Ok(heap_closure.as_term())
        }
    }

    /// Constructs a list of only the head and tail, and associated with the given process.
    fn cons(&mut self, head: Term, tail: Term) -> Result<Term, AllocErr> {
        let cons = Cons::new(head, tail);

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<Cons>())?.as_ptr() as *mut Cons;
            ptr::write(ptr, cons);
            let heap_cons = &*ptr;

            Ok(heap_cons.as_term())
        }
    }

    fn external_pid_with_node_id(
        &mut self,
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> Result<Term, MakePidError>
    where
        Self: core::marker::Sized,
    {
        let external_pid = ExternalPid::with_node_id(node_id, number, serial)?;
        let heap_external_pid = external_pid.clone_to_heap(self)?;

        Ok(heap_external_pid)
    }

    fn float(&mut self, f: f64) -> Result<Term, AllocErr> {
        let float = Float::new(f);

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<Float>())?.as_ptr() as *mut Float;
            ptr::write(ptr, float);
            let process_float = &*ptr;

            Ok(process_float.as_term())
        }
    }

    /// Constructs a heap-allocated binary from the given byte slice, and associated with the given
    /// process
    fn heapbin_from_bytes(&mut self, s: &[u8]) -> Result<Term, AllocErr> {
        let len = s.len();

        unsafe {
            // Allocates space on the process heap for the header + data
            let header_ptr = self.alloc_layout(HeapBin::layout_bytes(s))?.as_ptr() as *mut HeapBin;
            // Pointer to start of binary data
            let bin_ptr = header_ptr.offset(1) as *mut u8;
            // Construct the right header based on whether input string is only ASCII or includes
            // UTF8
            let header = HeapBin::new(len);
            // Write header
            ptr::write(header_ptr, header);
            // Copy binary data to destination
            ptr::copy_nonoverlapping(s.as_ptr(), bin_ptr, len);
            // Return a box term that points to the header
            let result = Term::make_boxed(header_ptr);

            Ok(result)
        }
    }

    /// Constructs a heap-allocated binary from the given string, and associated with the given
    /// process
    fn heapbin_from_str(&mut self, s: &str) -> Result<Term, AllocErr> {
        let len = s.len();

        unsafe {
            // Allocates space on the process heap for the header + data
            let header_ptr = self.alloc_layout(HeapBin::layout(s))?.as_ptr() as *mut HeapBin;
            // Pointer to start of binary data
            let bin_ptr = header_ptr.offset(1) as *mut u8;
            // Construct the right header based on whether input string is only ASCII or includes
            // UTF8
            let header = if s.is_ascii() {
                HeapBin::new_latin1(len)
            } else {
                HeapBin::new_utf8(len)
            };
            // Write header
            ptr::write(header_ptr, header);
            // Copy binary data to destination
            ptr::copy_nonoverlapping(s.as_ptr(), bin_ptr, len);
            // Return a box term that points to the header
            let result = Term::make_boxed(header_ptr);

            Ok(result)
        }
    }

    fn improper_list_from_iter<I>(&mut self, iter: I, last: Term) -> Result<Term, AllocErr>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        let mut acc = last;

        for element in iter.rev() {
            acc = self.cons(element, acc)?;
        }

        Ok(acc)
    }

    fn improper_list_from_slice(&mut self, slice: &[Term], tail: Term) -> Result<Term, AllocErr> {
        self.improper_list_from_iter(slice.iter().map(|t| *t), tail)
    }

    /// Constructs an integer value from any type that implements `Into<Integer>`,
    /// which currently includes `SmallInteger`, `BigInteger`, `usize` and `isize`.
    ///
    /// This operation will transparently handle constructing the correct type of term
    /// based on the input value, i.e. an immediate small integer for values that fit,
    /// else a heap-allocated big integer for larger values.
    fn integer<I: Into<Integer>>(&mut self, i: I) -> Result<Term, AllocErr>
    where
        Self: core::marker::Sized,
    {
        match i.into() {
            Integer::Small(small) => Ok(unsafe { small.as_term() }),
            Integer::Big(big) => big.clone_to_heap(self),
        }
    }

    fn charlist_from_str(&mut self, s: &str) -> Result<Term, AllocErr>
    where
        Self: core::marker::Sized,
    {
        self.list_from_chars(s.chars())
    }

    /// Constructs a list from the chars and associated with the given process.
    fn list_from_chars(&mut self, chars: Chars) -> Result<Term, AllocErr>
    where
        Self: core::marker::Sized,
    {
        let mut acc = Term::NIL;

        for character in chars.rev() {
            let code_point = self.integer(character)?;

            acc = self.cons(code_point, acc)?;
        }

        Ok(acc)
    }

    fn list_from_iter<I>(&mut self, iter: I) -> Result<Term, AllocErr>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        self.improper_list_from_iter(iter, Term::NIL)
    }

    fn list_from_slice(&mut self, slice: &[Term]) -> Result<Term, AllocErr> {
        self.improper_list_from_slice(slice, Term::NIL)
    }

    /// Constructs a map and associated with the given process.
    fn map_from_slice(&mut self, slice: &[(Term, Term)]) -> Result<Term, AllocErr>
    where
        Self: core::marker::Sized,
    {
        Map::from_slice(slice).clone_to_heap(self)
    }

    /// Creates a `Pid` or `ExternalPid` with the given `node`, `number` and `serial`.
    fn pid_with_node_id(
        &mut self,
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> Result<Term, MakePidError>
    where
        Self: core::marker::Sized,
    {
        if node_id == 0 {
            make_pid(number, serial).map_err(|error| error.into())
        } else {
            self.external_pid_with_node_id(node_id, number, serial)
        }
    }

    /// Constructs a reference-counted binary from the given byte slice, and associated with the
    /// given process
    fn procbin_from_bytes(&mut self, s: &[u8]) -> Result<Term, AllocErr> {
        // Allocates on global heap
        let bin = ProcBin::from_slice(s)?;
        // Allocates space on the process heap for the header
        let header_ptr = unsafe { self.alloc_layout(Layout::new::<ProcBin>())?.as_ptr() };
        // Write the header to the process heap
        unsafe { ptr::write(header_ptr as *mut ProcBin, bin) };
        // Returns a box term that points to the header
        let result = Term::make_boxed(header_ptr);
        Ok(result)
    }

    /// Constructs a reference-counted binary from the given string, and associated with the given
    /// process
    fn procbin_from_str(&mut self, s: &str) -> Result<Term, AllocErr> {
        // Allocates on global heap
        let bin = ProcBin::from_str(s)?;
        // Allocates space on the process heap for the header
        let header_ptr = unsafe { self.alloc_layout(Layout::new::<ProcBin>())?.as_ptr() };
        // Write the header to the process heap
        unsafe { ptr::write(header_ptr as *mut ProcBin, bin) };
        // Returns a box term that points to the header
        let result = Term::make_boxed(header_ptr);

        Ok(result)
    }

    /// Creates a `Reference` with the given `number` associated with the Process.
    fn reference(
        &mut self,
        scheduler_id: scheduler::ID,
        number: reference::Number,
    ) -> Result<Term, AllocErr> {
        let layout = Reference::layout();
        let reference_ptr = unsafe { self.alloc_layout(layout)?.as_ptr() as *mut Reference };
        let reference = Reference::new(scheduler_id, number);
        unsafe {
            // Write header
            ptr::write(reference_ptr, reference);
        }
        // Return box to tuple
        let reference = Term::make_boxed(reference_ptr);

        Ok(reference)
    }

    /// Either returns a `&str` to the pre-existing bytes in the heap binary, process binary, or
    /// aligned subbinary or creates a new aligned binary and returns the bytes from that new
    /// binary.
    fn str_from_binary<'heap>(
        &'heap mut self,
        binary: Term,
    ) -> Result<&'heap str, StrFromBinaryError>
    where
        Self: VirtualAlloc,
    {
        let bytes = self.bytes_from_binary(binary)?;

        str_from_binary_bytes(bytes)
    }

    /// Constructs a subbinary from the given original, and associated with the given process.
    ///
    /// Original must be a heap binary or a process binary.  To take the subbinary of a subbinary,
    /// use the first subbinary's original instead and combine the offsets.
    ///
    /// NOTE: If allocation fails for some reason, `Err(AllocErr)` is returned, this usually
    /// indicates that a process needs to be garbage collected, but in some cases may indicate
    /// that the global heap is out of space.
    fn subbinary_from_original(
        &mut self,
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        full_byte_len: usize,
        partial_byte_bit_len: u8,
    ) -> Result<Term, AllocErr> {
        let subbinary = SubBinary::from_original(
            original,
            byte_offset,
            bit_offset,
            full_byte_len,
            partial_byte_bit_len,
        );

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<SubBinary>())?.as_ptr() as *mut SubBinary;
            ptr::write(ptr, subbinary);
            let process_subbinary = &*ptr;

            Ok(process_subbinary.as_term())
        }
    }

    /// Constructs a `Tuple` from an `Iterator<Item = Term>` and accompanying `len`.
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the `iterator` provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `make_binary_from_str`.
    fn tuple_from_iter<I>(&mut self, iterator: I, len: usize) -> Result<Term, AllocErr>
    where
        I: Iterator<Item = Term>,
    {
        let layout = Tuple::layout(len);
        let tuple_ptr = unsafe { self.alloc_layout(layout)?.as_ptr() as *mut Tuple };
        let head_ptr = unsafe { tuple_ptr.offset(1) as *mut Term };
        let tuple = Tuple::new(len);
        unsafe {
            // Write header
            ptr::write(tuple_ptr, tuple);
            // Write each element
            for (index, element) in iterator.enumerate() {
                ptr::write(head_ptr.offset(index as isize), element);
            }
        }
        // Return box to tuple
        Ok(Term::make_boxed(tuple_ptr))
    }

    /// Constructs a `Tuple` from a slice of `Term`
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the slice provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `make_binary_from_str`.
    ///
    /// The resulting `Term` is a box pointing to the tuple header, and can itself be used in
    /// a slice passed to `make_tuple_from_slice` to produce nested tuples.
    fn tuple_from_slice(&mut self, elements: &[Term]) -> Result<Term, AllocErr> {
        self.tuple_from_slices(&[elements])
    }

    /// Constructs a `Tuple` from slices of `Term`
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the slice provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `make_binary_from_str`.
    ///
    /// The resulting `Term` is a box pointing to the tuple header, and can itself be used in
    /// a slice passed to `make_tuple_from_slice` to produce nested tuples.
    fn tuple_from_slices(&mut self, slices: &[&[Term]]) -> Result<Term, AllocErr> {
        let len = slices.iter().map(|slice| slice.len()).sum();
        let layout = Tuple::layout(len);
        let tuple_ptr = unsafe { self.alloc_layout(layout)?.as_ptr() as *mut Tuple };
        let head_ptr = unsafe { tuple_ptr.offset(1) as *mut Term };
        let tuple = Tuple::new(len);

        unsafe {
            // Write header
            ptr::write(tuple_ptr, tuple);
            let mut count = 0;

            // Write each element
            for slice in slices {
                for element in *slice {
                    ptr::write(head_ptr.offset(count), *element);
                    count += 1;
                }
            }
        }

        // Return box to tuple
        Ok(Term::make_boxed(tuple_ptr))
    }
}
impl<A, H> HeapAlloc for H
where
    A: HeapAlloc,
    H: DerefMut<Target = A>,
{
    #[inline]
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.deref_mut().alloc(need)
    }

    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> Result<NonNull<Term>, AllocErr> {
        self.deref_mut().alloc_layout(layout)
    }

    fn is_owner<T>(&mut self, ptr: *const T) -> bool {
        self.deref_mut().is_owner(ptr)
    }
}

fn str_from_binary_bytes<'heap>(bytes: &'heap [u8]) -> Result<&'heap str, StrFromBinaryError> {
    match core::str::from_utf8(bytes) {
        Ok(s) => Ok(unsafe { inherit_str_lifetime(s) }),
        Err(utf8_error) => Err(StrFromBinaryError::Utf8Error(utf8_error)),
    }
}

#[derive(Debug)]
pub enum MakePidError {
    Number,
    Serial,
    Alloc(AllocErr),
}

impl From<AllocErr> for MakePidError {
    fn from(alloc_err: AllocErr) -> Self {
        MakePidError::Alloc(alloc_err)
    }
}
impl From<pid::OutOfRange> for MakePidError {
    fn from(out_of_range: pid::OutOfRange) -> Self {
        match out_of_range {
            pid::OutOfRange::Number => MakePidError::Number,
            pid::OutOfRange::Serial => MakePidError::Serial,
        }
    }
}
