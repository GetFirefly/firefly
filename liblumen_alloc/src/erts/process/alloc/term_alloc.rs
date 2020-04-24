use core::alloc::Layout;
use core::any::Any;
use core::ffi::c_void;
use core::ptr;
use core::str::Chars;

use alloc::sync::Arc;
use alloc::vec::Vec;

use hashbrown::HashMap;

use liblumen_core::util::reference::bytes;
use liblumen_core::util::reference::str::inherit_lifetime as inherit_str_lifetime;

use crate::borrow::CloneToProcess;
use crate::erts::exception::{AllocResult, InternalResult};
use crate::erts::module_function_arity::Arity;
use crate::erts::string::Encoding;
use crate::erts::term::closure::{Creator, Index, OldUnique, Unique};
use crate::erts::term::prelude::*;
use crate::erts::Node;
use crate::scheduler;

use super::{Heap, VirtualAllocator};

/// A trait, like `Alloc`, specifically for allocation of terms on a process heap
pub trait TermAlloc: Heap {
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
    /// NOTE: If allocation fails for some reason, `Err(Alloc)` is returned, this usually
    /// indicates that a process needs to be garbage collected, but in some cases may indicate
    /// that the global heap is out of space.
    fn binary_from_bytes(&mut self, bytes: &[u8]) -> AllocResult<Term>
    where
        Self: VirtualAllocator<ProcBin>,
    {
        let len = bytes.len();

        // Allocate ProcBins for sizes greater than 64 bytes
        if len > 64 {
            match self.procbin_from_bytes(bytes) {
                Err(error) => Err(error),
                Ok(bin_ptr) => {
                    // Add the binary to the process's virtual binary heap
                    self.virtual_alloc(bin_ptr);

                    Ok(bin_ptr.into())
                }
            }
        } else {
            self.heapbin_from_bytes(bytes).map(|nn| nn.into())
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
        Self: VirtualAllocator<ProcBin>,
    {
        match binary.decode().unwrap() {
            TypedTerm::HeapBinary(bin_ptr) => {
                Ok(unsafe { bytes::inherit_lifetime(bin_ptr.as_ref().as_bytes()) })
            }
            TypedTerm::ProcBin(bin) => Ok(unsafe { bytes::inherit_lifetime(bin.as_bytes()) }),
            TypedTerm::BinaryLiteral(bin) => Ok(unsafe { bytes::inherit_lifetime(bin.as_bytes()) }),
            TypedTerm::SubBinary(bin) => {
                if bin.is_binary() {
                    if bin.bit_offset() == 0 {
                        Ok(unsafe { bytes::inherit_lifetime(bin.as_bytes_unchecked()) })
                    } else {
                        let aligned_byte_vec: Vec<u8> = bin.full_byte_iter().collect();
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
    /// NOTE: If allocation fails for some reason, `Err(Alloc)` is returned, this usually
    /// indicates that a process needs to be garbage collected, but in some cases may indicate
    /// that the global heap is out of space.
    fn binary_from_str(&mut self, s: &str) -> AllocResult<Term>
    where
        Self: VirtualAllocator<ProcBin>,
    {
        let len = s.len();
        // Allocate ProcBins for sizes greater than 64 bytes
        if len > HeapBin::MAX_SIZE {
            match self.procbin_from_str(s) {
                Err(error) => Err(error),
                Ok(bin_ptr) => {
                    // Add the binary to the process's virtual binary heap
                    self.virtual_alloc(bin_ptr);
                    Ok(bin_ptr.into())
                }
            }
        } else {
            self.heapbin_from_str(s).map(|nn| nn.into())
        }
    }

    /// Constructs an integer value from any type that implements `Into<Integer>`,
    /// which currently includes `SmallInteger`, `BigInteger`, `usize` and `isize`.
    ///
    /// This operation will transparently handle constructing the correct type of term
    /// based on the input value, i.e. an immediate small integer for values that fit,
    /// else a heap-allocated big integer for larger values.
    fn integer<I: Into<Integer>>(&mut self, i: I) -> AllocResult<Term>
    where
        Self: Sized,
    {
        match i.into() {
            Integer::Small(small) => Ok(small.into()),
            Integer::Big(big) => big.clone_to_heap(self),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn float(&mut self, f: f64) -> AllocResult<Float> {
        Ok(Float::new(f))
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn float(&mut self, f: f64) -> AllocResult<Boxed<Float>> {
        let float = Float::new(f);

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<Float>())?.as_ptr() as *mut Float;
            ptr.write(float);

            Ok(Boxed::new_unchecked(ptr))
        }
    }

    /// Constructs a list of only the head and tail, and associated with the given process.
    fn cons<T, U>(&mut self, head: T, tail: U) -> AllocResult<Boxed<Cons>>
    where
        T: Into<Term>,
        U: Into<Term>,
    {
        let cons = Cons::new(head.into(), tail.into());

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<Cons>())?.as_ptr() as *mut Cons;
            ptr.write(cons);

            Ok(Boxed::new_unchecked(ptr))
        }
    }

    fn improper_list_from_iter<I>(
        &mut self,
        iter: I,
        last: Term,
    ) -> AllocResult<Option<Boxed<Cons>>>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        let mut head: *mut Cons = ptr::null_mut();
        let mut acc: Term = last;

        for element in iter.rev() {
            head = self.cons(element, acc)?.as_ptr();
            acc = head.into();
        }

        if head.is_null() {
            // There were no elements in the iterator,
            // which actually makes this a proper list
            if last.is_nil() {
                // Empty list, no elements
                return Ok(None);
            } else if last.is_non_empty_list() {
                // We were given a cons cell as the last element,
                // so just return that
                let tail: Boxed<Cons> = last.dyn_cast();
                return Ok(Some(tail));
            } else {
                // Just a single element list
                return Ok(Some(self.cons(acc, Term::NIL)?));
            }
        }

        Ok(Boxed::new(head))
    }

    fn improper_list_from_slice(
        &mut self,
        slice: &[Term],
        tail: Term,
    ) -> AllocResult<Option<Boxed<Cons>>> {
        self.improper_list_from_iter(slice.iter().copied(), tail)
    }

    fn charlist_from_str(&mut self, s: &str) -> AllocResult<Option<Boxed<Cons>>>
    where
        Self: Sized,
    {
        self.list_from_chars(s.chars())
    }

    /// Constructs a list from the chars and associated with the given process.
    fn list_from_chars(&mut self, chars: Chars) -> AllocResult<Option<Boxed<Cons>>>
    where
        Self: Sized,
    {
        let mut head: *mut Cons = ptr::null_mut();
        let mut acc = Term::NIL;

        for character in chars.rev() {
            let codepoint = self.integer(character)?;
            head = self.cons(codepoint, acc)?.as_ptr();
            acc = head.into();
        }

        Ok(Boxed::new(head))
    }

    fn list_from_iter<I>(&mut self, iter: I) -> AllocResult<Option<Boxed<Cons>>>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        self.improper_list_from_iter(iter, Term::NIL)
    }

    fn list_from_slice(&mut self, slice: &[Term]) -> AllocResult<Option<Boxed<Cons>>> {
        self.improper_list_from_slice(slice, Term::NIL)
    }

    /// Constructs a map and associated with the given process.
    fn map_from_hash_map(&mut self, hash_map: HashMap<Term, Term>) -> AllocResult<Boxed<Map>>
    where
        Self: Sized,
    {
        let boxed = Map::from_hash_map(hash_map).clone_to_heap(self)?;
        let ptr: Boxed<Map> = boxed.dyn_cast();
        Ok(ptr)
    }

    /// Constructs a map and associated with the given process.
    fn map_from_slice(&mut self, slice: &[(Term, Term)]) -> AllocResult<Boxed<Map>>
    where
        Self: Sized,
    {
        let boxed = Map::from_slice(slice).clone_to_heap(self)?;
        let ptr: Boxed<Map> = boxed.dyn_cast();
        Ok(ptr)
    }

    #[inline]
    fn local_pid_with_node_id(
        &mut self,
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> InternalResult<Pid> {
        assert_eq!(node_id, 0);

        Ok(Pid::new(number, serial)?)
    }

    fn external_pid(
        &mut self,
        arc_node: Arc<Node>,
        number: usize,
        serial: usize,
    ) -> InternalResult<Boxed<ExternalPid>>
    where
        Self: Sized,
    {
        let pid = ExternalPid::new(arc_node, number, serial)?.clone_to_heap(self)?;
        let boxed: *mut ExternalPid = pid.dyn_cast();

        Ok(unsafe { Boxed::new_unchecked(boxed) })
    }

    /// Constructs a heap-allocated binary from the given byte slice, and associated with the given
    /// process
    #[inline]
    fn heapbin_from_bytes(&mut self, s: &[u8]) -> AllocResult<Boxed<HeapBin>> {
        HeapBin::from_slice(self, s, Encoding::Raw)
    }

    /// Constructs a heap-allocated binary from the given string, and associated with the given
    /// process
    #[inline]
    fn heapbin_from_str(&mut self, s: &str) -> AllocResult<Boxed<HeapBin>> {
        HeapBin::from_str(self, s)
    }

    /// Constructs a reference-counted binary from the given byte slice, and associated with the
    /// given process
    fn procbin_from_bytes(&mut self, s: &[u8]) -> AllocResult<Boxed<ProcBin>> {
        // Allocates on global heap
        let bin = ProcBin::from_slice(s, Encoding::Raw)?;
        unsafe {
            // Allocates space on the process heap for the header
            let ptr = self.alloc_layout(Layout::new::<ProcBin>())?.as_ptr() as *mut ProcBin;
            // Write the header to the process heap
            ptr.write(bin);
            Ok(Boxed::new_unchecked(ptr))
        }
    }

    /// Constructs a reference-counted binary from the given string, and associated with the given
    /// process
    fn procbin_from_str(&mut self, s: &str) -> AllocResult<Boxed<ProcBin>> {
        // Allocates on global heap
        let bin = ProcBin::from_str(s)?;
        unsafe {
            // Allocates space on the process heap for the header
            let ptr = self.alloc_layout(Layout::new::<ProcBin>())?.as_ptr() as *mut ProcBin;
            // Write the header to the process heap
            ptr.write(bin);
            Ok(Boxed::new_unchecked(ptr))
        }
    }

    /// Creates a `Reference` with the given `number` associated with the Process.
    fn reference(
        &mut self,
        scheduler_id: scheduler::ID,
        number: ReferenceNumber,
    ) -> AllocResult<Boxed<Reference>> {
        let layout = Reference::layout();
        let reference_ptr = unsafe { self.alloc_layout(layout)?.as_ptr() as *mut Reference };
        let reference = Reference::new(scheduler_id, number);

        unsafe {
            // Write header
            reference_ptr.write(reference);
            Ok(Boxed::new_unchecked(reference_ptr))
        }
    }

    fn resource(&mut self, value: Box<dyn Any>) -> AllocResult<Boxed<Resource>>
    where
        Self: Sized,
    {
        Resource::from_value(self, value)
    }

    /// Either returns a `&str` to the pre-existing bytes in the heap binary, process binary, or
    /// aligned subbinary or creates a new aligned binary and returns the bytes from that new
    /// binary.
    fn str_from_binary<'heap>(
        &'heap mut self,
        binary: Term,
    ) -> Result<&'heap str, StrFromBinaryError>
    where
        Self: VirtualAllocator<ProcBin>,
    {
        let bytes = self.bytes_from_binary(binary)?;

        str_from_binary_bytes(bytes)
    }

    /// Constructs a subbinary from the given original, and associated with the given process.
    ///
    /// `original` must be a heap binary or a process binary.  To take the subbinary of a subbinary,
    /// use the first subbinary's original instead and combine the offsets.
    ///
    /// NOTE: If allocation fails for some reason, `Err(Alloc)` is returned, this usually
    /// indicates that a process needs to be garbage collected, but in some cases may indicate
    /// that the global heap is out of space.
    fn subbinary_from_original(
        &mut self,
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        full_byte_len: usize,
        partial_byte_bit_len: u8,
    ) -> AllocResult<Boxed<SubBinary>> {
        let subbinary = SubBinary::from_original(
            original,
            byte_offset,
            bit_offset,
            full_byte_len,
            partial_byte_bit_len,
        );

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<SubBinary>())?.as_ptr() as *mut SubBinary;
            ptr.write(subbinary);

            Ok(Boxed::new_unchecked(ptr))
        }
    }

    /// Constructs a match context from some boxed binary term
    fn match_context_from_binary<B>(&mut self, binary: Boxed<B>) -> AllocResult<Boxed<MatchContext>>
    where
        B: ?Sized + Bitstring + Encode<Term>,
    {
        let match_ctx = MatchContext::new(binary.into());

        unsafe {
            let ptr =
                self.alloc_layout(Layout::new::<MatchContext>())?.as_ptr() as *mut MatchContext;
            ptr.write(match_ctx);

            Ok(Boxed::new_unchecked(ptr))
        }
    }

    /// Constructs a `Tuple` that needs to be filled with elements and then boxed.
    fn mut_tuple(&mut self, len: usize) -> AllocResult<Boxed<Tuple>> {
        Tuple::new(self, len)
    }

    /// Constructs a `Tuple` from an `Iterator<Item = Term>` and accompanying `len`.
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the `iterator` provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `binary_from_str`.
    fn tuple_from_iter<I>(&mut self, iterator: I, len: usize) -> AllocResult<Boxed<Tuple>>
    where
        I: Iterator<Item = Term>,
    {
        let mut tuple_box = Tuple::new(self, len)?;
        let tuple_ref = tuple_box.as_mut();
        let elements = tuple_ref.elements_mut();
        let mut elements_ptr = elements.as_mut_ptr();

        // Write each element
        let mut count = 0;
        for (index, element) in iterator.enumerate() {
            assert!(
                index < len,
                "unexpected out of bounds access in tuple_from_iter: len = {}, index = {}",
                len,
                index
            );
            unsafe {
                elements_ptr.write(element);
                elements_ptr = elements_ptr.offset(1);
            }
            count += 1;
        }
        debug_assert_eq!(
            len, count,
            "expected number of elements in iterator to match provided length"
        );

        Ok(tuple_box)
    }

    /// Constructs a `Tuple` from a slice of `Term`
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the slice provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `binary_from_str`.
    ///
    /// The resulting `Term` is a box pointing to the tuple header, and can itself be used in
    /// a slice passed to `tuple_from_slice` to produce nested tuples.
    fn tuple_from_slice(&mut self, elements: &[Term]) -> AllocResult<Boxed<Tuple>> {
        Tuple::from_slice(self, elements)
    }

    /// Constructs a `Tuple` from slices of `Term`
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the slice provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `binary_from_str`.
    ///
    /// The resulting `Term` is a box pointing to the tuple header, and can itself be used in
    /// a slice passed to `tuple_from_slice` to produce nested tuples.
    fn tuple_from_slices(&mut self, slices: &[&[Term]]) -> AllocResult<Boxed<Tuple>> {
        let len = slices.iter().map(|slice| slice.len()).sum();
        let mut tuple_box = Tuple::new(self, len)?;

        unsafe {
            let tuple_ref = tuple_box.as_mut();
            let elements = tuple_ref.elements_mut();
            let mut elements_ptr = elements.as_mut_ptr();

            // Write each element
            for slice in slices {
                for element in *slice {
                    elements_ptr.write(*element);
                    elements_ptr = elements_ptr.offset(1);
                }
            }
        }

        Ok(tuple_box)
    }

    /// Constructs a `Closure` from a slice of `Term`
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the slice provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `binary_from_str`.
    ///
    /// The resulting `Term` is a box pointing to the closure header, and can itself be used in
    /// a slice passed to `closure_with_env_from_slice` to produce nested closures or tuples.
    fn anonymous_closure_with_env_from_slice(
        &mut self,
        module: Atom,
        index: Index,
        old_unique: OldUnique,
        unique: Unique,
        arity: Arity,
        native: Option<*const c_void>,
        creator: Creator,
        slice: &[Term],
    ) -> AllocResult<Boxed<Closure>> {
        Closure::from_slice(
            self, module, index, old_unique, unique, arity, native, creator, slice,
        )
    }

    /// Constructs a `Closure` from slices of `Term`
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the slice provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `binary_from_str`.
    ///
    /// The resulting `Term` is a box pointing to the closure header, and can itself be used in
    /// a slice passed to `closure_with_env_from_slice` to produce nested closures or tuples.
    fn anonymous_closure_with_env_from_slices(
        &mut self,
        module: Atom,
        index: Index,
        old_unique: OldUnique,
        unique: Unique,
        arity: Arity,
        native: Option<*const c_void>,
        creator: Creator,
        slices: &[&[Term]],
    ) -> AllocResult<Boxed<Closure>> {
        let len = slices.iter().map(|slice| slice.len()).sum();
        let mut closure_box = Closure::new_anonymous(
            self, module, index, old_unique, unique, arity, native, creator, len,
        )?;

        unsafe {
            let closure_ref = closure_box.as_mut();
            let env_slice = closure_ref.env_slice_mut();
            let mut env_ptr = env_slice.as_mut_ptr();

            // Write each element
            for slice in slices {
                for element in *slice {
                    env_ptr.write(*element);
                    env_ptr = env_ptr.offset(1);
                }
            }
        }

        Ok(closure_box)
    }

    fn export_closure(
        &mut self,
        module: Atom,
        function: Atom,
        arity: u8,
        native: Option<*const c_void>,
    ) -> AllocResult<Boxed<Closure>> {
        Closure::new_export(self, module, function, arity, native)
    }
}

impl<T> TermAlloc for T where T: Heap {}

impl<T, H> TermAlloc for T
where
    H: TermAlloc,
    T: core::ops::DerefMut<Target = H>,
{
}

fn str_from_binary_bytes<'heap>(bytes: &'heap [u8]) -> Result<&'heap str, StrFromBinaryError> {
    match core::str::from_utf8(bytes) {
        Ok(s) => Ok(unsafe { inherit_str_lifetime(s) }),
        Err(utf8_error) => Err(StrFromBinaryError::Utf8Error(utf8_error)),
    }
}
