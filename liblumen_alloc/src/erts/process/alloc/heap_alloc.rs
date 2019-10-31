use core::alloc::Layout;
use core::any::Any;
use core::ops::DerefMut;
use core::ptr::{self, NonNull};
use core::str::Chars;
use core::mem;
use core::cmp;

use alloc::sync::Arc;
use alloc::vec::Vec;

use hashbrown::HashMap;

use liblumen_core::sys::sysconf::MIN_ALIGN;
use liblumen_core::util::reference::bytes;
use liblumen_core::util::reference::str::inherit_lifetime as inherit_str_lifetime;

use crate::{ModuleFunctionArity, VirtualAlloc};
use crate::scheduler;
use crate::borrow::CloneToProcess;
use crate::erts::exception::{self, AllocResult};
use crate::erts::process::code::Code;
use crate::erts::string::Encoding;
use crate::erts::term::prelude::*;

/// A trait, like `Alloc`, specifically for allocation of terms on a process heap
pub trait HeapAlloc {
    /// Perform a heap allocation.
    ///
    /// If space on the process heap is not immediately available, then the allocation
    /// will be pushed into a heap fragment which will then be later moved on to the
    /// process heap during garbage collection
    unsafe fn alloc(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        let align = cmp::max(mem::align_of::<Term>(), MIN_ALIGN);
        let size = need * mem::size_of::<Term>();
        let layout = Layout::from_size_align(size, align).unwrap();
        self.alloc_layout(layout)
    }

    /// Same as `alloc`, but takes a `Layout` rather than the size in words
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>>;

    /// Returns true if the given pointer is owned by this process/heap
    fn is_owner<T>(&mut self, ptr: *const T) -> bool where T: ?Sized;

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
        Self: VirtualAlloc,
    {
        let len = bytes.len();

        // Allocate ProcBins for sizes greater than 64 bytes
        if len > 64 {
            match self.procbin_from_bytes(bytes) {
                Err(error) => Err(error),
                Ok(bin_ptr) => {
                    // Add the binary to the process's virtual binary heap
                    let bin = bin_ptr.as_ref();
                    self.virtual_alloc(bin);

                    Ok(bin_ptr.into())
                }
            }
        } else {
            self.heapbin_from_bytes(bytes)
                .map(|nn| nn.into())
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
        match binary.decode().unwrap() {
            TypedTerm::HeapBinary(bin_ptr) => {
                Ok(unsafe { bytes::inherit_lifetime(bin_ptr.as_ref().as_bytes()) })
            }
            TypedTerm::ProcBin(bin) => {
                Ok(unsafe { bytes::inherit_lifetime(bin.as_bytes()) })
            }
            TypedTerm::BinaryLiteral(bin) => {
                Ok(unsafe { bytes::inherit_lifetime(bin.as_bytes()) })
            }
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
        Self: VirtualAlloc,
    {
        let len = s.len();
        // Allocate ProcBins for sizes greater than 64 bytes
        if len > HeapBin::MAX_SIZE {
            match self.procbin_from_str(s) {
                Err(error) => Err(error),
                Ok(bin_ptr) => {
                    // Add the binary to the process's virtual binary heap
                    let bin = bin_ptr.as_ref();
                    self.virtual_alloc(bin);
                    Ok(bin_ptr.into())
                }
            }
        } else {
            self.heapbin_from_str(s)
                .map(|nn| nn.into())
        }
    }

    /// Constructs a list of only the head and tail, and associated with the given process.
    fn cons(&mut self, head: Term, tail: Term) -> AllocResult<Term> {
        let cons = Cons::new(head, tail);

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<Cons>())?.as_ptr() as *mut Cons;
            ptr.write(cons);

            Ok(ptr.into())
        }
    }

    fn external_pid_with_node_id(
        &mut self,
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> exception::Result<Term>
    where
        Self: Sized,
    {
        let external_pid = ExternalPid::with_node_id(node_id, number, serial)?;
        let heap_external_pid = external_pid.clone_to_heap(self)?;

        Ok(heap_external_pid)
    }

    #[cfg(target_arch = "x86_64")]
    fn float(&mut self, f: f64) -> AllocResult<Term> {
        Ok(Float::new(f).encode().unwrap())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn float(&mut self, f: f64) -> AllocResult<Term> {
        let float = Float::new(f);

        unsafe {
            let ptr = self.alloc_layout(Layout::new::<Float>())?.as_ptr() as *mut Float;
            ptr.write(float);

            Ok(ptr.into())
        }
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

    fn improper_list_from_iter<I>(&mut self, iter: I, last: Term) -> AllocResult<Term>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        let mut acc = last;

        for element in iter.rev() {
            acc = self.cons(element, acc)?;
        }

        Ok(acc)
    }

    fn improper_list_from_slice(&mut self, slice: &[Term], tail: Term) -> AllocResult<Term> {
        self.improper_list_from_iter(slice.iter().map(|t| *t), tail)
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

    fn charlist_from_str(&mut self, s: &str) -> AllocResult<Term>
    where
        Self: Sized,
    {
        self.list_from_chars(s.chars())
    }

    /// Constructs a list from the chars and associated with the given process.
    fn list_from_chars(&mut self, chars: Chars) -> AllocResult<Term>
    where
        Self: Sized,
    {
        let mut acc = Term::NIL;

        for character in chars.rev() {
            let code_point = self.integer(character)?;

            acc = self.cons(code_point, acc)?;
        }

        Ok(acc)
    }

    fn list_from_iter<I>(&mut self, iter: I) -> AllocResult<Term>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        self.improper_list_from_iter(iter, Term::NIL)
    }

    fn list_from_slice(&mut self, slice: &[Term]) -> AllocResult<Term> {
        self.improper_list_from_slice(slice, Term::NIL)
    }

    /// Constructs a map and associated with the given process.
    fn map_from_hash_map(&mut self, hash_map: HashMap<Term, Term>) -> AllocResult<Term>
    where
        Self: Sized,
    {
        Map::from_hash_map(hash_map).clone_to_heap(self)
    }

    /// Constructs a map and associated with the given process.
    fn map_from_slice(&mut self, slice: &[(Term, Term)]) -> AllocResult<Term>
    where
        Self: Sized,
    {
        Map::from_slice(slice).clone_to_heap(self)
    }

    /// Creates a `Pid` or `ExternalPid` with the given `node`, `number` and `serial`.
    fn pid_with_node_id(
        &mut self,
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> exception::Result<Term>
    where
        Self: Sized,
    {
        if node_id == 0 {
            let pid = Pid::new(number, serial)?
                .encode()
                .unwrap();
            Ok(pid)
        } else {
            self.external_pid_with_node_id(node_id, number, serial)
        }
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

    fn resource(&mut self, value: Box<dyn Any>) -> AllocResult<Term>
    where
        Self: Sized,
    {
        Ok(Resource::from_value(self, value)?.into())
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
            assert!(index < len, "unexpected out of bounds access in tuple_from_iter: len = {}, index = {}", len, index);
            unsafe {
                elements_ptr.write(element);
                elements_ptr = elements_ptr.offset(1);
            }
            count += 1;
        }
        debug_assert_eq!(len, count, "expected number of elements in iterator to match provided length");

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
    fn closure_with_env_from_slice(
        &mut self,
        mfa: Arc<ModuleFunctionArity>,
        code: Code,
        creator: Term,
        slice: &[Term],
    ) -> AllocResult<Boxed<Closure>> {
        Closure::from_slice(self, mfa, code, creator, slice)
    }

    /// Constructs a `Closure` from slices of `Term`
    ///
    /// Be aware that this does not allocate non-immediate terms in `elements` on the process heap,
    /// it is expected that the slice provided is constructed from either immediate terms, or
    /// terms which were returned from other constructor functions, e.g. `binary_from_str`.
    ///
    /// The resulting `Term` is a box pointing to the closure header, and can itself be used in
    /// a slice passed to `closure_with_env_from_slice` to produce nested closures or tuples.
    fn closure_with_env_from_slices(
        &mut self,
        mfa: Arc<ModuleFunctionArity>,
        code: Code,
        creator: Term,
        slices: &[&[Term]],
    ) -> AllocResult<Boxed<Closure>> {
        let len = slices.iter().map(|slice| slice.len()).sum();
        let mut closure_box = Closure::new(self, mfa, code, creator, len)?;

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
}
impl<A, H> HeapAlloc for H
where
    A: HeapAlloc,
    H: DerefMut<Target = A>,
{
    #[inline]
    unsafe fn alloc(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        self.deref_mut().alloc(need)
    }

    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        self.deref_mut().alloc_layout(layout)
    }

    fn is_owner<T>(&mut self, ptr: *const T) -> bool where T: ?Sized {
        self.deref_mut().is_owner(ptr)
    }
}

fn str_from_binary_bytes<'heap>(bytes: &'heap [u8]) -> Result<&'heap str, StrFromBinaryError> {
    match core::str::from_utf8(bytes) {
        Ok(s) => Ok(unsafe { inherit_str_lifetime(s) }),
        Err(utf8_error) => Err(StrFromBinaryError::Utf8Error(utf8_error)),
    }
}
