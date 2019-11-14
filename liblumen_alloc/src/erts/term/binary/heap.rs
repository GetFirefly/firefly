use core::alloc::Layout;
use core::convert::TryFrom;
use core::ptr;
use core::slice;
use core::str;
use core::iter;
use core::mem;

use crate::borrow::CloneToProcess;
use crate::erts::{self, HeapAlloc};
use crate::erts::exception::AllocResult;
use crate::erts::string::Encoding;
use crate::erts::term::prelude::*;
use crate::erts::term::encoding::Header;

use super::prelude::*;


/// Process heap allocated binary, smaller than 64 bytes
#[derive(Debug)]
#[repr(C)]
pub struct HeapBin {
    header: Header<HeapBin>,
    flags: BinaryFlags,
    data: [u8]
}
impl_dynamic_header!(HeapBin, Term::HEADER_HEAPBIN);
impl HeapBin {
    pub const MAX_SIZE: usize = 64;

    /// Creates a new `HeapBin` from a str slice, by copying it to the heap
    pub fn from_str<A>(heap: &mut A, s: &str) -> AllocResult<Boxed<Self>>
    where
        A: ?Sized + HeapAlloc,
    {
        let encoding = Encoding::from_str(s);

        Self::from_slice(heap, s.as_bytes(), encoding)
    }

    /// Creates a new `HeapBin` from a byte slice, by copying it to the heap
    pub fn from_slice<A>(heap: &mut A, s: &[u8], encoding: Encoding) -> AllocResult<Boxed<Self>>
    where
        A: ?Sized + HeapAlloc,
    {
        let (layout, flags_offset, data_offset) = Self::layout_for(s);

        unsafe {
            match heap.alloc_layout(layout) {
                Ok(non_null) => {
                    Ok(Self::copy_slice_to_internal(non_null.as_ptr() as *mut u8, s, encoding, flags_offset, data_offset))
                }
                Err(_) => Err(alloc!()),
            }
        }
    }

    // This function handles the low-level parts of creating a `HeapBin` at the given pointer
    #[inline]
    unsafe fn copy_slice_to_internal(dst: *mut u8, s: &[u8], encoding: Encoding, flags_offset: usize, data_offset: usize) -> Boxed<Self> {
        let len = s.len();
        // Write header
        let arity = erts::to_word_size(len + mem::size_of::<BinaryFlags>());
        let header = Header::from_arity(arity);
        ptr::write(dst as *mut Header<HeapBin>, header);
        let flags_ptr = dst.offset(flags_offset as isize) as *mut BinaryFlags;
        let flags = BinaryFlags::new(encoding).set_size(len);
        ptr::write(flags_ptr, flags);
        let data_ptr = dst.add(data_offset);
        ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, len);

        Self::from_raw_parts(dst, len)
    }

    fn layout_for(s: &[u8]) -> (Layout, usize, usize) {
        let (base_layout, flags_offset) = Layout::new::<Header<HeapBin>>()
            .extend(Layout::new::<BinaryFlags>())
            .unwrap();
        let (unpadded_layout, data_offset) = base_layout
            .extend(Layout::for_value(s))
            .unwrap();
        // We pad to alignment so that the Layout produced here
        // matches that returned by `Layout::for_value` on the
        // final `HeapBin`
        let layout = unpadded_layout
            .pad_to_align()
            .unwrap();

        (layout, flags_offset, data_offset)
    }

    #[inline]
    pub fn full_byte_iter<'a>(&'a self) -> iter::Copied<slice::Iter<'a, u8>> {
        self.data.iter().copied()
    }

    /// Given a raw pointer to some memory, and a length in units of `Self::Element`,
    /// this function produces a fat pointer to `Self` which represents a value
    /// containing `len` elements in its variable-length field
    ///
    /// For example, given a pointer to the memory containing a `Tuple`, and the
    /// number of elements it contains, this function produces a valid pointer of
    /// type `Tuple`
    unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Boxed<HeapBin> {
        // Invariants of slice::from_raw_parts.
        assert!(!ptr.is_null());
        assert!(len <= isize::max_value() as usize);

        let slice = core::slice::from_raw_parts_mut(ptr as *mut u8, len);
        let ptr = slice as *mut [u8] as *mut HeapBin;
        Boxed::new_unchecked(ptr)
    }
}

impl From<*mut Term> for Boxed<HeapBin> {
    fn from(ptr: *mut Term) -> Boxed<HeapBin> {
        unsafe { HeapBin::from_raw_term(ptr) }
    }
}
impl Into<*mut Term> for Boxed<HeapBin> {
    fn into(self) -> *mut Term {
        self.cast::<Term>().as_ptr()
    }
}

impl<E: crate::erts::term::arch::Repr> Boxable<E> for HeapBin {}

impl<E: crate::erts::term::arch::Repr> UnsizedBoxable<E> for HeapBin {
    unsafe fn from_raw_term(ptr: *mut E) -> Boxed<HeapBin> {
        let header_ptr = ptr as *mut Header<HeapBin>;
        // Get precise size in bytes of binary data
        let flags = &*(header_ptr.add(1) as *mut BinaryFlags);
        let bin_len = flags.get_size();

        Self::from_raw_parts(ptr as *const u8, bin_len)
    }
}

impl Bitstring for HeapBin {
    #[inline]
    fn full_byte_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    unsafe fn as_byte_ptr(&self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }
}

impl Binary for HeapBin {
    #[inline]
    fn flags(&self) -> &BinaryFlags {
        &self.flags
    }
}

impl AlignedBinary for HeapBin {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl MaybePartialByte for HeapBin {
    #[inline]
    fn partial_byte_bit_len(&self) -> u8 {
        0
    }

    #[inline]
    fn total_bit_len(&self) -> usize {
        self.full_byte_len() * 8
    }

    #[inline]
    fn total_byte_len(&self) -> usize {
        self.full_byte_len()
    }
}

impl IndexByte for HeapBin {
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.data[index]
    }
}

impl CloneToProcess for HeapBin {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + HeapAlloc,
    {
        let encoding = self.encoding();
        let ptr = HeapBin::from_slice(heap, &self.data, encoding)?;
        Ok(ptr.into())
    }

    fn size_in_words(&self) -> usize {
        self.header.arity()
    }
}

impl TryFrom<TypedTerm> for Boxed<HeapBin> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::HeapBinary(bin_ptr) => Ok(bin_ptr),
            _ => Err(TypeError),
        }
    }
}
