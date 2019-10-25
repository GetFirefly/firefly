use core::alloc::Layout;
use core::convert::{TryFrom, TryInto};
use core::ptr;
use core::slice;
use core::str;
use core::iter;

use alloc::borrow::ToOwned;
use alloc::string::String;

use crate::borrow::CloneToProcess;
use crate::erts::{self, HeapAlloc};
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
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
impl HeapBin {
    pub const MAX_SIZE: usize = 64;

    /// Constructs a reference to a `HeapBin` given a pointer to
    /// the memory containing the struct and the length of its variable-length
    /// data
    ///
    /// # Implementation Details
    ///
    /// You should note that this struct is a dynamically-sized type (DST), and
    /// as such, the rules around how you can use it is far more restrictive
    /// than a typical statically sized struct. Notably, you can only ever
    /// construct a reference to a HeapBin, you can't create a HeapBin directly;
    /// clearly this seems like a chicken and egg problem since it is mostly
    /// meaningless to construct references to things you can't create in the
    /// first place.
    ///
    /// So you may be asking "how the hell does this even work?!", which is a
    /// great question! Luckily it has a fairly straightforward, yet mind-bendy
    /// explanation:
    ///
    /// If we redefined `HeapBin` as `HeapBin<T>`, where the `data` field has
    /// type `T`, then we could say that the structural layout of `[HeapBin<u8>; 1]`
    /// is equivalent to that of `HeapBin<[u8; 1]>`. This should make intuitive
    /// sense, as the variable-length part of the structure occurs at the end, and
    /// the layout of `[T; 1]` is equivalent to the layout of `T` itself.
    ///
    /// So given this structural equivalence, the second piece of the puzzle is
    /// found in unsizing coercions performed by Rust. If we create a slice from
    /// a pointer to the memory where the `HeapBin` is allocated, and the length
    /// of the variable-sized component of that `HeapBin`; and then cast that
    /// slice to a `*const HeapBin` - Rust will treat this as a coercion to an
    /// unsized type, due to the fact that the `HeapBin` struct contains an unsized
    /// field, and it is the last field in the struct [1].
    ///
    /// In effect, the pointer of our first slice acts as the base pointer for the
    /// struct, and the layout rules for the sized fields are as you would expect.
    /// Where things get confusing is that the length we gave to `slice::from_raw_parts`
    /// is used as the length of the unsized slice at the end (i.e. the `data` field);
    /// this is due to the unsizing coercion performed by Rust.
    ///
    /// - [1](https://github.com/rust-lang/rfcs/blob/master/text/0982-dst-coercion.md)
    /// - [2](http://doc.rust-lang.org/1.38.0/std/marker/trait.Unsize.html)
    #[inline]
    pub(in crate::erts::term) unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Boxed<Self> {
        // Invariants of slice::from_raw_parts.
        assert!(!ptr.is_null());
        assert!(len <= isize::max_value() as usize);

        let slice = core::slice::from_raw_parts(ptr as *const (), len);
        Boxed::new_unchecked(slice as *const [()] as *mut Self)
    }

    /// Reifies a reference to a `HeapBin` from a pointer to `Term`
    ///
    /// It is expected that the `Term` on the other end is the header
    #[inline]
    pub unsafe fn from_raw_term(term: *mut Term) -> Boxed<Self> {
        let header = &*(term as *mut Header<HeapBin>);
        let arity = header.arity();

        Self::from_raw_parts(term as *const u8, arity)
    }

    /// Creates a new `HeapBin` from a str slice, by copying it to the heap
    pub fn from_str<A>(heap: &mut A, s: &str) -> Result<Boxed<Self>, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        let encoding = Encoding::from_str(s);

        Self::from_slice(heap, s.as_bytes(), encoding)
    }

    /// Creates a new `HeapBin` from a byte slice, by copying it to the heap
    pub fn from_slice<A>(heap: &mut A, s: &[u8], encoding: Encoding) -> Result<Boxed<Self>, Alloc> 
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

    /// Creates a new `HeapBin` at `dst` by copying the given slice.
    ///
    /// This function is very unsafe, and is currently only used in the garbage collector when
    /// moving data that gets copied into a heap-allocated binary
    pub(in crate::erts) unsafe fn copy_slice_to(dst: *mut u8, s: &[u8], encoding: Encoding) -> Boxed<Self> {
        let (_layout, flags_offset, data_offset) = Self::layout_for(s);

        Self::copy_slice_to_internal(dst, s, encoding, flags_offset, data_offset)
    }

    // This function handles the low-level parts of creating a `HeapBin` at the given pointer
    #[inline]
    unsafe fn copy_slice_to_internal(dst: *mut u8, s: &[u8], encoding: Encoding, flags_offset: usize, data_offset: usize) -> Boxed<Self> {
        let len = s.len();
        // Write header
        let header = Header::from_arity(len);
        ptr::write(dst as *mut Header<HeapBin>, header);
        let flags_ptr = dst.offset(flags_offset as isize) as *mut BinaryFlags;
        ptr::write(flags_ptr, BinaryFlags::new(encoding));
        let data_ptr = dst.offset(data_offset as isize);
        ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, len);

        Self::from_raw_parts(dst, len)
    }

    fn layout_for(s: &[u8]) -> (Layout, usize, usize) {
        let (base_layout, flags_offset) = Layout::new::<Term>()
            .extend(Layout::new::<usize>())
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

impl IndexByte for HeapBin {
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.data[index]
    }
}

impl CloneToProcess for HeapBin {
    fn clone_to_heap<A>(&self, heap: &mut A) -> Result<Term, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        let encoding = self.encoding();
        let ptr = HeapBin::from_slice(heap, &self.data, encoding)?;
        Ok(ptr.into())
    }

    fn size_in_words(&self) -> usize {
        let layout = Layout::for_value(self);
        erts::to_word_size(layout.size())
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

impl TryInto<String> for &HeapBin {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        match str::from_utf8(self.as_bytes()) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => Err(badarg!()),
        }
    }
}

impl TryInto<Vec<u8>> for &HeapBin {
    type Error = runtime::Exception;

    #[inline]
    fn try_into(self) -> Result<Vec<u8>, Self::Error> {
        Ok(self.as_bytes().to_vec())
    }
}
