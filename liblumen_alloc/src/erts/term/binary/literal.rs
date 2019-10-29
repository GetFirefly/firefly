use core::alloc::Layout;
use core::ptr;
use core::slice;
use core::iter;
use core::str;

use liblumen_core::offset_of;

use crate::borrow::CloneToProcess;
use crate::erts::HeapAlloc;
use crate::erts::exception::AllocResult;
use crate::erts::string::{self, Encoding};
use crate::erts::term::prelude::Term;
use crate::erts::term::encoding::Header;

use super::prelude::*;

/// This struct is used to represent binary literals which are compiled into
/// the final executable. At runtime, we identify them by the fact that they
/// are boxed literals, which point to a header flagged as a ProcBin. In these
/// cases, rather than reifying a ProcBin, we reify a BinaryLiteral instead.
///
/// We are able to do this because the header struct of BinaryLiteral and ProcBin are
/// structured so that the header field comes first, and the flags field (a usize)
/// comes second in BinaryLiteral while the field containing the pointer to the
/// ProcBinInner comes second in ProcBin. Since pointers are always 8-byte aligned on
/// all target platforms, BinaryLiteral sets the lowest bit to 1 in its flags field,
/// allowing us to unambiguously identify BinaryLiteral from ProcBin. Due to this, the
/// byte size of a BinaryLiteral is shifted left one bit, unlike the other binary types.
///
/// BinaryLiterals always have an effective static lifetime, as the underlying
/// bytes are stored as constant values in the executable. They can not be modified,
/// and cloning is extremely cheap as it is effectively the same as cloning a pointer.
///
/// Since they share the same header tag as a ProcBin, some care is necessary to ensure
/// that a BinaryLiteral is not ever reified as a ProcBin, but as all references to a
/// BinaryLiteral occur through a box flagged as literal, this is easily avoided. Likewise,
/// the garbage collector does not ever try to interact with literals, beyond copying the
/// references to them.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BinaryLiteral {
    header: Header<BinaryLiteral>,
    flags: BinaryFlags,
    bytes: *mut u8,
}
impl BinaryLiteral {
    #[inline]
    pub fn flags_offset() -> usize {
        offset_of!(BinaryLiteral, flags)
    }

    #[inline]
    pub fn from_raw_bytes(bytes: *mut u8, size: usize, encoding: Option<Encoding>) -> Self {
        let flags = BinaryFlags::new_literal(encoding.unwrap_or(Encoding::Raw))
            .set_size(size);
        Self {
            header: Default::default(),
            flags,
            bytes,
        }
    }

    #[inline]
    pub fn make_arch64_parts_from_str(s: &'static str) -> (u64, u64) {
        use crate::erts::term::arch::arch_64;
        let header = arch_64::RawTerm::HEADER_PROCBIN;
        let encoding = if string::is_latin1(s) {
            Encoding::Latin1
        } else {
            Encoding::Utf8
        };
        let flags = BinaryFlags::new_literal(encoding)
            .set_size(s.len())
            .as_u64();
        (header, flags)
    }

    #[inline]
    pub fn make_arch32_parts_from_str(s: &'static str) -> (u32, u32) {
        use crate::erts::term::arch::arch_32;
        let header = arch_32::RawTerm::HEADER_PROCBIN;
        let encoding = if string::is_latin1(s) {
            Encoding::Latin1
        } else {
            Encoding::Utf8
        };
        let flags = BinaryFlags::new_literal(encoding)
            .set_size(s.len())
            .as_u32();
        (header, flags)
    }


    #[inline]
    pub fn make_arch64_parts_from_slice(s: &'static [u8]) -> (u64, u64) {
        use crate::erts::term::arch::arch_64;
        let header = arch_64::RawTerm::HEADER_PROCBIN;
        let flags = BinaryFlags::new_literal(Encoding::Raw)
            .set_size(s.len())
            .as_u64();
        (header, flags)
    }

    #[inline]
    pub fn make_arch32_parts_from_slice(s: &'static [u8]) -> (u32, u32) {
        use crate::erts::term::arch::arch_32;
        let header = arch_32::RawTerm::HEADER_PROCBIN;
        let flags = BinaryFlags::new_literal(Encoding::Raw)
            .set_size(s.len())
            .as_u32();
        (header, flags)
    }

    #[inline]
    pub fn full_byte_iter<'a>(&'a self) -> iter::Copied<slice::Iter<'a, u8>> {
        self.as_bytes().iter().copied()
    }
}
impl Bitstring for BinaryLiteral {
    #[inline]
    fn full_byte_len(&self) -> usize {
        self.flags.get_size()
    }

    #[inline]
    unsafe fn as_byte_ptr(&self) -> *mut u8 {
        self.bytes
    }
}
impl Binary for BinaryLiteral {
    #[inline]
    fn flags(&self) -> &BinaryFlags {
        &self.flags
    }
}
impl AlignedBinary for BinaryLiteral {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.bytes, self.full_byte_len())
        }
    }
}
impl IndexByte for BinaryLiteral {
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.as_bytes()[index]
    }
}

impl CloneToProcess for BinaryLiteral {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + HeapAlloc,
    {
        unsafe {
            // Allocate space for the header
            let layout = Layout::new::<Self>();
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            // Write the binary header
            ptr::write(ptr, self.clone());
            // Reify result term
            Ok(ptr.into())
        }
    }
}
