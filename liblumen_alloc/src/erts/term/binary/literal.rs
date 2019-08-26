use core::alloc::Layout;
use core::cmp;
use core::convert::TryInto;
use core::fmt::{self, Debug};
use core::mem;
use core::ptr;
use core::slice;
use core::str;

use alloc::borrow::ToOwned;
use alloc::string::String;

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
use crate::erts::string::Encoding;
use crate::erts::term::{AsTerm, Boxed, Term, to_word_size};
use crate::erts::string;
use crate::erts::HeapAlloc;

use super::aligned_binary::AlignedBinary;
use super::constants;
use super::constants::{FLAG_IS_RAW_BIN, FLAG_IS_LATIN1_BIN, FLAG_IS_UTF8_BIN, FLAG_IS_LITERAL};
use super::{ProcBin, HeapBin, Original, SubBinary, MatchContext, Bitstring};

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
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BinaryLiteral {
    pub(super) header: Term,
    pub(crate) flags: usize,
    bytes: *mut u8,
}
impl Debug for BinaryLiteral {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BinaryLiteral")
            .field("header", &self.header)
            .field("flags", &format_args!("{:#b}", self.flags))
            .field("bytes", &self.bytes)
            .finish()
    }
}
impl BinaryLiteral {
    #[inline]
    pub unsafe fn from_raw(term: *mut BinaryLiteral) -> Self {
        *term
    }

    #[inline]
    pub fn from_raw_bytes(bytes: *mut u8, size: usize) -> Self {
        let arityval = to_word_size(mem::size_of::<Self>() - mem::size_of::<Term>());
        Self {
            header: Term::make_header(arityval, Term::FLAG_PROCBIN),
            flags: (size << 1) | FLAG_IS_RAW_BIN | FLAG_IS_LITERAL,
            bytes,
        }
    }


    #[inline]
    pub fn from_raw_latin1_bytes(bytes: *mut u8, size: usize) -> Self {
        let arityval = to_word_size(mem::size_of::<Self>() - mem::size_of::<Term>());
        Self {
            header: Term::make_header(arityval, Term::FLAG_PROCBIN),
            flags: (size << 1) | FLAG_IS_LATIN1_BIN | FLAG_IS_LITERAL,
            bytes,
        }
    }

    #[inline]
    pub fn from_raw_utf8_bytes(bytes: *mut u8, size: usize) -> Self {
        let arityval = to_word_size(mem::size_of::<Self>() - mem::size_of::<Term>());
        Self {
            header: Term::make_header(arityval, Term::FLAG_PROCBIN),
            flags: (size << 1) | FLAG_IS_UTF8_BIN | FLAG_IS_LITERAL,
            bytes,
        }
    }

    #[inline]
    pub fn make_arch64_parts_from_str(s: &'static str) -> (u64, u64) {
        use crate::erts::term::arch::arch64;
        let arityval = 2;
        let header = arch64::make_header(arityval, arch64::FLAG_PROCBIN);
        let flags = if string::is_latin1(s) {
            constants::arch64::FLAG_IS_LATIN1_BIN | constants::arch64::FLAG_IS_LITERAL
        } else {
            constants::arch64::FLAG_IS_UTF8_BIN | constants::arch64::FLAG_IS_LITERAL
        };
        let size = s.len() as u64;
        (header, (size << 1) | flags)
    }

    #[inline]
    pub fn make_arch32_parts_from_str(s: &'static str) -> (u32, u32) {
        use crate::erts::term::arch::arch32;
        let arityval = 2;
        let header = arch32::make_header(arityval, arch32::FLAG_PROCBIN);
        let flags = if string::is_latin1(s) {
            constants::arch32::FLAG_IS_LATIN1_BIN | constants::arch32::FLAG_IS_LATIN1_BIN
        } else {
            constants::arch32::FLAG_IS_UTF8_BIN | constants::arch32::FLAG_IS_LITERAL
        };
        let size = s.len() as u32;
        (header, (size << 1) | flags)
    }


    #[inline]
    pub fn make_arch64_parts_from_slice(s: &'static [u8]) -> (u64, u64) {
        use crate::erts::term::arch::arch64;
        let arityval = 2;
        let header = arch64::make_header(arityval, arch64::FLAG_PROCBIN);
        let size = s.len() as u64;
        (header, (size << 1) | constants::arch64::FLAG_IS_RAW_BIN | constants::arch64::FLAG_IS_LITERAL)
    }

    #[inline]
    pub fn make_arch32_parts_from_slice(s: &'static [u8]) -> (u32, u32) {
        use crate::erts::term::arch::arch32;
        let arityval = 2;
        let header = arch32::make_header(arityval, arch32::FLAG_PROCBIN);
        let size = s.len() as u32;
        (header, (size << 1) | constants::arch32::FLAG_IS_RAW_BIN | constants::arch32::FLAG_IS_LITERAL)
    }

    #[inline]
    pub fn bytes(&self) -> *mut u8 {
        self.bytes
    }

    #[inline]
    fn full_byte_len(&self) -> usize {
        (self.flags & !constants::FLAG_MASK) >> 1
    }

    #[inline]
    pub fn encoding(&self) -> Encoding {
        super::encoding_from_flags(self.flags)
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        self.flags & constants::BIN_TYPE_MASK == FLAG_IS_RAW_BIN
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.flags & constants::BIN_TYPE_MASK == FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.flags & constants::BIN_TYPE_MASK == FLAG_IS_UTF8_BIN
    }

    /// Converts this binary to a `&str` slice.
    ///
    /// This conversion does not move the string, it can be considered as
    /// creating a new reference with a lifetime attached to that of `self`.
    ///
    /// Due to the fact that the lifetime of the `&str` cannot outlive the
    /// `ProcBin`, this does not require incrementing the reference count.
    #[allow(unused)]
    pub fn as_str<'a>(&'a self) -> &'a str {
        assert!(
            self.is_latin1() || self.is_utf8(),
            "cannot convert a binary containing non-UTF-8/non-ASCII characters to &str"
        );
        unsafe {
            let bytes = self.as_bytes();
            str::from_utf8_unchecked(bytes)
        }
    }
}

impl Eq for BinaryLiteral {}

impl PartialEq<Boxed<HeapBin>> for BinaryLiteral {
    fn eq(&self, other: &Boxed<HeapBin>) -> bool {
        self.eq(other.as_ref())
    }
}

impl PartialEq<MatchContext> for BinaryLiteral {
    fn eq(&self, other: &MatchContext) -> bool {
        other.eq(self)
    }
}

impl PartialEq<SubBinary> for BinaryLiteral {
    fn eq(&self, other: &SubBinary) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<ProcBin> for BinaryLiteral {
    fn partial_cmp(&self, other: &ProcBin) -> Option<core::cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

impl PartialOrd<HeapBin> for BinaryLiteral {
    /// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
    /// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
    fn partial_cmp(&self, other: &HeapBin) -> Option<core::cmp::Ordering> {
        self.as_bytes().partial_cmp(other.as_bytes())
    }
}

impl PartialOrd<Boxed<HeapBin>> for BinaryLiteral {
    fn partial_cmp(&self, other: &Boxed<HeapBin>) -> Option<cmp::Ordering> {
        self.partial_cmp(other.as_ref())
    }
}

impl PartialOrd<MatchContext> for BinaryLiteral {
    fn partial_cmp(&self, other: &MatchContext) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|ordering| ordering.reverse())
    }
}

impl PartialOrd<SubBinary> for BinaryLiteral {
    fn partial_cmp(&self, other: &SubBinary) -> Option<cmp::Ordering> {
        other.partial_cmp(self).map(|ordering| ordering.reverse())
    }
}

impl TryInto<String> for BinaryLiteral {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        if self.is_latin1() || self.is_utf8() {
            Ok(self.as_str().to_owned())
        } else {
            match str::from_utf8(self.as_bytes()) {
                Ok(s) => Ok(s.to_owned()),
                Err(_) => Err(badarg!()),
            }
        }
    }
}

unsafe impl AsTerm for BinaryLiteral {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed_literal(self as *const Self)
    }
}

impl AlignedBinary for BinaryLiteral {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.bytes(), self.full_byte_len())
        }
    }
}

impl Bitstring for BinaryLiteral {
    fn full_byte_len(&self) -> usize {
        self.full_byte_len()
    }
}

impl Original for BinaryLiteral {
    fn byte(&self, index: usize) -> u8 {
        let full_byte_len = self.full_byte_len();

        assert!(
            index < full_byte_len,
            "index ({}) >= full_byte_len ({})",
            index,
            full_byte_len
        );

        unsafe { *self.bytes().add(index) }
    }
}

impl CloneToProcess for BinaryLiteral {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        unsafe {
            // Allocate space for the header
            let layout = Layout::new::<Self>();
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            // Write the binary header
            ptr::write(ptr, self.clone());
            // Reify result term
            Ok(Term::make_boxed_literal(ptr))
        }
    }
}
