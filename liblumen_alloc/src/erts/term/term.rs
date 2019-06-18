#![allow(unused_parens)]

use core::cmp;
use core::mem;
use core::ptr;
use core::fmt;

use crate::borrow::CloneToProcess;
use crate::erts::{InvalidTermError, ProcessControlBlock};

use super::*;

macro_rules! unwrap_immediate1 {
    ($val:expr) => {
        ($val & !Self::MASK_IMMEDIATE1)
    };
}

macro_rules! unwrap_immediate2 {
    ($val:expr) => {
        ($val & !Self::MASK_IMMEDIATE2)
    };
}

/// This struct is a general tagged pointer type for Erlang terms.
///
/// It is generally equivalent to a reference, with the exception of
/// the class of terms called "immediates", which are not boxed values,
/// but stored in the tagged value itself.
///
/// For immediates, the high 6 bits of the value are used for tags, while
/// boxed values use the high 2 bits as tag, and point to either another box,
/// the NONE value, or a header word. The header word leaves the high 2 bits
/// zeroed, and the next 4 high bits (arityval) as tag for the type of object
/// the header is for. In some cases the header contains part of the value, in
/// others the value begins immediately following the header word.
///
/// The raw term value is tagged, so it is not something you should access directly
/// unless you know what you are doing. Most of the time use of that value is left
/// to the internals of `Term` itself.
///
/// Since `Term` values are often pointers, it should be given the same considerations
/// that you would give a raw pointer/reference anywhere else
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Term(usize);
impl Term {
    const PRIMARY_SHIFT: usize = (mem::size_of::<usize>() * 8) - 2;
    const IMMEDIATE1_SHIFT: usize = (mem::size_of::<usize>() * 8) - 4;
    const IMMEDIATE2_SHIFT: usize = (mem::size_of::<usize>() * 8) - 6;
    const HEADER_TAG_SHIFT: usize = (mem::size_of::<usize>() * 8) - 6;

    // Primary types
    pub const FLAG_HEADER: usize = 0;
    pub const FLAG_LIST: usize = 1 << Self::PRIMARY_SHIFT;
    pub const FLAG_BOXED: usize = 2 << Self::PRIMARY_SHIFT;
    pub const FLAG_IMMEDIATE: usize = 3 << Self::PRIMARY_SHIFT;
    pub const FLAG_IMMEDIATE2: usize = Self::FLAG_IMMEDIATE | (2 << Self::IMMEDIATE1_SHIFT);
    // NOTE: This flag is only used with BOXED and LIST terms, and indicates that the term
    // is a pointer to a literal, rather than a pointer to a term on the process heap/stack.
    // Literals are stored as constants in the compiled code, so these terms are never GCed.
    pub const FLAG_LITERAL: usize = 4 << Self::PRIMARY_SHIFT;
    // First class immediates
    pub const FLAG_PID: usize = 0 | Self::FLAG_IMMEDIATE;
    pub const FLAG_PORT: usize = (1 << Self::IMMEDIATE1_SHIFT) | Self::FLAG_IMMEDIATE;
    pub const FLAG_SMALL_INTEGER: usize = (3 << Self::IMMEDIATE1_SHIFT) | Self::FLAG_IMMEDIATE;
    // Second class immediates
    pub const FLAG_ATOM: usize = 0 | Self::FLAG_IMMEDIATE2;
    pub const FLAG_CATCH: usize = (1 << Self::IMMEDIATE2_SHIFT) | Self::FLAG_IMMEDIATE2;
    pub const FLAG_UNUSED_1: usize = (2 << Self::IMMEDIATE2_SHIFT) | Self::FLAG_IMMEDIATE2;
    pub const FLAG_NIL: usize = (3 << Self::IMMEDIATE2_SHIFT) | Self::FLAG_IMMEDIATE2;
    // Header types
    pub const FLAG_TUPLE: usize = 0 | Self::FLAG_HEADER;
    pub const FLAG_NONE: usize = (1 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_POS_BIG_INTEGER: usize = (2 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_NEG_BIG_INTEGER: usize = (3 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_REFERENCE: usize = (4 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_CLOSURE: usize = (5 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_FLOAT: usize = (6 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    #[allow(unused)]
    pub const FLAG_UNUSED_3: usize = (7 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_PROCBIN: usize = (8 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_HEAPBIN: usize = (9 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_SUBBINARY: usize = (10 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_MATCH_CTX: usize = (11 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_EXTERN_PID: usize = (12 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_EXTERN_PORT: usize = (13 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_EXTERN_REF: usize = (14 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;
    pub const FLAG_MAP: usize = (15 << Self::HEADER_TAG_SHIFT) | Self::FLAG_HEADER;

    // The primary tag is given by masking bits 0-2
    #[cfg(target_pointer_width = "64")]
    pub const MASK_PRIMARY: usize = 0xC000_0000_0000_0000;
    #[cfg(target_pointer_width = "32")]
    pub const MASK_PRIMARY: usize = 0xC000_0000;

    // First class immediate tags are given by masking bits 2-4
    #[cfg(target_pointer_width = "64")]
    pub const MASK_IMMEDIATE1_TAG: usize = 0x3000_0000_0000_0000;
    #[cfg(target_pointer_width = "32")]
    pub const MASK_IMMEDIATE1_TAG: usize = 0x3000_0000;

    // Second class immediate tags are given by masking bits 4-6
    #[cfg(target_pointer_width = "64")]
    pub const MASK_IMMEDIATE2_TAG: usize = 0x0C00_0000_0000_0000;
    #[cfg(target_pointer_width = "32")]
    pub const MASK_IMMEDIATE2_TAG: usize = 0x0C00_0000;

    // To mask off the entire immediate header, we mask off both the primary and immediate tag
    pub const MASK_IMMEDIATE1: usize = Self::MASK_PRIMARY | Self::MASK_IMMEDIATE1_TAG;
    pub const MASK_IMMEDIATE2: usize = Self::MASK_IMMEDIATE1 | Self::MASK_IMMEDIATE2_TAG;

    // Header is composed of 2 primary tag bits, and 4 subtag bits:
    pub const MASK_HEADER: usize = Self::MASK_HEADER_PRIMARY | Self::MASK_HEADER_ARITYVAL;
    // The primary tag is used to identify that a word is a header
    pub const MASK_HEADER_PRIMARY: usize = Self::MASK_PRIMARY;

    // The arityval is a subtag that identifies the boxed type
    // This value is used as a marker in some checks, but it is essentially equivalent
    // to `FLAG_TUPLE & !FLAG_HEADER`, which is simply the value 0
    pub const ARITYVAL: usize = 0;
    // The following is a mask for the actual arityval value
    #[cfg(target_pointer_width = "64")]
    pub const MASK_HEADER_ARITYVAL: usize = 0x3C00_0000_0000_0000;
    #[cfg(target_pointer_width = "32")]
    pub const MASK_HEADER_ARITYVAL: usize = 0x3C00_0000;

    /// The pattern 0b0101 out to usize bits, but with the header bits
    /// masked out, and flagged as the none value
    #[cfg(target_pointer_width = "64")]
    const NONE_VAL: usize = 0x155555554AAAAAAAusize & !Self::MASK_HEADER;
    #[cfg(target_pointer_width = "32")]
    const NONE_VAL: usize = 0x55555555usize & !Self::MASK_HEADER;

    /// Used to represent the absence of any meaningful value, in particular
    /// it is used by the process heap allocator/garbage collector
    pub const NONE: Self = Self(Self::NONE_VAL | Self::FLAG_NONE);
    /// Represents the catch flag
    pub const CATCH: Self = Self(Self::FLAG_CATCH);
    /// Represents the singleton nil value
    pub const NIL: Self = Self(Self::FLAG_NIL);

    #[allow(unused)]
    #[cfg(target_pointer_width = "64")]
    pub const MASK_EXTERN_PID_NODE: usize = 0x0000_0000_0000_000F;
    #[cfg(target_pointer_width = "32")]
    pub const MASK_EXTERN_PID_NODE: usize = 0x0000_000F;

    /// Creates a new `Term` from a raw term value.
    ///
    /// # Safety
    ///
    /// Be very careful when using this function, as incorrectly
    /// manufacturing the raw value will result in undefined behavior,
    /// especially if the tagging scheme is incorrectly applied.
    #[inline]
    pub unsafe fn from_raw(val: usize) -> Self {
        Self(val)
    }

    /// Executes the destructor for the underlying term, when the
    /// underlying term has a destructor which needs to run, such
    /// as `ProcBin`, which needs to be dropped in order to ensure
    /// that the reference count is decremented properly.
    /// 
    /// NOTE: This does not follow move markers, it is assumed that
    /// moved terms are live and not to be released
    #[inline]
    pub fn release(self) {
        // Follow boxed terms and release them
        if self.is_boxed() {
            let boxed_ptr = self.boxed_val();
            let boxed = unsafe { *boxed_ptr };
            // Do not follow moves
            if is_move_marker(boxed) {
                return;
            }
            // Ensure ref-counted binaries are released properly
            if boxed.is_procbin() {
                unsafe { ptr::drop_in_place(boxed_ptr as *mut ProcBin) };
                return;
            } 
            // Ensure we walk tuples and release all their elements
            if boxed.is_tuple() {
                let tuple = unsafe { &*(boxed_ptr as *mut Tuple) };
                for element in tuple.iter() {
                    element.release();
                }
                return;
            }
            return;
        } 
        // Ensure we walk lists and release all their elements
        if self.is_list() {
            let cons_ptr = self.list_val();
            let mut cons = unsafe { *cons_ptr };
            loop {
                // Do not follow moves
                if cons.is_move_marker() {
                    return;
                }
                // If we reached the end of the list, we're done
                if cons.head.is_nil() {
                    return;
                }
                // Otherwise release the head term
                cons.head.release();
                // This is more of a sanity check, as the head will be nil for EOL
                if cons.tail.is_nil() {
                    return;
                } 
                // If the tail is proper, move into the cell it represents
                if cons.tail.is_list() {
                    let tail_ptr = cons.tail.list_val();
                    cons = unsafe { *tail_ptr };
                    continue;
                }
                // Otherwise if the tail is improper, release it, and we're done
                cons.tail.release();
                return;
            }
        }
    }

    /// Casts the `Term` into its raw `usize` form
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0
    }

    /// Returns true if this term is the none value
    #[inline]
    pub fn is_none(&self) -> bool {
        self.eq(&Self::NONE)
    }

    /// Returns true if this term is nil
    #[inline]
    pub fn is_nil(&self) -> bool {
        self.eq(&Self::NIL)
    }

    /// Returns true if this term is a literal
    #[inline]
    pub fn is_literal(&self) -> bool {
        (self.is_boxed() || self.is_list()) && self.0 & Self::FLAG_LITERAL == Self::FLAG_LITERAL
    }

    /// Returns true if this term is a list
    #[inline]
    pub fn is_list(&self) -> bool {
        self.0 & Self::MASK_PRIMARY == Self::FLAG_LIST
    }

    /// Returns true if this term is an atom
    #[inline]
    pub fn is_atom(&self) -> bool {
        self.is_immediate2() && self.0 & Self::MASK_IMMEDIATE2 == Self::FLAG_ATOM
    }

    /// Returns true if this term is a number (float or integer)
    #[inline]
    pub fn is_number(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Returns true if this term is either a small or big integer
    #[inline]
    pub fn is_integer(&self) -> bool {
        self.is_smallint() || self.is_bigint()
    }

    /// Returns true if this term is a small integer (i.e. fits in a usize)
    #[inline]
    pub fn is_smallint(&self) -> bool {
        self.is_immediate() && self.0 & Self::MASK_IMMEDIATE1 == Self::FLAG_SMALL_INTEGER
    }

    /// Returns true if this term is a big integer (i.e. arbitrarily large)
    #[inline]
    pub fn is_bigint(&self) -> bool {
        match self.0 & Self::MASK_HEADER {
            Self::FLAG_POS_BIG_INTEGER | Self::FLAG_NEG_BIG_INTEGER => true,
            _ => false
        }
    }

    /// Returns true if this term is a float
    #[inline]
    pub fn is_float(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_FLOAT
    }

    /// Returns true if this term is a tuple
    #[inline]
    pub fn is_tuple(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_TUPLE
    }

    /// Returns true if this term is a tuple of arity `arity`
    #[inline]
    pub fn is_tuple_with_arity(&self, arity: usize) -> bool {
        if self.is_tuple() {
            let ptr = self.0 as *const Term;
            let header = unsafe { *ptr };
            header.arityval() == arity
        } else {
            false
        }
    }

    /// Returns true if this term is a map
    #[inline]
    pub fn is_map(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_MAP
    }

    /// Returns true if this term is a pid
    #[inline]
    pub fn is_pid(&self) -> bool {
        self.is_local_pid() || self.is_remote_pid()
    }

    /// Returns true if this term is a pid on the local node
    #[inline]
    pub fn is_local_pid(&self) -> bool {
        self.is_immediate() && self.0 & Self::MASK_IMMEDIATE1 == Self::FLAG_PID
    }

    /// Returns true if this term is a pid on some other node
    #[inline]
    pub fn is_remote_pid(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_EXTERN_PID
    }

    /// Returns true if this term is a pid
    #[inline]
    pub fn is_port(&self) -> bool {
        self.is_local_port() || self.is_remote_port()
    }

    /// Returns true if this term is a port on the local node
    #[inline]
    pub fn is_local_port(&self) -> bool {
        self.is_immediate() && self.0 & Self::MASK_IMMEDIATE1 == Self::FLAG_PORT
    }

    /// Returns true if this term is a port on some other node
    #[inline]
    pub fn is_remote_port(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_EXTERN_PORT
    }

    /// Returns true if this term is a reference
    #[inline]
    pub fn is_reference(&self) -> bool {
        self.is_local_reference() || self.is_remote_reference()
    }

    /// Returns true if this term is a reference on the local node
    #[inline]
    pub fn is_local_reference(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_REFERENCE
    }

    /// Returns true if this term is a reference on some other node
    #[inline]
    pub fn is_remote_reference(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_EXTERN_REF
    }

    /// Returns true if this term is a binary
    #[inline]
    pub fn is_binary(&self) -> bool {
        self.is_procbin() || self.is_heapbin() || self.is_subbinary() || self.is_match_context()
    }

    /// Returns true if this term is a reference-counted binary
    #[inline]
    pub fn is_procbin(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_PROCBIN
    }

    /// Returns true if this term is a binary on a process heap
    #[inline]
    pub fn is_heapbin(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_HEAPBIN
    }

    /// Returns true if this term is a sub-binary reference
    #[inline]
    pub fn is_subbinary(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_SUBBINARY
    }

    /// Returns true if this term is a binary match context
    #[inline]
    pub fn is_match_context(&self) -> bool {
        self.0 & Self::MASK_HEADER == Self::FLAG_MATCH_CTX
    }

    /// Returns true if this term is an immediate value
    #[inline]
    pub fn is_immediate(&self) -> bool {
        self.0 & Self::MASK_PRIMARY == Self::FLAG_IMMEDIATE
    }

    fn is_immediate2(&self) -> bool {
        self.is_immediate() && self.0 & Self::MASK_IMMEDIATE1 == Self::FLAG_IMMEDIATE2
    }

    /// Returns true if this term is a constant value
    ///
    /// NOTE: Currently the meaning of constant in this context is
    /// equivalent to that of immediates, i.e. an immediate is a constant,
    /// and only immediates are constants. Realistically we should be able
    /// to support constants of arbitrary term type, but this is derived
    /// from BEAM for now
    #[inline]
    pub fn is_const(&self) -> bool {
        self.is_immediate()
    }

    /// Returns true if this term is a boxed pointer
    #[inline]
    pub fn is_boxed(&self) -> bool {
        self.0 & Self::MASK_PRIMARY == Self::FLAG_BOXED
    }

    /// Returns true if this term is a header
    #[inline]
    pub fn is_header(&self) -> bool {
        self.0 & Self::MASK_PRIMARY == Self::FLAG_HEADER
    }

    /// Returns true if this term is a transparent header
    ///
    /// NOTE: This will panic if called on a non-header term
    #[inline]
    pub fn header_is_transparent(&self) -> bool {
        debug_assert!(self.is_header());
        self.0 & Self::MASK_HEADER_ARITYVAL == Self::ARITYVAL
    }

    /// An alias for `header_is_transparent` to better convey intended usage
    ///
    /// NOTE: This will panic if called on a non-header term
    #[inline]
    pub fn header_is_arityval(&self) -> bool {
        self.header_is_transparent()
    }

    /// Returns true if this term is a header for a non-transparent value
    ///
    /// NOTE: This function is safe to call on any term
    #[inline]
    pub fn is_thing(&self) -> bool {
        self.is_header() && self.header_is_thing()
    }

    /// Returns true if this term is a header for a non-transparent value
    ///
    /// NOTE: This will panic if called on a non-header term
    #[inline]
    pub fn header_is_thing(&self) -> bool {
        debug_assert!(self.is_header());
        !self.header_is_transparent()
    }

    /// Given a boxed term, this function returns a pointer to the boxed value
    ///
    /// NOTE: This is used internally by GC, you should use `to_typed_term` everywhere else
    #[inline]
    pub(crate) fn boxed_val(&self) -> *mut Term {
        assert!(self.is_boxed());
        unwrap_immediate1!(self.0) as *mut Term
    }

    /// Given a list term, this function returns a pointer to the underlying `Cons`
    ///
    /// NOTE: This is used internally by GC, you should use `to_typed_term` everywhere else
    #[inline]
    pub(crate) fn list_val(&self) -> *mut Cons {
        assert!(self.is_list());
        unwrap_immediate1!(self.0) as *mut Cons
    }

    /// Given a header term, this function returns the raw arity value,
    /// i.e. all flags are stripped, leaving the remaining bits untouched
    ///
    /// NOTE: This function will panic if called on anything but a header term
    #[inline]
    pub fn arityval(&self) -> usize {
        assert!(self.is_header());
        self.0 & !Self::MASK_HEADER
    }

    /// Given a pointer to a generic term, converts it to its typed representation
    pub fn to_typed_term(&self) -> Result<TypedTerm, InvalidTermError> {
        let val = self.0;
        match val & Self::MASK_PRIMARY {
            Self::FLAG_HEADER => unsafe { Self::header_to_typed_term(self, val) },
            Self::FLAG_LIST => {
                let ptr = unwrap_immediate1!(val) as *mut Cons;
                let is_literal = val & Self::FLAG_LITERAL == Self::FLAG_LITERAL;
                if is_literal {
                    Ok(TypedTerm::List(unsafe { Boxed::from_raw_literal(ptr) }))
                } else {
                    Ok(TypedTerm::List(unsafe { Boxed::from_raw(ptr) }))
                }
            }
            Self::FLAG_BOXED => {
                let ptr = unwrap_immediate1!(val) as *mut Term;
                let is_literal = val & Self::FLAG_LITERAL == Self::FLAG_LITERAL;
                if is_literal {
                    Ok(TypedTerm::Literal(unsafe { *ptr }))
                } else {
                    Ok(TypedTerm::Boxed(unsafe { *ptr }))
                }
            }
            Self::FLAG_IMMEDIATE => {
                match val & Self::MASK_IMMEDIATE1 {
                    Self::FLAG_PID => {
                        Ok(TypedTerm::Pid(unsafe { Pid::from_raw(unwrap_immediate1!(val)) }))
                    }
                    Self::FLAG_PORT => {
                        Ok(TypedTerm::Port(unsafe { Port::from_raw(unwrap_immediate1!(val)) }))
                    }
                    Self::FLAG_IMMEDIATE2 => {
                        match val & Self::MASK_IMMEDIATE2 {
                            Self::FLAG_ATOM => {
                                Ok(TypedTerm::Atom(unsafe { Atom::from_id(unwrap_immediate2!(val)) }))
                            }
                            Self::FLAG_CATCH => Ok(TypedTerm::Catch),
                            Self::FLAG_UNUSED_1 => Err(InvalidTermError::InvalidTag),
                            Self::FLAG_NIL => Ok(TypedTerm::Nil),
                            _ => Err(InvalidTermError::InvalidTag),
                        }
                    }
                    Self::FLAG_SMALL_INTEGER => {
                        let unwrapped = unwrap_immediate1!(val);
                        let small = unsafe { SmallInteger::from_untagged_term(unwrapped) };
                        Ok(TypedTerm::SmallInteger(small))
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    unsafe fn header_to_typed_term(ptr: &Term, val: usize) -> Result<TypedTerm, InvalidTermError> {
        let ptr = ptr as *const _ as *mut Term;
        let ty = match val & Self::MASK_HEADER {
            Self::FLAG_TUPLE => TypedTerm::Tuple(Boxed::from_raw(ptr as *mut Tuple)),
            Self::FLAG_NONE => {
                if val & !Self::MASK_HEADER == Self::NONE_VAL {
                    TypedTerm::None
                } else {
                    // Garbage value
                    return Err(InvalidTermError::InvalidTag);
                }
            }
            Self::FLAG_POS_BIG_INTEGER => {
                TypedTerm::BigInteger(Boxed::from_raw(ptr as *mut BigInteger))
            }
            Self::FLAG_NEG_BIG_INTEGER => {
                TypedTerm::BigInteger(Boxed::from_raw(ptr as *mut BigInteger))
            },
            Self::FLAG_REFERENCE => {
                TypedTerm::Reference(Reference::from_raw(ptr as *mut Reference))
            } // RefThing in erl_term.h
            Self::FLAG_CLOSURE => TypedTerm::Closure(Boxed::from_raw(ptr as *mut Closure)), /* ErlFunThing in erl_fun.h */
            Self::FLAG_FLOAT => TypedTerm::Float(Float::from_raw(ptr as *mut Float)),
            Self::FLAG_UNUSED_3 => return Err(InvalidTermError::InvalidTag),
            Self::FLAG_PROCBIN => TypedTerm::ProcBin(ProcBin::from_raw(ptr as *mut ProcBin)),
            Self::FLAG_HEAPBIN => TypedTerm::HeapBinary(HeapBin::from_raw(ptr as *mut HeapBin)),
            Self::FLAG_SUBBINARY => {
                TypedTerm::SubBinary(SubBinary::from_raw(ptr as *mut SubBinary))
            }
            Self::FLAG_MATCH_CTX => {
                TypedTerm::MatchContext(MatchContext::from_raw(ptr as *mut MatchContext))
            }
            Self::FLAG_EXTERN_PID => {
                TypedTerm::ExternalPid(Boxed::from_raw(ptr as *mut ExternalPid))
            }
            Self::FLAG_EXTERN_PORT => {
                TypedTerm::ExternalPort(Boxed::from_raw(ptr as *mut ExternalPort))
            }
            Self::FLAG_EXTERN_REF => {
                TypedTerm::ExternalReference(Boxed::from_raw(ptr as *mut ExternalReference))
            }
            Self::FLAG_MAP => TypedTerm::Map(Boxed::from_raw(ptr as *mut MapHeader)),
            _ => unreachable!(),
        };
        Ok(ty)
    }
}
impl fmt::Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if cfg!(target_pointer_width = "64") {
            write!(f, "Term({:064b})", self.0)
        } else {
            write!(f, "Term({:032b})", self.0)
        }
    }
}
impl PartialOrd<Term> for Term {
    fn partial_cmp(&self, other: &Term) -> Option<cmp::Ordering> {
        if let Ok(ref lhs) = self.to_typed_term() {
            if let Ok(ref rhs) = other.to_typed_term() {
                return lhs.partial_cmp(rhs);
            }
        }
        None
    }
}
impl CloneToProcess for Term {
    fn clone_to_process(&self, process: &mut ProcessControlBlock) -> Term {
        assert!(self.is_immediate() || self.is_boxed() || self.is_list());
        let tt = self.to_typed_term().unwrap();
        tt.clone_to_process(process)
    }
}
