use core::alloc::Layout;
use core::fmt::{self, Debug};
use core::mem;
use core::hash;
use core::marker::PhantomData;
use core::convert::TryInto;

use hashbrown::HashMap;
use thiserror::Error;

use crate::borrow::CloneToProcess;
use crate::erts::{self, HeapAlloc};
use crate::erts::exception::{AllocResult, Result};

use super::prelude::*;
use super::arch::Word;

/// Represents the various conditions under which encoding can fail
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermEncodingError {
    /// Occurs when attempting to encode an unaligned pointer
    #[error("invalid attempt to encode unaligned pointer")]
    InvalidAlignment,
    /// Occurs when attempting to encode a value that cannot fit
    /// within the encoded types valid range, e.g. an integer that
    /// is too large
    #[error("invalid attempt to encode a value outside the valid range")]
    ValueOutOfRange,
}

/// Used to indicate that some value was not a valid encoding of a term
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermDecodingError {
    /// Occurs primarily with header tags, as there are some combinations
    /// of tag bits that are unused. Primary tags are (currently) fully
    /// occupied, and so invalid tags are not representable. That may
    /// change if additional representations are added that have free
    /// primary tags
    #[error("invalid type tag")]
    InvalidTag,
    /// Decoding a Term that is a move marker is considered an error,
    /// code which needs to be aware of move markers should already be
    /// manually checking for this possibility
    #[error("tried to decode a move marker")]
    MoveMarker,
    /// Decoding a Term that is a none value is considered an error,
    /// as this value is primarily used to indicate an error; and in cases
    /// where it represents a real value (namely as a move marker), it is
    /// meaningless to decode it. In all other cases, it represents uninitialized
    /// memory
    #[error("tried to decode a none value")]
    NoneValue,
}

/// This trait provides the ability to encode a type to some output encoding.
///
/// More specifically with regard to terms, it encodes immediates directly,
/// or for boxed terms, encodes the header. For boxed terms it is then
/// necessary to use the `Boxable::as_box` trait function to obtain an
/// encoded pointer when needed.
pub trait Encode<T: Encoded> {
    fn encode(&self) -> Result<T>;
}

/// This is a marker trait for terms which can be boxed
pub trait Boxable<T: Encoded> {}

/// This is a marker trait for boxed terms which are stored as literals
pub trait Literal<T: Encoded> : Boxable<T> {}

/// Boxable terms require a header term to be built during
/// construction of the type, which specifies the type and
/// size in words of the non-header portion of the data
/// structure.
///
/// This struct is a safe abstraction over that header to
/// ensure that architecture-specific details do not leak
/// out of the `arch` or `encoding` modules.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Header<T: ?Sized> {
    value: Term,
    _phantom: PhantomData<T>,
}
impl<T: Sized> Header<T> {
    /// Returns the statically known base size (in bytes) of any value of type `T`
    #[allow(unused)]
    #[inline]
    fn static_size() -> usize {
        mem::size_of::<T>()
    }

    /// Returns the statically known base arity (in words) of any value of type `T`
    #[inline]
    fn static_arity() -> usize {
        mem::size_of::<T>() - mem::size_of::<Self>()
    }
}
impl<T: ?Sized> Header<T> {
    /// Returns the size in bytes of the value this header represents, including the header
    #[inline]
    pub fn size(&self) -> usize {
        mem::size_of::<Term>() * (self.arity() + 1)
    }

    /// Returns the size in words of the value this header represents, including the header
    #[inline]
    pub fn size_in_words(&self) -> usize {
        erts::to_word_size(self.size())
    }

    /// Returns the size in words of the value this header represents, not including the header
    #[inline]
    pub fn arity(&self) -> usize {
        unsafe { self.value.arity() }
    }

    #[inline]
    fn to_word_size(size: usize) -> usize {
        use liblumen_core::alloc::alloc_utils::round_up_to_multiple_of;

        round_up_to_multiple_of(size, mem::size_of::<Term>()) / mem::size_of::<Term>()
    }
}
impl<T: ?Sized> Debug for Header<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::erts::term::arch::Repr;

        let name = core::any::type_name::<T>();
        write!(f, "Header<{}>({:#b})", name, self.value.as_usize())
    }
}
const_assert_eq!(mem::size_of::<Header<usize>>(), mem::size_of::<usize>());
impl Header<Map> {
    pub fn from_map(map: &HashMap<Term, Term>) -> Self {
        use crate::erts::term::arch::Repr;

        // NOTE: This size only accounts for the HashMap header, not the values
        let layout = Layout::for_value(map);
        let map_size = layout.size();
        let arity = Self::static_arity() + Self::to_word_size(map_size);
        let value = Term::encode_header(arity.try_into().unwrap(), Term::HEADER_MAP);
        Self { value, _phantom: PhantomData }
    }
}

/// This is a marker trait for dynamically-sized types which have headers
pub trait DynamicHeader {
    /// The header tag associated with this type
    const TAG: Word;
}
impl DynamicHeader for Closure { const TAG: Word = Term::HEADER_CLOSURE; }
impl DynamicHeader for Tuple { const TAG: Word = Term::HEADER_TUPLE; }
impl DynamicHeader for HeapBin { const TAG: Word = Term::HEADER_HEAPBIN; }
impl<T: ?Sized + DynamicHeader> Header<T> {
    pub fn from_arity(arity: usize) -> Self {
        use crate::erts::term::arch::Repr;

        let value = Term::encode_header(arity.try_into().unwrap(), T::TAG);
        Self { value, _phantom: PhantomData }
    }
}

/// This is a marker trait for types which have headers that can be statically derived
pub trait StaticHeader {
    /// The header tag associated with this type
    const TAG: Word;
}

impl StaticHeader for BigInteger { const TAG: Word = Term::HEADER_BIG_INTEGER; }
impl StaticHeader for ExternalPid { const TAG: Word = Term::HEADER_EXTERN_PID; }
impl StaticHeader for ExternalPort { const TAG: Word = Term::HEADER_EXTERN_PORT; }
impl StaticHeader for ExternalReference { const TAG: Word = Term::HEADER_EXTERN_REF; }
impl StaticHeader for Reference { const TAG: Word = Term::HEADER_REFERENCE; }
impl StaticHeader for Resource { const TAG: Word = Term::HEADER_RESOURCE_REFERENCE; }
impl StaticHeader for BinaryLiteral { const TAG: Word = Term::HEADER_BINARY_LITERAL; }
impl StaticHeader for ProcBin { const TAG: Word = Term::HEADER_PROCBIN; }
impl StaticHeader for SubBinary { const TAG: Word = Term::HEADER_SUBBINARY; }
impl StaticHeader for MatchContext { const TAG: Word = Term::HEADER_MATCH_CTX; }
impl<T: StaticHeader> Default for Header<T> {
    fn default() -> Self {
        use crate::erts::term::arch::Repr;

        let arity = Self::to_word_size(Self::static_arity());
        let value = Term::encode_header(arity.try_into().unwrap(), T::TAG);
        Self { value, _phantom: PhantomData }
    }
}


/// This trait is used to implement unsafe dynamic casts from an implementation
/// of `Encoded` to some type `T`.
///
/// Currently this is used to implement `dyn_cast` from `Term` to `*const T`,
/// where `T` is typically either `Term` or `Cons`.
///
/// NOTE: Care should be taken to avoid implementing this unless absolutely necessary,
/// it is mostly intended for use cases where the type of a `Encoded` value has already been
/// validated, and we don't want to go through the `TypedTerm` machinery for some reason.
/// As a general rule, use safer methods for extracting values from an `Encoded`, as this
/// defers all safety checks to runtime when used.
pub trait Cast<T>: Encoded {
    /// Perform a dynamic cast from `Self` to `T`
    ///
    /// It is expected that implementations of this function will assert that `self` is
    /// actually a value of the destination type when concrete. However, it is allowed
    /// to implement this generically for pointers, but should be considered very unsafe
    /// as there is no way to protect against treating some region of memory as the wrong
    /// type when used in that way. Always prefer safer alternatives where possible.
    fn dyn_cast(self) -> T;
}

/// This trait defines the common API for low-level term representations, i.e. `Term`.
/// It contains all common functions for working directly with encoded terms where the
/// implementation of those functions may depend on platform/architecture details.
///
/// Since terms may provide greater or fewer immediate types based on platform restrictions,
/// it is necessary for each representation to define these common functions in order to prevent
/// tying higher-level code to low-level details such as the specific bit-width of a
/// machine word.
///
/// NOTE: This trait requires that implementations implement `Copy` because higher-level code
/// currently depends on those semantics. Some functions, such as `decode`, take a reference
/// to `self` to prevent copying the original term in cases where the location in memory is
/// important. However, several functions of this trait do not, and it is assumed that those
/// functions are not dependent on a specific memory address. If that constraint is violated
pub trait Encoded: Sized + Copy + Send + PartialEq<Self> + PartialOrd<Self> + Ord + hash::Hash {
    /// Decodes `Self` into a `TypedTerm`, unless the encoded value is
    /// invalid or malformed.
    ///
    /// NOTE: Implementations should attempt to catch all possible decoding errors
    /// to make this as safe as possible. The only exception to this rule should
    /// be the case of decoding a pointer which can not be validated unless it
    /// is dereferenced.
    fn decode(&self) -> Result<TypedTerm>;

    /// Returns `true` if the encoded value represents `NONE`
    fn is_none(self) -> bool;
    /// Returns `true` if the encoded value represents a pointer to a term
    fn is_boxed(self) -> bool;
    /// Returns `true` if the encoded value is the header of a non-immediate term
    fn is_header(self) -> bool;
    /// Returns `true` if the encoded value is an immediate value
    fn is_immediate(self) -> bool;
    /// Returns `true` if the encoded value represents a pointer to a literal value
    fn is_literal(self) -> bool;

    /// Returns `true` if the encoded value represents the empty list
    fn is_nil(self) -> bool;
    /// Returns `true` if the encoded value represents a `Cons` value (non-empty list)
    fn is_list(self) -> bool;
    /// This is an alias for `is_cons` which better expresses intent in some instances
    fn is_non_empty_list(self) -> bool {
        self.is_list()
    }
    /// Returns `true` if the encoded value is an atom
    fn is_atom(self) -> bool;
    /// Returns `true` if the encoded value is a boolean
    fn is_boolean(self) -> bool {
        if !self.is_atom() {
            return false;
        }
        match self.decode() {
            Ok(TypedTerm::Atom(a)) => a.is_boolean(),
            _ => false,
        }
    }
    /// Returns `true` if the encoded value is a fixed-width integer value
    fn is_smallint(self) -> bool;
    /// Returns `true` if the encoded value is a arbitrary-width integer value
    fn is_bigint(self) -> bool;
    /// Returns `true` if the encoded value is an integer of either
    /// fixed or arbitrary width
    fn is_integer(&self) -> bool {
        if self.is_smallint() {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::BigInteger(_)) => true,
            _ => false
        }
    }
    /// Returns `true` if the encoded value is a float
    ///
    /// NOTE: This function returns true if either the term is an immediate float,
    /// or if it is the header of a packed float. It does not unwrap boxed values.
    fn is_float(self) -> bool;
    /// Returns `true` if the encoded value is either a float or integer
    fn is_number(self) -> bool {
        match self.decode() {
            Ok(TypedTerm::SmallInteger(_))
            | Ok(TypedTerm::BigInteger(_))
            | Ok(TypedTerm::Float(_)) => true,
            _ => false
        }
    }
    /// Returns `true` if the encoded value is the header of a `Tuple`
    fn is_tuple(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `Map`
    fn is_map(self) -> bool;
    /// Returns `true` if the encoded value is a `Pid`
    fn is_local_pid(self) -> bool;
    /// Returns `true` if the encoded value is the header of an `ExternalPid`
    fn is_remote_pid(self) -> bool;
    /// Returns `true` if the encoded value is a `Port`
    fn is_local_port(self) -> bool;
    /// Returns `true` if the encoded value is the header of an `ExternalPort`
    fn is_remote_port(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `Reference`
    fn is_local_reference(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `ExternalReference`
    fn is_remote_reference(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `Resource`
    fn is_resource_reference(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `ProcBin`
    fn is_procbin(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `HeapBin`
    fn is_heapbin(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `SubBinary`
    fn is_subbinary(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `MatchContext`
    fn is_match_context(self) -> bool;
    /// Returns `true` if the encoded value is the header of a `Closure`
    fn is_function(self) -> bool;

    /// Returns `true` if this term is a bitstring type,
    /// where the number of bits is evenly divisible by 8 (i.e. one byte)
    fn is_binary(&self) -> bool {
        match self.decode().expect("invalid term") {
            TypedTerm::HeapBinary(_)
            | TypedTerm::ProcBin(_)
            | TypedTerm::BinaryLiteral(_) => true,
            TypedTerm::SubBinary(bin) => bin.partial_byte_bit_len() == 0,
            TypedTerm::MatchContext(bin) => bin.partial_byte_bit_len() == 0,
            _ => false,
        }
    }

    /// Returns `true` if this term is a bitstring type
    fn is_bitstring(&self) -> bool {
        match self.decode().expect("invalid term") {
            TypedTerm::HeapBinary(_)
            | TypedTerm::ProcBin(_)
            | TypedTerm::BinaryLiteral(_)
            | TypedTerm::SubBinary(_)
            | TypedTerm::MatchContext(_) => true,
            _ => false,
        }
    }

    /// Returns true if this term is a pid type
    fn is_pid(&self) -> bool {
        if self.is_local_pid() {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::ExternalPid(_)) => true,
            _ => false
        }
    }

    /// Returns true if this term is a port type
    fn is_port(&self) -> bool {
        if self.is_local_port() {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::ExternalPort(_)) => true,
            _ => false
        }
    }

    /// Returns true if this term is a reference type
    fn is_reference(&self) -> bool {
        match self.decode() {
            Ok(TypedTerm::Reference(_))
                | Ok(TypedTerm::ExternalReference(_))
                | Ok(TypedTerm::ResourceReference(_)) => true,
            _ => false
        }
    }

    /// Returns true if this a term that the runtime should accept as an argument.
    fn is_runtime(&self) -> bool {
        self.is_immediate() || self.is_boxed() || self.is_literal() || self.is_non_empty_list()
    }

    /// Returns the size in bytes of the term in memory
    fn sizeof(&self) -> usize {
        match self.decode() {
            Ok(tt) => tt.sizeof(),
            Err(_) => mem::size_of::<Term>(),
        }
    }

    /// Returns the arity of this term, which reflects the number of words of data
    /// follow this term in memory.
    ///
    /// Returns zero for immediates/pointers
    fn arity(&self) -> usize {
        let size = self.sizeof();
        erts::to_word_size(size - mem::size_of::<Term>())
    }
}

impl CloneToProcess for Term {
    fn clone_to_process(&self, process: &crate::erts::process::Process) -> Term {
        if self.is_immediate() || self.is_literal() {
            *self
        } else if self.is_boxed() || self.is_non_empty_list() {
            let tt = self.decode().unwrap();
            tt.clone_to_process(process)
        } else {
            panic!("clone_to_process called on invalid term type: {:?}", self);
        }
    }

    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + HeapAlloc,
    {
        debug_assert!(self.is_runtime());
        if self.is_immediate() || self.is_literal() {
            Ok(*self)
        } else if self.is_boxed() || self.is_non_empty_list() {
            let tt = self.decode().unwrap();
            tt.clone_to_heap(heap)
        } else {
            panic!("clone_to_heap called on invalid term type: {:?}", self);
        }
    }

    fn size_in_words(&self) -> usize {
        self.sizeof()
    }
}
