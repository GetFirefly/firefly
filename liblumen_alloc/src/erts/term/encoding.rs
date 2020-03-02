use core::alloc::Layout;
use core::convert::TryInto;
use core::fmt::{self, Debug};
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

use alloc::sync::Arc;

use std::backtrace::Backtrace;

use hashbrown::HashMap;
use thiserror::Error;

use liblumen_term::Encoding as TermEncoding;

use crate::borrow::CloneToProcess;
use crate::erts::exception::{AllocResult, InternalResult};
use crate::erts::fragment::HeapFragment;
use crate::erts::process::alloc::TermAlloc;

use super::arch::{Repr, Word};
use super::prelude::*;

/// Represents the various conditions under which encoding can fail
#[derive(Error, Debug, Clone, PartialEq, Eq)]
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
#[derive(Error, Debug, Clone)]
pub enum TermDecodingError {
    /// Occurs primarily with header tags, as there are some combinations
    /// of tag bits that are unused. Primary tags are (currently) fully
    /// occupied, and so invalid tags are not representable. That may
    /// change if additional representations are added that have free
    /// primary tags
    #[error("invalid type tag")]
    InvalidTag { backtrace: Arc<Backtrace> },
    /// Decoding a Term that is a move marker is considered an error,
    /// code which needs to be aware of move markers should already be
    /// manually checking for this possibility
    #[error("tried to decode a move marker")]
    MoveMarker { backtrace: Arc<Backtrace> },
    /// Decoding a Term that is a none value is considered an error,
    /// as this value is primarily used to indicate an error; and in cases
    /// where it represents a real value (namely as a move marker), it is
    /// meaningless to decode it. In all other cases, it represents uninitialized
    /// memory
    #[error("tried to decode a none value")]
    NoneValue { backtrace: Arc<Backtrace> },
}

impl Eq for TermDecodingError {}

impl PartialEq for TermDecodingError {
    fn eq(&self, other: &Self) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
}

/// This trait provides the ability to encode a type to some output encoding.
///
/// More specifically with regard to terms, it encodes immediates directly,
/// or for boxed terms, encodes the header. For boxed terms it is then
/// necessary to use the `Boxable::as_box` trait function to obtain an
/// encoded pointer when needed.
pub trait Encode<T: Encoded> {
    fn encode(&self) -> InternalResult<T>;
}

/// This is a marker trait for terms which can be boxed
pub trait Boxable<T: Repr> {}

/// This is a marker trait for boxed terms which are stored as literals
pub trait Literal<T: Repr>: Boxable<T> {}

/// This trait provides functionality for obtaining a pointer to an
/// unsized type from a raw term. For example, the `Tuple` type consists
/// of an arbitrary number of `Term` elements, and as such it is a
/// dynamically sized type (i.e. `?Sized`). Raw pointers to dynamically
/// sized types cannot be constructed without a size, so the job of this
/// trait is to provide functions to determine a size automatically from
/// such terms and construct a pointer, given only a pointer to a term
/// header.
///
/// Implementors of this trait are dynamically-sized types (DSTs), and
/// as such, the rules around how you can use them are far more restrictive
/// than a typical statically sized struct. Notably, you can only ever
/// construct a reference to one, you can't create a raw pointer to one or
/// construct an instance of one directly.
///
/// Clearly this seems like a chicken and egg problem since it is mostly
/// meaningless to construct references to things you can't create in the
/// first place. So how do values to DSTs get constructed? There are a few
/// key principles:
///
/// First, consider slices; the only way to make sense of a slice as a concept
/// is to know the position where the slice starts, the size of its elements
/// and the length of the slice. Rust gives us tools to construct these given
/// a pointer with a sized element type, and the length of the slice. This is
/// part of the solution, but it doesn't answer the question of how we deal with
/// dynamically sized _structs_.
///
/// The second piece of the puzzle is structural equivalence. Consider our `Tuple`
/// type, it is a struct consisting of a header containing the arity, and then some
/// arbitrary number of elements. If we expressed it's type as `Tuple<[Term]>`, where
/// the `elements` field is given the type `[Term]`; then a value of `Tuple<[Term]>`
/// is structurally equivalent to a value of `[Tuple<Term>; 1]`. Put another way, since
/// our variable-length field occurs at the end of the struct, we're really saying
/// that the layout of `[Term; 1]` is the same as `Term`, which is intuitively obvious.
///
/// The final piece of the puzzle is given by another Rust feature: unsizing coercions.
/// When Rust sees a cast from a sized type to an unsized type, it performs an unsizing
/// coercion.
///
/// For our purposes, the coercion here is from `[T; N]` to `CustomType<T>`, which
/// is allowed when `CustomType<T>` only has a single, non-PhantomData field involving `T`.
///
/// So given a pointer to a `Tuple`, if we construct a slice of `[Term; N]` and cast it to
/// a pointer to `Tuple`, Rust performs the coercion by first filling in the fields of `Tuple`
/// from the pointed-to memory, then coercing the `[Term; N]` to `[Term]` using the address of
/// the unsized field plus the size `N` to construct the fat pointer required for the `[Term]`
/// value.
///
/// To be clear: the pointer we use to construct the `[T; N]` slice that we coerce, is a
/// pointer to memory that contains the sized fields of `CustomType<T>` _followed by_ memory that
/// contains the actual `[T; N]` value. Rust is essentially casting the pointer given by
/// adding the offset of the unsized field to the base pointer we provided, plus the size
/// `N` to coerce the sized type to the type of the unsized field. The `N` provided is
/// _not_ the total size of the struct in units of `T`, it is always the number of elements
/// contained in the unsized field.
///
/// # Caveats
///
/// - This only works for types that follow Rusts' unsized coercion rules
/// - It is necessary to know the size of the variable-length region, which is generally true
/// for the types we are using this on, thanks to storing the arity in words of all non-immediate
/// types; but it has to be in terms of the element size. For example, `HeapBin` has a slice
/// of bytes, not `Term`, and the arity of the `HeapBin` is the size in words including extra
/// fields, so if we used that arity value, we'd get completely incorrect results. In the case of
/// `HeapBin`, we actually store the binary data size in the `flags` field, so we are able to use
/// that to obtain the `N` for our `[u8; N]` slice. Just be aware that similar steps will be
/// necessary for types that have non-word-sized elements.
///
/// - [DST Coercion RFC](https://github.com/rust-lang/rfcs/blob/master/text/0982-dst-coercion.md)
/// - [Unsize Trait](http://doc.rust-lang.org/1.38.0/std/marker/trait.Unsize.html)
/// - [Coercion - Nomicon](http://doc.rust-lang.org/1.38.0/nomicon/coercions.html)
pub trait UnsizedBoxable<T: Repr>: Boxable<T> + DynamicHeader {
    // The type of element contained in the dynamically sized
    // area of this type. By default this is specified as `()`,
    // with the assumption that elements are word-sized. For
    // non-word-sized elements, this is incorrect, e.g. for binary
    // data, as found in `HeapBin`
    // type Element: Sized;

    /// Given a pointer, this function dereferences the original term header,
    /// and uses its arity value to construct a fat pointer to the real term
    /// type.
    ///
    /// The implementation for this function is auto-implemented by default,
    /// but should be overridden if the number of elements in the dynamically
    /// sized portion of the type are not inferred by the arity produced from
    /// the header.
    unsafe fn from_raw_term(ptr: *mut T) -> Boxed<Self>;
}

/// Boxable terms require a header term to be built during
/// construction of the type, which specifies the type and
/// size in words of the non-header portion of the data
/// structure.
///
/// This struct is a safe abstraction over that header to
/// ensure that architecture-specific details do not leak
/// out of the `arch` or `encoding` modules.
#[repr(transparent)]
#[derive(PartialEq, Eq)]
pub struct Header<T: ?Sized> {
    value: Term,
    _phantom: PhantomData<T>,
}
impl<T: ?Sized> Clone for Header<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            _phantom: PhantomData,
        }
    }
}
impl<T: ?Sized> Copy for Header<T> {}
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
        mem::size_of::<Term>() * self.size_in_words()
    }

    /// Returns the size in words of the value this header represents, including the header
    #[inline]
    pub fn size_in_words(&self) -> usize {
        self.arity() + 1
    }

    /// Returns the size in words of the value this header represents, not including the header
    #[inline]
    pub fn arity(&self) -> usize {
        self.value.arity()
    }

    #[inline]
    fn to_word_size(size: usize) -> usize {
        use liblumen_core::alloc::utils::round_up_to_multiple_of;

        round_up_to_multiple_of(size, mem::size_of::<Term>()) / mem::size_of::<Term>()
    }
}
impl<T: ?Sized> Debug for Header<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = core::any::type_name::<T>();
        write!(f, "Header<{}>({:#b})", name, self.value.as_usize())
    }
}
const_assert_eq!(mem::size_of::<Header<usize>>(), mem::size_of::<usize>());
impl Header<Map> {
    pub fn from_map(map: &HashMap<Term, Term>) -> Self {
        // NOTE: This size only accounts for the HashMap header, not the values
        let layout = Layout::for_value(map);
        let map_size = layout.size();
        let arity = Self::static_arity() + Self::to_word_size(map_size);
        let value = Term::encode_header(arity.try_into().unwrap(), Term::HEADER_MAP);
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

/// This is a marker trait for dynamically-sized types which have headers
pub trait DynamicHeader {
    /// The header tag associated with this type
    const TAG: Word;

    /// Returns the header of this value
    fn header(&self) -> Header<Self>;
}
#[macro_export]
macro_rules! impl_dynamic_header {
    ($typ:ty, $tag:expr) => {
        impl crate::erts::term::encoding::DynamicHeader for $typ {
            const TAG: crate::erts::term::arch::Word = $tag;

            #[inline]
            fn header(&self) -> crate::erts::term::encoding::Header<Self> {
                self.header
            }
        }
    };
}
impl<T: ?Sized + DynamicHeader> Header<T> {
    pub fn from_arity(arity: usize) -> Self {
        let arity = arity.try_into().unwrap();
        let value = Term::encode_header(arity, T::TAG);
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

/// This is a marker trait for types which have headers that can be statically derived
pub trait StaticHeader {
    /// The header tag associated with this type
    const TAG: Word;

    /// Returns the header of this value
    fn header(&self) -> Header<Self>;
}
#[macro_export]
macro_rules! impl_static_header {
    ($typ:ty, $tag:expr) => {
        impl crate::erts::term::encoding::StaticHeader for $typ {
            const TAG: crate::erts::term::arch::Word = $tag;

            #[inline]
            fn header(&self) -> crate::erts::term::encoding::Header<Self> {
                self.header
            }
        }
    };
}
impl<T: StaticHeader> Default for Header<T> {
    fn default() -> Self {
        let arity = Self::to_word_size(Self::static_arity());
        let value = Term::encode_header(arity.try_into().unwrap(), T::TAG);
        Self {
            value,
            _phantom: PhantomData,
        }
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
/// then you may end up with a partial term which leads to out of bounds memory addresses, or
/// other undefined behavior.
pub trait Encoded: Repr + Copy {
    /// Decodes `Self` into a `TypedTerm`, unless the encoded value is
    /// invalid or malformed.
    ///
    /// NOTE: Implementations should attempt to catch all possible decoding errors
    /// to make this as safe as possible. The only exception to this rule should
    /// be the case of decoding a pointer which can not be validated unless it
    /// is dereferenced.
    fn decode(&self) -> Result<TypedTerm, TermDecodingError>;

    /// Returns `true` if the encoded value represents `NONE`
    #[inline]
    fn is_none(self) -> bool {
        Self::Encoding::is_none(self.value())
    }
    /// Returns `true` if the encoded value represents a pointer to a term
    #[inline]
    fn is_boxed(self) -> bool {
        Self::Encoding::is_boxed(self.value())
    }
    /// Returns `true` if the encoded value is the header of a non-immediate term
    #[inline]
    fn is_header(self) -> bool {
        Self::Encoding::is_header(self.value())
    }
    /// Returns `true` if the encoded value is an immediate value
    #[inline]
    fn is_immediate(self) -> bool {
        Self::Encoding::is_immediate(self.value())
    }
    /// Returns `true` if the encoded value represents a pointer to a literal value
    #[inline]
    fn is_literal(self) -> bool {
        Self::Encoding::is_literal(self.value())
    }

    /// Returns `true` if the encoded value represents the empty list
    #[inline]
    fn is_nil(self) -> bool {
        Self::Encoding::is_nil(self.value())
    }
    /// Returns `true` if the encoded value represents a nil or `Cons` value (empty or non-empty
    /// list)
    #[inline]
    fn is_list(self) -> bool {
        Self::Encoding::is_list(self.value())
    }
    /// Returns `true` if the encoded value represents a `Cons` value (non-empty list)
    #[inline]
    fn is_non_empty_list(self) -> bool {
        Self::Encoding::is_non_empty_list(self.value())
    }
    /// Returns `true` if the encoded value is an atom
    fn is_atom(self) -> bool {
        Self::Encoding::is_atom(self.value())
    }
    /// Returns `true` if the encoded value is a boolean
    fn is_boolean(self) -> bool {
        Self::Encoding::is_boolean(self.value())
    }
    /// Returns `true` if the encoded value is a fixed-width integer value
    fn is_smallint(self) -> bool {
        Self::Encoding::is_smallint(self.value())
    }
    /// Returns `true` if the encoded value is the header of a arbitrary-width integer value
    fn is_bigint(self) -> bool {
        Self::Encoding::is_bigint(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a arbitrary-width integer value
    fn is_boxed_bigint(self) -> bool {
        Self::Encoding::is_boxed_bigint(self.value())
    }
    /// Returns `true` if the encoded value is an integer of either
    /// fixed or arbitrary width
    fn is_integer(&self) -> bool {
        if self.is_smallint() {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::BigInteger(_)) => true,
            _ => false,
        }
    }
    /// Returns `true` if the encoded value is a float
    ///
    /// NOTE: This function returns true if either the term is an immediate float,
    /// or if it is the header of a packed float. It does not unwrap boxed values.
    fn is_float(self) -> bool {
        Self::Encoding::is_float(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `Float`
    fn is_boxed_float(self) -> bool {
        Self::Encoding::is_boxed_float(self.value())
    }
    /// Returns `true` if the encoded value is either a float or integer
    fn is_number(self) -> bool {
        match self.decode() {
            Ok(TypedTerm::SmallInteger(_))
            | Ok(TypedTerm::BigInteger(_))
            | Ok(TypedTerm::Float(_)) => true,
            _ => false,
        }
    }
    /// Returns `true` if the encoded value is the header of a `Tuple`
    fn is_tuple(self) -> bool {
        Self::Encoding::is_tuple(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `Tuple`
    fn is_boxed_tuple(self) -> bool {
        Self::Encoding::is_boxed_tuple(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `Map`
    fn is_map(self) -> bool {
        Self::Encoding::is_map(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `Map`
    fn is_boxed_map(self) -> bool {
        Self::Encoding::is_boxed_map(self.value())
    }
    /// Returns `true` if the encoded value is a `Pid`
    fn is_local_pid(self) -> bool {
        Self::Encoding::is_local_pid(self.value())
    }
    /// Returns `true` if the encoded value is the header of an `ExternalPid`
    fn is_remote_pid(self) -> bool {
        Self::Encoding::is_local_pid(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to an `ExternalPid`
    fn is_boxed_remote_pid(self) -> bool {
        Self::Encoding::is_boxed_remote_pid(self.value())
    }
    /// Returns `true` if the encoded value is a `Port`
    fn is_local_port(self) -> bool {
        Self::Encoding::is_local_port(self.value())
    }
    /// Returns `true` if the encoded value is the header of an `ExternalPort`
    fn is_remote_port(self) -> bool {
        Self::Encoding::is_remote_port(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to an `ExternalPort`
    fn is_boxed_remote_port(self) -> bool {
        Self::Encoding::is_boxed_remote_port(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `Reference`
    fn is_local_reference(self) -> bool {
        Self::Encoding::is_local_reference(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `Reference`
    fn is_boxed_local_reference(self) -> bool {
        Self::Encoding::is_boxed_local_reference(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `ExternalReference`
    fn is_remote_reference(self) -> bool {
        Self::Encoding::is_remote_reference(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `ExternalReference`
    fn is_boxed_remote_reference(self) -> bool {
        Self::Encoding::is_boxed_remote_reference(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `Resource`
    fn is_resource_reference(self) -> bool {
        Self::Encoding::is_resource_reference(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `Resource`
    fn is_boxed_resource_reference(self) -> bool {
        Self::Encoding::is_boxed_resource_reference(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `ProcBin`
    fn is_procbin(self) -> bool {
        Self::Encoding::is_procbin(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `ProcBin`
    fn is_boxed_procbin(self) -> bool {
        Self::Encoding::is_boxed_procbin(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `HeapBin`
    fn is_heapbin(self) -> bool {
        Self::Encoding::is_heapbin(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `HeapBin`
    fn is_boxed_heapbin(self) -> bool {
        Self::Encoding::is_boxed_heapbin(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `SubBinary`
    fn is_subbinary(self) -> bool {
        Self::Encoding::is_subbinary(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `SubBinary`
    fn is_boxed_subbinary(self) -> bool {
        Self::Encoding::is_boxed_subbinary(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `MatchContext`
    fn is_match_context(self) -> bool {
        Self::Encoding::is_match_context(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `MatchContext`
    fn is_boxed_match_context(self) -> bool {
        Self::Encoding::is_boxed_match_context(self.value())
    }
    /// Returns `true` if the encoded value is the header of a `Closure`
    fn is_function(self) -> bool {
        Self::Encoding::is_function(self.value())
    }
    /// Returns `true` if the encoded value is a pointer to a `Tuple`
    fn is_boxed_function(self) -> bool {
        Self::Encoding::is_boxed_function(self.value())
    }

    /// Returns `true` if this term is a bitstring type,
    /// where the number of bits is evenly divisible by 8 (i.e. one byte)
    fn is_binary(&self) -> bool {
        match self.decode().expect("invalid term") {
            TypedTerm::HeapBinary(_) | TypedTerm::ProcBin(_) | TypedTerm::BinaryLiteral(_) => true,
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
            _ => false,
        }
    }

    /// Returns true if this term is a port type
    fn is_port(&self) -> bool {
        if self.is_local_port() {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::ExternalPort(_)) => true,
            _ => false,
        }
    }

    /// Returns true if this term is a reference type
    fn is_reference(&self) -> bool {
        match self.decode() {
            Ok(TypedTerm::Reference(_))
            | Ok(TypedTerm::ExternalReference(_))
            | Ok(TypedTerm::ResourceReference(_)) => true,
            _ => false,
        }
    }

    /// Returns true if this is a term that is valid for use as an argument
    /// in the runtime, as a key in a datstructure, or other position in which
    /// an immediate or a reference is required or desirable
    fn is_valid(&self) -> bool {
        Self::Encoding::is_valid(self.value())
    }

    /// Returns the size in bytes of the term in memory
    fn sizeof(&self) -> usize {
        Self::Encoding::sizeof(self.value())
    }

    /// Returns the arity of this term, which reflects the number of words of data
    /// following this term in memory.
    ///
    /// Returns zero for immediates/pointers
    fn arity(&self) -> usize {
        Self::Encoding::arity(self.value())
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
        A: ?Sized + TermAlloc,
    {
        if self.is_immediate() || self.is_literal() {
            Ok(*self)
        } else if self.is_boxed() || self.is_non_empty_list() {
            // There is no need to clone the actual object to this
            // heap if it is already there, just clone a pointer
            let ptr: *mut Term = self.dyn_cast();
            if heap.contains(ptr) {
                // Just return self
                Ok(*self)
            } else {
                // We're good to clone
                let tt = self.decode().unwrap();
                tt.clone_to_heap(heap)
            }
        } else {
            panic!("clone_to_heap called on invalid term type: {:?}", self);
        }
    }

    fn clone_to_fragment(&self) -> AllocResult<(Term, NonNull<HeapFragment>)> {
        let tt = self.decode().unwrap();
        tt.clone_to_fragment()
    }

    fn size_in_words(&self) -> usize {
        let tt = self.decode().unwrap();
        tt.size_in_words()
    }
}
