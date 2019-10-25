use core::fmt::{self, Debug, Display};
use core::hash::Hash;

use liblumen_core::offset_of;

use crate::erts::term::prelude::*;

use super::Tag;

pub trait Repr: Sized + Debug + Display + PartialEq<Self> + Eq + PartialOrd<Self> + Ord + Hash + Send {
    type Word: Debug + fmt::Binary;

    fn as_usize(self) -> usize;

    fn type_of(self) -> Tag<Self::Word>;

    fn encode_immediate(value: Self::Word, tag: Self::Word) -> Self;
    fn encode_header(value: Self::Word, tag: Self::Word) -> Self;
    fn encode_list(value: *const Cons) -> Self;
    fn encode_box<U>(value: *const U) -> Self where U: ?Sized;
    fn encode_literal<U>(value: *const U) -> Self where U: ?Sized;

    unsafe fn decode_list(self) -> Boxed<Cons>;
    unsafe fn decode_smallint(self) -> SmallInteger;
    unsafe fn decode_immediate(self) -> Self::Word;
    unsafe fn decode_atom(self) -> Atom;
    unsafe fn decode_pid(self) -> Pid;
    unsafe fn decode_port(self) -> Port;

    /// Used to access the value stored in a header word.
    ///
    /// NOTE: This is unsafe to use on any term except one where the tag is a valid header type
    unsafe fn decode_header_value(&self) -> Self::Word;

    fn decode_header(&self, tag: Tag<Self::Word>, literal: Option<bool>) -> Result<TypedTerm, TermDecodingError> {
        let ptr = Boxed::new(self as *const _ as *mut u64).ok_or(TermDecodingError::NoneValue)?;
        match tag {
            // Tuple cannot be constructed directly, as it is a dynamically-sized type,
            // instead we construct a fat pointer which requires the length of the tuple;
            // to get that we have to access the arity stored in the tuple header
            //
            // NOTE: This happens with a few other types as well, so if you see this pattern,
            // the reasoning is the same for each case
            Tag::Tuple => {
                let header = ptr.cast::<Header<Tuple>>().as_ref();
                let arity = header.arity();
                Ok(TypedTerm::Tuple(Tuple::from_raw_parts(ptr.as_ptr() as *mut u8, arity)))
            }
            Tag::Closure => {
                let header = ptr.cast::<Header<Closure>>().as_ref();
                let arity = header.arity();
                Ok(TypedTerm::Closure(Closure::from_raw_parts(ptr.as_ptr() as *mut u8, arity)))
            }
            Tag::HeapBinary => {
                let header = ptr.cast::<Header<HeapBin>>().as_ref();
                let arity = header.arity();
                Ok(TypedTerm::HeapBinary(HeapBin::from_raw_parts(ptr.as_ptr() as *mut u8, arity)))
            }
            #[cfg(not(target_arch = "x86_64"))]
            Tag::Float => Ok(TypedTerm::Float(ptr.cast::<Float>())),
            Tag::BigInteger => Ok(TypedTerm::BigInteger(ptr.cast::<BigInteger>())),
            Tag::Reference => Ok(TypedTerm::Reference(ptr.cast::<Reference>())),
            Tag::ResourceReference => Ok(TypedTerm::ResourceReference(ptr.cast::<ResourceReference>())),
            Tag::ProcBin => {
                match literal {
                    Some(false) => Ok(TypedTerm::ProcBin(ptr.cast::<ProcBin>())),
                    Some(true) => Ok(TypedTerm::BinaryLiteral(ptr.cast::<BinaryLiteral>())),
                    None => {
                        let offset = offset_of!(BinaryLiteral, flags);
                        debug_assert_eq!(offset, offset_of!(ProcBin, inner));
                        let flags_ptr = (self as *const _ as *const u8).offset(offset as isize) as *const BinaryFlags;
                        let flags = *flags_ptr;
                        if flags.is_literal() {
                            Ok(TypedTerm::BinaryLiteral(ptr.cast::<BinaryLiteral>()))
                        } else {
                            Ok(TypedTerm::ProcBin(ptr.cast::<ProcBin>()))
                        }
                    }
                }
            }
            Tag::SubBinary => Ok(TypedTerm::SubBinary(ptr.cast::<SubBinary>())),
            Tag::MatchContext => Ok(TypedTerm::MatchContext(ptr.cast::<MatchContext>())),
            Tag::ExternalPid => Ok(TypedTerm::ExternalPid(ptr.cast::<ExternalPid>())),
            Tag::ExternalPort => Ok(TypedTerm::ExternalPort(ptr.cast::<ExternalPort>())),
            Tag::ExternalReference => Ok(TypedTerm::ExternalReference(ptr.cast::<ExternalReference>())),
            Tag::Map => Ok(TypedTerm::Map(ptr.cast::<Map>())),
            Tag::None => Err(TermDecodingError::NoneValue),
            _ => Err(TermDecodingError::InvalidTag)
        }
    }

    /// Decodes this raw term as a header, any non-header values will result in a panic
    ///
    /// NOTE: This is assumed to be used during decoding when this term has already been
    /// typechecked as a header type.
    #[inline]
    unsafe fn decode_header_unchecked(&self, tag: Tag<Self::Word>, literal: Option<bool>) -> TypedTerm {
        match self.decode_header(tag, literal) {
            Ok(term) => term,
            Err(_) => panic!("invalid type tag: {:?}", tag)
        }
    }
}
