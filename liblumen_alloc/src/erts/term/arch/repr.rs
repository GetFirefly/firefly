use core::fmt::{self, Debug, Display};
use core::hash::Hash;

use crate::erts::exception;
use crate::erts::term::prelude::*;

use super::Tag;

pub trait Repr:
    Sized + Copy + Debug + Display + PartialEq<Self> + Eq + PartialOrd<Self> + Ord + Hash + Send
{
    type Word: Clone + Copy + PartialEq + Eq + Debug + fmt::Binary;

    fn as_usize(&self) -> usize;

    fn word_to_usize(word: Self::Word) -> usize;

    fn value(&self) -> Self::Word;

    fn type_of(&self) -> Tag<Self::Word>;

    fn encode_immediate(value: Self::Word, tag: Self::Word) -> Self;
    fn encode_header(value: Self::Word, tag: Self::Word) -> Self;
    fn encode_list(value: *const Cons) -> Self;
    fn encode_box<U>(value: *const U) -> Self
    where
        U: ?Sized;
    fn encode_literal<U>(value: *const U) -> Self
    where
        U: ?Sized;

    unsafe fn decode_box(self) -> *mut Self;
    unsafe fn decode_list(self) -> Boxed<Cons>;
    unsafe fn decode_smallint(self) -> SmallInteger;
    unsafe fn decode_immediate(self) -> Self::Word;
    unsafe fn decode_atom(self) -> Atom;
    unsafe fn decode_pid(self) -> Pid;
    unsafe fn decode_port(self) -> Port;

    /// Access the raw header value contained in this term
    ///
    /// # Safety
    ///
    /// This function is unsafe because it makes the assumption that
    /// type checking has already been performed and the implementation may
    /// assume that it is operating on a header word. Undefined behaviour
    /// will result if this is invoked on any other term.
    unsafe fn decode_header_value(&self) -> Self::Word;

    fn decode_header(
        &self,
        tag: Tag<Self::Word>,
        literal: Option<bool>,
    ) -> exception::Result<TypedTerm>
    where
        Self: Encoded,
    {
        let ptr =
            Boxed::new(self as *const _ as *mut u64).ok_or_else(|| TermDecodingError::NoneValue)?;
        match tag {
            // Tuple cannot be constructed directly, as it is a dynamically-sized type,
            // instead we construct a fat pointer which requires the length of the tuple;
            // to get that we have to access the arity stored in the tuple header
            //
            // NOTE: This happens with a few other types as well, so if you see this pattern,
            // the reasoning is the same for each case
            Tag::Tuple => {
                let tuple = unsafe { Tuple::from_raw_term(ptr.cast::<Self>().as_ptr()) };
                Ok(TypedTerm::Tuple(tuple))
            }
            Tag::Closure => {
                let closure = unsafe { Closure::from_raw_term(ptr.cast::<Self>().as_ptr()) };
                Ok(TypedTerm::Closure(closure))
            }
            Tag::HeapBinary => {
                let bin = unsafe { HeapBin::from_raw_term(ptr.cast::<Self>().as_ptr()) };
                Ok(TypedTerm::HeapBinary(bin))
            }
            #[cfg(not(target_arch = "x86_64"))]
            Tag::Float => Ok(TypedTerm::Float(ptr.cast::<Float>())),
            Tag::BigInteger => Ok(TypedTerm::BigInteger(ptr.cast::<BigInteger>())),
            Tag::Reference => Ok(TypedTerm::Reference(ptr.cast::<Reference>())),
            Tag::ResourceReference => Ok(TypedTerm::ResourceReference(ptr.cast::<Resource>())),
            Tag::ProcBin => match literal {
                Some(false) => Ok(TypedTerm::ProcBin(ptr.cast::<ProcBin>())),
                Some(true) => Ok(TypedTerm::BinaryLiteral(ptr.cast::<BinaryLiteral>())),
                None => {
                    let offset = BinaryLiteral::flags_offset();
                    debug_assert_eq!(offset, ProcBin::inner_offset());
                    let flags_ptr = unsafe {
                        (self as *const _ as *const u8).offset(offset as isize)
                            as *const BinaryFlags
                    };
                    let flags = unsafe { *flags_ptr };
                    if flags.is_literal() {
                        Ok(TypedTerm::BinaryLiteral(ptr.cast::<BinaryLiteral>()))
                    } else {
                        Ok(TypedTerm::ProcBin(ptr.cast::<ProcBin>()))
                    }
                }
            },
            Tag::SubBinary => Ok(TypedTerm::SubBinary(ptr.cast::<SubBinary>())),
            Tag::MatchContext => Ok(TypedTerm::MatchContext(ptr.cast::<MatchContext>())),
            Tag::ExternalPid => Ok(TypedTerm::ExternalPid(ptr.cast::<ExternalPid>())),
            Tag::ExternalPort => Ok(TypedTerm::ExternalPort(ptr.cast::<ExternalPort>())),
            Tag::ExternalReference => Ok(TypedTerm::ExternalReference(
                ptr.cast::<ExternalReference>(),
            )),
            Tag::Map => Ok(TypedTerm::Map(ptr.cast::<Map>())),
            Tag::None => Err(TermDecodingError::NoneValue.into()),
            _ => Err(TermDecodingError::InvalidTag.into()),
        }
    }

    /// Decodes this raw term as a header, any non-header values will result in a panic
    ///
    /// NOTE: This is assumed to be used during decoding when this term has already been
    /// typechecked as a header type.
    #[inline]
    unsafe fn decode_header_unchecked(
        &self,
        tag: Tag<Self::Word>,
        literal: Option<bool>,
    ) -> TypedTerm
    where
        Self: Encoded,
    {
        match self.decode_header(tag.clone(), literal) {
            Ok(term) => term,
            Err(_) => panic!("invalid type tag: {:?}", tag),
        }
    }
}
