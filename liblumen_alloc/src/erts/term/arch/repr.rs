use core::fmt::{Debug, Display};
use core::hash::Hash;

use liblumen_term::Tag;

use alloc::sync::Arc;

use crate::erts::term::prelude::*;
use std::backtrace::Backtrace;

pub trait Repr:
    Sized + Copy + Debug + Display + PartialEq<Self> + Eq + PartialOrd<Self> + Ord + Hash + Send
{
    type Encoding: liblumen_term::Encoding;

    fn as_usize(&self) -> usize;

    fn value(&self) -> <Self::Encoding as liblumen_term::Encoding>::Type;

    fn decode_header(
        &self,
        tag: Tag<<Self::Encoding as liblumen_term::Encoding>::Type>,
        literal: Option<bool>,
    ) -> Result<TypedTerm, TermDecodingError>
    where
        Self: Encoded,
    {
        let ptr = Boxed::new(self as *const _ as *mut u64).ok_or_else(|| {
            TermDecodingError::NoneValue {
                backtrace: Arc::new(Backtrace::capture()),
            }
        })?;
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
            Tag::None => Err(TermDecodingError::NoneValue {
                backtrace: Arc::new(Backtrace::capture()),
            }),
            _ => Err(TermDecodingError::InvalidTag {
                backtrace: Arc::new(Backtrace::capture()),
            }),
        }
    }

    /// Decodes this raw term as a header, any non-header values will result in a panic
    ///
    /// NOTE: This is assumed to be used during decoding when this term has already been
    /// typechecked as a header type.
    #[inline]
    unsafe fn decode_header_unchecked(
        &self,
        tag: Tag<<Self::Encoding as liblumen_term::Encoding>::Type>,
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
