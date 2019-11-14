mod arch;
mod encoding;
mod typed_term;
mod atom;
mod binary;
mod boxed;
mod closure;
mod float;
mod integer;
mod list;
mod map;
mod port;
mod resource;
mod tuple;
mod release;
pub(super) mod pid;
pub(super) mod reference;
pub mod index;
pub mod convert;

use core::fmt;

// This module provides a limited set of exported types/traits for convenience
pub mod prelude {
    pub use liblumen_core::cmp::ExactEq;

    // Export the platform term representation
    pub use super::arch::RawTerm as Term;
    pub use super::release::Release;
    // Export the encoding traits and types as they are used hand-in-hand with Term
    pub use super::encoding::{Encode, Encoded, Boxable, UnsizedBoxable, Literal, Cast};
    pub use super::encoding::{Header, DynamicHeader, StaticHeader};
    // Export the encoding errors
    pub use super::encoding::{TermEncodingError, TermDecodingError};
    // Export the boxed term wrapper
    pub use super::boxed::Boxed;
    // Export the typed term wrapper
    pub use super::typed_term::TypedTerm;
    // Export the primary term types
    pub use super::atom::{Atom, AtomError};
    pub use super::closure::Closure;
    pub use super::float::Float;
    pub use super::integer::{Integer, SmallInteger, BigInteger};
    pub use super::list::{List, ImproperList, MaybeImproper, Cons, ListBuilder, HeaplessListBuilder};
    pub use super::map::Map;
    pub use super::pid::{AnyPid, Pid, ExternalPid, InvalidPidError};
    pub use super::port::{Port, ExternalPort};
    pub use super::reference::{Reference, ExternalReference, ReferenceNumber};
    pub use super::tuple::Tuple;
    pub use super::resource::Resource;
    // Re-export the binary type prelude
    pub use super::binary::prelude::*;
    // Export tuple indexing
    pub use super::index::{TupleIndex, ZeroBasedIndex, OneBasedIndex, IndexError};
    // Export error types
    pub use super::convert::{TypeError, BoolError};
    pub use super::integer::TryIntoIntegerError;
    pub use super::BadArgument;

    pub(super) use crate::{impl_dynamic_header, impl_static_header};
}

/// This error is produced when a term is given to a runtime
/// function is invalid for that function
#[derive(Clone, Copy)]
pub struct BadArgument(self::prelude::Term);
impl BadArgument {
    #[inline]
    pub fn new(term: self::prelude::Term) -> Self {
        Self(term)
    }

    #[inline]
    pub fn argument(&self) -> self::prelude::Term {
        self.0
    }
}
impl fmt::Display for BadArgument {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "bad argument: {:?}", self.0)
    }
}
