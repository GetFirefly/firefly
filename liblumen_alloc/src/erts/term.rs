mod arch;
mod atom;
mod binary;
mod boxed;
pub mod closure;
pub mod convert;
mod encoding;
mod float;
pub mod index;
mod integer;
mod list;
mod map;
pub(super) mod pid;
mod port;
pub(super) mod reference;
mod release;
mod resource;
mod tuple;
mod typed_term;

use core::fmt;

// This module provides a limited set of exported types/traits for convenience
pub mod prelude {
    pub use liblumen_core::cmp::ExactEq;

    // Export the platform term representation
    pub use super::arch::RawTerm as Term;
    pub use super::release::Release;
    // Export the encoding traits and types as they are used hand-in-hand with Term
    pub use super::encoding::{Boxable, Cast, Encode, Encoded, Literal, UnsizedBoxable};
    pub use super::encoding::{DynamicHeader, Header, StaticHeader};
    // Export the encoding errors
    pub use super::encoding::{TermDecodingError, TermEncodingError};
    // Export the boxed term wrapper
    pub use super::boxed::Boxed;
    // Export the typed term wrapper
    pub use super::typed_term::TypedTerm;
    // Export the primary term types
    pub use super::atom::{Atom, AtomError, TryAtomFromTermError};
    pub use super::closure::Closure;
    pub use super::float::Float;
    pub use super::integer::{BigInteger, Integer, SmallInteger};
    pub use super::list::{
        Cons, HeaplessListBuilder, ImproperList, List, ListBuilder, MaybeImproper,
    };
    pub use super::map::Map;
    pub use super::pid::{AnyPid, ExternalPid, InvalidPidError, Pid};
    pub use super::port::{ExternalPort, Port};
    pub use super::reference::{ExternalReference, Reference, ReferenceNumber};
    pub use super::resource::Resource;
    pub use super::tuple::Tuple;
    // Re-export the binary type prelude
    pub use super::binary::prelude::*;
    // Export tuple indexing
    pub use super::index::{IndexError, OneBasedIndex, TupleIndex, ZeroBasedIndex};
    // Export error types
    pub use super::convert::{BoolError, TypeError};
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
