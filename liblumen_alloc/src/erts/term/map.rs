use core::convert::{TryFrom, TryInto};

use super::*;

/// Placeholder for map header
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MapHeader(usize);

impl MapHeader {
    pub fn get(&self, _key: Term) -> Option<Term> {
        unimplemented!()
    }

    pub fn is_key(&self, _key: Term) -> bool {
        unimplemented!()
    }

    pub fn len(&self) -> usize {
        unimplemented!()
    }
}

unsafe impl AsTerm for MapHeader {
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self)
    }
}
impl crate::borrow::CloneToProcess for MapHeader {
    fn clone_to_heap<A: HeapAlloc>(&self, _heap: &mut A) -> Result<Term, AllocErr> {
        unimplemented!()
    }
}

impl TryFrom<Term> for Boxed<MapHeader> {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Boxed<MapHeader>, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Boxed<MapHeader> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Boxed<MapHeader>, Self::Error> {
        match typed_term {
            TypedTerm::Map(map_header) => Ok(map_header),
            _ => Err(TypeError),
        }
    }
}
