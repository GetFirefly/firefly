use core::alloc::AllocErr;
use core::cmp;
use core::convert::TryInto;
use core::mem;

use alloc::string::String;

use num_bigint::{BigInt, Sign};

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::ProcessControlBlock;

use super::*;

/// Concrete `Term` types, i.e. resolved to concrete values, or pointers to values.
///
/// Most runtime code should use this, or the types referenced here,
/// rather than working with raw `Term`s.
///
/// In some cases, these types contain pointers to `Term`, this is primarily the
/// container types, but to properly work on these containers, you must resolve the
/// inner types as well. In these situations, the pointer is _not_ the tagged value,
/// instead, you must dereference the pointer as `Term` and ask it to resolve itself
/// to its typed form.
#[derive(Debug)]
pub enum TypedTerm {
    List(Boxed<Cons>),
    Tuple(Boxed<Tuple>),
    Map(Boxed<Map>),
    Boxed(Boxed<Term>),
    Literal(Term),
    Pid(Pid),
    Port(Port),
    Reference(Boxed<Reference>),
    ExternalPid(Boxed<ExternalPid>),
    ExternalPort(Boxed<ExternalPort>),
    ExternalReference(Boxed<ExternalReference>),
    SmallInteger(SmallInteger),
    BigInteger(Boxed<BigInteger>),
    Float(Float),
    Atom(Atom),
    ProcBin(ProcBin),
    HeapBinary(HeapBin),
    SubBinary(SubBinary),
    MatchContext(MatchContext),
    Closure(Boxed<Closure>),
    Catch,
    Nil,
    None,
}
impl TypedTerm {
    #[inline]
    pub fn is_none(&self) -> bool {
        self.eq(&Self::None)
    }

    #[inline]
    pub fn is_nil(&self) -> bool {
        self.eq(&Self::Nil)
    }

    #[inline]
    pub fn is_catch(&self) -> bool {
        self.eq(&Self::Catch)
    }

    #[inline]
    pub fn is_number(&self) -> bool {
        match self {
            &Self::Float(_) => true,
            &Self::SmallInteger(_) => true,
            &Self::BigInteger(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_binary(&self) -> bool {
        match self {
            &Self::ProcBin(_) => true,
            &Self::HeapBinary(_) => true,
            &Self::SubBinary(_) => true,
            &Self::MatchContext(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_function(&self) -> bool {
        if let &Self::Closure(_) = self {
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn is_pid(&self) -> bool {
        match self {
            &Self::Pid(_) => true,
            &Self::ExternalPid(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_reference(&self) -> bool {
        match self {
            &Self::Reference(_) => true,
            &Self::ExternalReference(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_list(&self) -> bool {
        if let &Self::List(_) = self {
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn is_tuple(&self) -> bool {
        if let &Self::Tuple(_) = self {
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn is_map(&self) -> bool {
        if let &Self::Map(_) = self {
            true
        } else {
            false
        }
    }
}

macro_rules! partial_eq_impl_boxed {
    ($input:expr => $($variant:path),*) => {
        $(
            if let (&$variant(ref lhs), &$variant(ref rhs)) = $input {
                return lhs.eq(rhs);
            }
        )*

        return false;
    }
}

impl PartialEq<TypedTerm> for TypedTerm {
    fn eq(&self, other: &Self) -> bool {
        if mem::discriminant(self) != mem::discriminant(other) {
            return false;
        }
        match (self, other) {
            (&Self::Catch, &Self::Catch) => true,
            (&Self::Nil, &Self::Nil) => true,
            (&Self::None, &Self::None) => true,
            (Self::Boxed(self_boxed), Self::Boxed(other_boxed)) => self_boxed
                .to_typed_term()
                .unwrap()
                .eq(&other_boxed.to_typed_term().unwrap()),
            boxed => {
                partial_eq_impl_boxed! { boxed =>
                    Self::List,
                    Self::Tuple,
                    Self::Map,
                    Self::Pid,
                    Self::Port,
                    Self::Reference,
                    Self::ExternalPid,
                    Self::ExternalPort,
                    Self::ExternalReference,
                    Self::BigInteger,
                    Self::ProcBin,
                    Self::HeapBinary,
                    Self::SubBinary,
                    Self::MatchContext,
                    Self::Closure,
                    Self::Boxed,
                    Self::Literal,
                    Self::SmallInteger,
                    Self::Float,
                    Self::Atom
                }
            }
        }
    }
}
impl Eq for TypedTerm {}

macro_rules! partial_cmp_impl {
    (@try_with_equiv $input:expr => $variant:path as [$($equiv:path),*] , $($rest:tt)*) => {
        match $input {
            (&$variant(ref lhs), &$variant(ref rhs)) => { return lhs.partial_cmp(rhs); }
            $(
                (&$variant(ref lhs), &$equiv(ref rhs)) => { return lhs.partial_cmp(rhs); }
                (&$equiv(ref lhs), &$variant(ref rhs)) => { return lhs.partial_cmp(rhs); }
                (&$equiv(ref lhs), &$equiv(ref rhs)) => { return lhs.partial_cmp(rhs); }
            )*
            (&$variant(_), _) => { return Some(Ordering::Greater); }
            $(
                (&$equiv(_), _) => { return Some(Ordering::Greater); }
            )*
            _ => ()
        }

        partial_cmp_impl!(@try_is_constant $input => $($rest)*);
    };
    (@try_with_equiv $input:expr => $($rest:tt)*) => {
        partial_cmp_impl!(@try_without_equiv $input => $($rest)*);
    };
    (@try_with_equiv $input:expr => ) => {
        (());
    };
    (@try_without_equiv $input:expr => $variant:path , $($rest:tt)*) => {
        match $input {
            (&$variant(ref lhs), &$variant(ref rhs)) => { return lhs.partial_cmp(rhs); }
            (&$variant(_), _) => { return Some(Ordering::Greater); }
            _ => ()
        }

        partial_cmp_impl!(@try_is_constant $input => $($rest)*);
    };
    (@try_without_equiv $input:expr => ) => {
        (());
    };
    (@try_is_constant $input:expr => $variant:path where constant , $($rest:tt)*) => {
        match $input {
            (&$variant, &$variant) => return Some(Ordering::Equal),
            (&$variant, _) => return Some(Ordering::Greater),
            _ => ()
        }

        partial_cmp_impl!(@try_is_constant $input => $($rest)*);
    };
    (@try_is_constant $input:expr => $($rest:tt)*) => {
        partial_cmp_impl!(@try_is_invalid $input => $($rest)*);
    };
    (@try_is_invalid $input:expr => $variant:path where invalid , $($rest:tt)*) => {
        if let (&$variant, _) = $input {
            return None;
        }
        if let (_, &$variant) = $input {
            return None;
        }

        partial_cmp_impl!(@try_is_constant $input => $($rest)*);
    };
    (@try_is_invalid $input:expr => $($rest:tt)*) => {
        partial_cmp_impl!(@try_with_equiv $input => $($rest)*);
    };
    (($lhs:expr, $rhs:expr) => $($rest:tt)*) => {
        let input = ($lhs, $rhs);
        partial_cmp_impl!(@try_is_constant input => $($rest)*);

        // Fallback
        // Flip the arguments, then invert the result to avoid duplicating the above
        $rhs.partial_cmp($lhs).map(|option| option.reverse())
    };
}

impl PartialOrd<TypedTerm> for TypedTerm {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        use core::cmp::Ordering;
        if let Self::Boxed(boxed) = self {
            return boxed.to_typed_term().unwrap().partial_cmp(other);
        };
        if let Self::Boxed(boxed) = other {
            return boxed
                .to_typed_term()
                .unwrap()
                .partial_cmp(self)
                .map(|option| option.reverse());
        };
        // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
        partial_cmp_impl! { (self, other) =>
            Self::Catch where invalid,
            Self::None where invalid,
            Self::ProcBin as [Self::HeapBinary, Self::SubBinary, Self::MatchContext],
            Self::List,
            Self::Nil where constant,
            Self::Map,
            Self::Tuple,
            Self::ExternalPid,
            Self::Pid,
            Self::ExternalPort,
            Self::Port,
            Self::Closure,
            Self::ExternalReference,
            Self::Reference,
            Self::Atom,
            Self::BigInteger,
            Self::Float as [Self::SmallInteger],
        }
    }
}

unsafe impl AsTerm for TypedTerm {
    unsafe fn as_term(&self) -> Term {
        match self {
            &Self::List(ref inner) => inner.as_term(),
            &Self::Tuple(ref inner) => inner.as_term(),
            &Self::Map(ref inner) => inner.as_term(),
            &Self::Boxed(ref inner) => Term::make_boxed(inner),
            &Self::Literal(ref inner) => Term::make_boxed_literal(inner),
            &Self::Pid(ref inner) => inner.as_term(),
            &Self::Port(ref inner) => inner.as_term(),
            &Self::Reference(ref inner) => inner.as_term(),
            &Self::ExternalPid(ref inner) => inner.as_term(),
            &Self::ExternalPort(ref inner) => inner.as_term(),
            &Self::ExternalReference(ref inner) => inner.as_term(),
            &Self::SmallInteger(ref inner) => inner.as_term(),
            &Self::BigInteger(ref inner) => inner.as_term(),
            &Self::Float(ref inner) => inner.as_term(),
            &Self::Atom(ref inner) => inner.as_term(),
            &Self::ProcBin(ref inner) => inner.as_term(),
            &Self::HeapBinary(ref inner) => inner.as_term(),
            &Self::SubBinary(ref inner) => inner.as_term(),
            &Self::MatchContext(ref inner) => inner.as_term(),
            &Self::Closure(ref inner) => inner.as_term(),
            &Self::Catch => Term::CATCH,
            &Self::Nil => Term::NIL,
            &Self::None => Term::NONE,
        }
    }
}

impl CloneToProcess for TypedTerm {
    fn clone_to_process(&self, process: &ProcessControlBlock) -> Term {
        // Immediates are just copied and returned, all other terms
        // are expected to require allocation, so we delegate to those types
        match self {
            &Self::List(ref inner) => inner.clone_to_process(process),
            &Self::Tuple(ref inner) => inner.clone_to_process(process),
            &Self::Map(ref inner) => inner.clone_to_process(process),
            &Self::Boxed(ref inner) => inner.clone_to_process(process),
            &Self::Literal(inner) => inner,
            &Self::Pid(inner) => unsafe { inner.as_term() },
            &Self::Port(inner) => unsafe { inner.as_term() },
            &Self::Reference(ref inner) => inner.clone_to_process(process),
            &Self::ExternalPid(ref inner) => inner.clone_to_process(process),
            &Self::ExternalPort(ref inner) => inner.clone_to_process(process),
            &Self::ExternalReference(ref inner) => inner.clone_to_process(process),
            &Self::SmallInteger(inner) => unsafe { inner.as_term() },
            &Self::BigInteger(ref inner) => inner.clone_to_process(process),
            &Self::Float(inner) => inner.clone_to_process(process),
            &Self::Atom(inner) => unsafe { inner.as_term() },
            &Self::ProcBin(ref inner) => inner.clone_to_process(process),
            &Self::HeapBinary(ref inner) => inner.clone_to_process(process),
            &Self::SubBinary(ref inner) => inner.clone_to_process(process),
            &Self::MatchContext(ref inner) => inner.clone_to_process(process),
            &Self::Closure(ref inner) => inner.clone_to_process(process),
            &Self::Catch => Term::CATCH,
            &Self::Nil => Term::NIL,
            &Self::None => Term::NONE,
        }
    }

    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
        // Immediates are just copied and returned, all other terms
        // are expected to require allocation, so we delegate to those types
        match self {
            &Self::List(ref inner) => inner.clone_to_heap(heap),
            &Self::Tuple(ref inner) => inner.clone_to_heap(heap),
            &Self::Map(ref inner) => inner.clone_to_heap(heap),
            &Self::Boxed(ref inner) => inner.to_typed_term().unwrap().clone_to_heap(heap),
            &Self::Literal(inner) => Ok(inner),
            &Self::Pid(inner) => Ok(unsafe { inner.as_term() }),
            &Self::Port(inner) => Ok(unsafe { inner.as_term() }),
            &Self::Reference(ref inner) => inner.clone_to_heap(heap),
            &Self::ExternalPid(ref inner) => inner.clone_to_heap(heap),
            &Self::ExternalPort(ref inner) => inner.clone_to_heap(heap),
            &Self::ExternalReference(ref inner) => inner.clone_to_heap(heap),
            &Self::SmallInteger(inner) => Ok(unsafe { inner.as_term() }),
            &Self::BigInteger(ref inner) => inner.clone_to_heap(heap),
            &Self::Float(inner) => inner.clone_to_heap(heap),
            &Self::Atom(inner) => Ok(unsafe { inner.as_term() }),
            &Self::ProcBin(ref inner) => inner.clone_to_heap(heap),
            &Self::HeapBinary(ref inner) => inner.clone_to_heap(heap),
            &Self::SubBinary(ref inner) => inner.clone_to_heap(heap),
            &Self::MatchContext(ref inner) => inner.clone_to_heap(heap),
            &Self::Closure(ref inner) => inner.clone_to_heap(heap),
            &Self::Catch => Ok(Term::CATCH),
            &Self::Nil => Ok(Term::NIL),
            &Self::None => Ok(Term::NONE),
        }
    }

    fn size_in_words(&self) -> usize {
        match self {
            &Self::List(ref inner) => inner.size_in_words(),
            &Self::Tuple(ref inner) => inner.size_in_words(),
            &Self::Map(ref inner) => inner.size_in_words(),
            &Self::Boxed(ref inner) => inner.size_in_words(),
            &Self::Reference(ref inner) => inner.size_in_words(),
            &Self::ExternalPid(ref inner) => inner.size_in_words(),
            &Self::ExternalPort(ref inner) => inner.size_in_words(),
            &Self::ExternalReference(ref inner) => inner.size_in_words(),
            &Self::BigInteger(ref inner) => inner.size_in_words(),
            &Self::Float(inner) => inner.size_in_words(),
            &Self::ProcBin(ref inner) => inner.size_in_words(),
            &Self::HeapBinary(ref inner) => inner.size_in_words(),
            &Self::SubBinary(ref inner) => inner.size_in_words(),
            &Self::MatchContext(ref inner) => inner.size_in_words(),
            &Self::Closure(ref inner) => inner.size_in_words(),
            _ => 1,
        }
    }
}

impl TryInto<bool> for TypedTerm {
    type Error = BoolError;

    fn try_into(self) -> Result<bool, Self::Error> {
        match self {
            TypedTerm::Atom(atom) => match atom.name() {
                "false" => Ok(false),
                "true" => Ok(true),
                _ => Err(BoolError::NotABooleanName),
            },
            _ => Err(BoolError::Type),
        }
    }
}

impl TryInto<f64> for TypedTerm {
    type Error = TypeError;

    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            TypedTerm::SmallInteger(small_integer) => Ok(small_integer.into()),
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(big_integer) => Ok(big_integer.into()),
                TypedTerm::Float(float) => Ok(float.into()),
                _ => Err(TypeError),
            },
            _ => Err(TypeError),
        }
    }
}

impl TryInto<isize> for TypedTerm {
    type Error = TypeError;

    fn try_into(self) -> Result<isize, Self::Error> {
        match self {
            TypedTerm::SmallInteger(small_integer) => Ok(small_integer.into()),
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();

                match big_int.to_bytes_be() {
                    (Sign::NoSign, _) => Ok(0),
                    (sign, bytes) => {
                        let integer_usize = bytes
                            .iter()
                            .fold(0_usize, |acc, byte| (acc << 8) | (*byte as usize));

                        let integer_isize = if sign == Sign::Minus {
                            -(integer_usize as isize)
                        } else {
                            assert_eq!(sign, Sign::Plus);

                            integer_usize as isize
                        };

                        Ok(integer_isize)
                    }
                }
            }
            TypedTerm::Boxed(boxed) => boxed.to_typed_term().unwrap().try_into(),
            _ => Err(TypeError),
        }
    }
}

impl TryInto<String> for TypedTerm {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        match self {
            TypedTerm::Boxed(boxed) => boxed.to_typed_term().unwrap().try_into(),
            TypedTerm::HeapBinary(heap_binary) => heap_binary.try_into(),
            TypedTerm::SubBinary(subbinary) => subbinary.try_into(),
            TypedTerm::ProcBin(process_binary) => process_binary.try_into(),
            TypedTerm::MatchContext(match_context) => match_context.try_into(),
            _ => Err(badarg!()),
        }
    }
}
