use core::alloc::Layout;
use core::cmp;
use core::convert::TryInto;
use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};
use core::mem;

use num_bigint::{BigInt, Sign};

use liblumen_core::cmp::ExactEq;

use crate::borrow::CloneToProcess;
use crate::erts::alloc::TermAlloc;
use crate::erts::exception::{AllocResult, Exception};
use crate::erts::Process;

use super::prelude::*;

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
pub enum TypedTerm {
    List(Boxed<Cons>),
    Tuple(Boxed<Tuple>),
    Map(Boxed<Map>),
    Pid(Pid),
    Port(Port),
    Reference(Boxed<Reference>),
    ExternalPid(Boxed<ExternalPid>),
    ExternalPort(Boxed<ExternalPort>),
    ExternalReference(Boxed<ExternalReference>),
    SmallInteger(SmallInteger),
    BigInteger(Boxed<BigInteger>),
    #[cfg(target_arch = "x86_64")]
    Float(Float),
    #[cfg(not(target_arch = "x86_64"))]
    Float(Boxed<Float>),
    Atom(Atom),
    ResourceReference(Boxed<Resource>),
    BinaryLiteral(Boxed<BinaryLiteral>),
    ProcBin(Boxed<ProcBin>),
    HeapBinary(Boxed<HeapBin>),
    SubBinary(Boxed<SubBinary>),
    MatchContext(Boxed<MatchContext>),
    Closure(Boxed<Closure>),
    Nil,
}
impl fmt::Debug for TypedTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Self::Nil => write!(f, "Nil"),
            &Self::Pid(pid) => write!(f, "{:?}", pid),
            &Self::Port(port) => write!(f, "{:?}", port),
            &Self::Atom(atom) => write!(f, "{:?}", atom),
            &Self::SmallInteger(small) => write!(f, "{:?}", small),
            #[cfg(target_arch = "x86_64")]
            &Self::Float(float) => write!(f, "{:?}", float),
            #[cfg(not(target_arch = "x86_64"))]
            &Self::Float(boxed) => write!(f, "{:?}", boxed),
            &Self::BigInteger(boxed) => write!(f, "{:?}", boxed),
            &Self::List(boxed) => write!(f, "{:?}", boxed),
            &Self::Tuple(boxed) => write!(f, "{:?}", boxed),
            &Self::Map(boxed) => write!(f, "{:?}", boxed),
            &Self::Reference(boxed) => write!(f, "{:?}", boxed),
            &Self::ExternalPid(boxed) => write!(f, "{:?}", boxed),
            &Self::ExternalPort(boxed) => write!(f, "{:?}", boxed),
            &Self::ExternalReference(boxed) => write!(f, "{:?}", boxed),
            &Self::ResourceReference(boxed) => write!(f, "{:?}", boxed),
            &Self::BinaryLiteral(boxed) => write!(f, "{:?}", boxed),
            &Self::ProcBin(boxed) => write!(f, "{:?}", boxed),
            &Self::HeapBinary(boxed) => write!(f, "{:?}", boxed),
            &Self::SubBinary(boxed) => write!(f, "{:?}", boxed),
            &Self::MatchContext(boxed) => write!(f, "{:?}", boxed),
            &Self::Closure(boxed) => write!(f, "{:?}", boxed),
        }
    }
}
impl TypedTerm {
    #[inline]
    pub fn is_nil(&self) -> bool {
        self.eq(&Self::Nil)
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
            &Self::BinaryLiteral(_) => true,
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

    pub fn is_function_with_arity(&self, arity: u8) -> bool {
        match self {
            Self::Closure(closure) => closure.as_ref().arity() == arity,
            _ => false,
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
    pub fn is_proper_list(&self) -> bool {
        if let &Self::List(cons) = self {
            cons.is_proper()
        } else {
            self.is_nil()
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

    #[inline]
    pub fn sizeof(&self) -> usize {
        use TypedTerm::*;

        match self {
            // These terms are either immediates or have statically known sizes,
            // so we can just use mem::size_of_val(term)
            Atom(term) => mem::size_of_val(term),
            Pid(term) => mem::size_of_val(term),
            Port(term) => mem::size_of_val(term),
            SmallInteger(term) => mem::size_of_val(term),
            Float(term) => mem::size_of_val(term.as_ref()),
            List(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            Map(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            Reference(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            ExternalPid(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            ExternalPort(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            ExternalReference(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            ResourceReference(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            MatchContext(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            BinaryLiteral(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            SubBinary(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            ProcBin(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            BigInteger(term_ptr) => mem::size_of_val(term_ptr.as_ref()),
            // These terms are dynamically-sized types, so we need to calculate their
            // layouts dynamically as well, we rely on Layout::for_value for this
            Tuple(term_ptr) => Layout::for_value(term_ptr.as_ref()).size(),
            Closure(term_ptr) => Layout::for_value(term_ptr.as_ref()).size(),
            HeapBinary(term_ptr) => Layout::for_value(term_ptr.as_ref()).size(),
            Nil => mem::size_of::<Term>(),
        }
    }
}

impl Display for TypedTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TypedTerm::*;

        match self {
            Atom(term) => write!(f, "{}", term),
            Pid(term) => write!(f, "{}", term),
            Port(term) => write!(f, "{}", term),
            SmallInteger(term) => write!(f, "{}", term),
            Float(term) => write!(f, "{}", term.as_ref()),
            List(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            Map(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            Reference(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            ExternalPid(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            ExternalPort(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            ExternalReference(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            ResourceReference(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            MatchContext(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            BinaryLiteral(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            SubBinary(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            ProcBin(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            BigInteger(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            Tuple(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            Closure(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            HeapBinary(term_ptr) => write!(f, "{}", term_ptr.as_ref()),
            Nil => write!(f, "[]"),
        }
    }
}

impl Hash for TypedTerm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use crate::erts::term::arch::Repr;
        use TypedTerm::*;

        match self {
            Atom(term) => term.hash(state),
            Pid(term) => term.hash(state),
            Port(term) => term.hash(state),
            SmallInteger(term) => term.hash(state),
            Float(term) => term.as_ref().hash(state),
            List(term_ptr) => term_ptr.as_ref().hash(state),
            Map(term_ptr) => term_ptr.as_ref().hash(state),
            Reference(term_ptr) => term_ptr.as_ref().hash(state),
            ExternalPid(term_ptr) => term_ptr.as_ref().hash(state),
            ExternalPort(term_ptr) => term_ptr.as_ref().hash(state),
            ExternalReference(term_ptr) => term_ptr.as_ref().hash(state),
            ResourceReference(term_ptr) => term_ptr.as_ref().hash(state),
            MatchContext(term_ptr) => term_ptr.as_ref().hash(state),
            BinaryLiteral(term_ptr) => term_ptr.as_ref().hash(state),
            SubBinary(term_ptr) => term_ptr.as_ref().hash(state),
            ProcBin(term_ptr) => term_ptr.as_ref().hash(state),
            BigInteger(term_ptr) => term_ptr.as_ref().hash(state),
            Tuple(term_ptr) => term_ptr.as_ref().hash(state),
            Closure(term_ptr) => term_ptr.as_ref().hash(state),
            HeapBinary(term_ptr) => term_ptr.as_ref().hash(state),
            Nil => Term::NIL.as_usize().hash(state),
        };
    }
}

impl ExactEq for TypedTerm {
    fn exact_eq(&self, other: &Self) -> bool {
        if self.is_number() && other.is_number() {
            if mem::discriminant(self) != mem::discriminant(other) {
                return false;
            }
        }

        self.eq(other)
    }
}

/// `PartialEq`'s `eq` MUST agree with `PartialOrd`'s `partial_cmp` and `Ord`'s `cmp`, so because
/// `partial_cmp` and `cmp` MUST convert between numeric types to allow for ordering of all types,
/// `PartialEq`'s `eq` must also do conversion.  To get a non-converting `eq`-like function use
/// `exactly_eq`.
impl PartialEq<TypedTerm> for TypedTerm {
    fn eq(&self, other: &Self) -> bool {
        match self {
            TypedTerm::SmallInteger(lhs) => match other {
                TypedTerm::SmallInteger(rhs) => lhs.eq(rhs),
                // Flip order so that only type that will be conversion target needs to
                // implement `PartialEq` between types.
                TypedTerm::Float(rhs) => rhs.eq(lhs),
                TypedTerm::BigInteger(rhs) => rhs.eq(lhs),
                _ => false,
            },
            TypedTerm::Float(lhs) => match other {
                TypedTerm::SmallInteger(rhs) => lhs.eq(rhs),
                TypedTerm::Float(rhs) => lhs.eq(rhs),
                TypedTerm::BigInteger(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::BigInteger(lhs) => match other {
                TypedTerm::SmallInteger(rhs) => lhs.eq(rhs),
                TypedTerm::Float(rhs) => lhs.eq(rhs),
                TypedTerm::BigInteger(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Reference(lhs) => match other {
                TypedTerm::Reference(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::ResourceReference(lhs) => match other {
                TypedTerm::ResourceReference(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Closure(lhs) => match other {
                TypedTerm::Closure(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::ExternalPid(lhs) => match other {
                TypedTerm::ExternalPid(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::ExternalPort(lhs) => match other {
                TypedTerm::ExternalPort(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::ExternalReference(lhs) => match other {
                TypedTerm::ExternalReference(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Tuple(lhs) => match other {
                TypedTerm::Tuple(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Map(lhs) => match other {
                TypedTerm::Map(rhs) => lhs.eq(rhs),
                _ => false,
            },
            // Bitstrings in likely order
            TypedTerm::HeapBinary(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.eq(rhs),
                TypedTerm::ProcBin(rhs) => rhs.eq(lhs),
                TypedTerm::BinaryLiteral(rhs) => rhs.eq(lhs),
                TypedTerm::SubBinary(rhs) => rhs.eq(lhs),
                TypedTerm::MatchContext(rhs) => rhs.eq(lhs),
                _ => false,
            },
            TypedTerm::ProcBin(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.eq(rhs),
                TypedTerm::ProcBin(rhs) => lhs.eq(rhs),
                TypedTerm::BinaryLiteral(rhs) => lhs.eq(rhs),
                TypedTerm::SubBinary(rhs) => rhs.eq(lhs),
                TypedTerm::MatchContext(rhs) => rhs.eq(lhs),
                _ => false,
            },
            TypedTerm::BinaryLiteral(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.eq(rhs),
                TypedTerm::ProcBin(rhs) => lhs.eq(rhs),
                TypedTerm::BinaryLiteral(rhs) => lhs.eq(rhs),
                TypedTerm::SubBinary(rhs) => rhs.eq(lhs),
                TypedTerm::MatchContext(rhs) => rhs.eq(lhs),
                _ => false,
            },
            TypedTerm::SubBinary(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.eq(rhs),
                TypedTerm::ProcBin(rhs) => lhs.eq(rhs),
                TypedTerm::BinaryLiteral(rhs) => lhs.eq(rhs),
                TypedTerm::SubBinary(rhs) => lhs.eq(rhs),
                TypedTerm::MatchContext(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::MatchContext(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.eq(rhs),
                TypedTerm::ProcBin(rhs) => lhs.eq(rhs),
                TypedTerm::BinaryLiteral(rhs) => lhs.eq(rhs),
                TypedTerm::SubBinary(rhs) => rhs.eq(lhs),
                TypedTerm::MatchContext(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Atom(lhs) => match other {
                TypedTerm::Atom(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Port(lhs) => match other {
                TypedTerm::Port(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Pid(lhs) => match other {
                TypedTerm::Pid(rhs) => lhs.eq(rhs),
                _ => false,
            },
            TypedTerm::Nil => match other {
                TypedTerm::Nil => true,
                _ => false,
            },
            TypedTerm::List(lhs) => match other {
                TypedTerm::List(rhs) => lhs.eq(rhs),
                _ => false,
            },
        }
    }
}
impl Eq for TypedTerm {}

/// All terms in Erlang and Elixir are completely ordered.
///
/// number < atom < reference < function < port < pid < tuple < map < list < bitstring
///
/// > When comparing two numbers of different types (a number being either an integer or a float), a
/// > conversion to the type with greater precision will always occur, unless the comparison
/// > operator used is either === or !==. A float will be considered more precise than an integer,
/// > unless the float is greater/less than +/-9007199254740992.0 respectively, at which point all
/// > the significant figures of the float are to the left of the decimal point. This behavior
/// > exists so that the comparison of large numbers remains transitive.
/// >
/// > The collection types are compared using the following rules:
/// >
/// > * Tuples are compared by size, then element by element.
/// > * Maps are compared by size, then by keys in ascending term order, then by values in key
/// >   order.   In the specific case of maps' key ordering, integers are always considered to be
/// >   less than floats.
/// > * Lists are compared element by element.
/// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
/// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
impl Ord for TypedTerm {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        use cmp::Ordering::*;

        match self {
            // Numbers
            // Order is SmallInteger, Float, BigInt because of the order of their ranges
            TypedTerm::SmallInteger(lhs) => match other {
                TypedTerm::SmallInteger(rhs) => lhs.cmp(rhs),
                // Flip order so that only type that will be conversion target needs to
                // implement `PartialOrd` between types.
                TypedTerm::Float(rhs) => (*lhs).partial_cmp(rhs).unwrap(),
                TypedTerm::BigInteger(rhs) => lhs.partial_cmp(rhs).unwrap(),
                _ => Less,
            },
            TypedTerm::Float(lhs) => match other {
                TypedTerm::SmallInteger(rhs) => lhs.partial_cmp(rhs).unwrap(),
                TypedTerm::Float(rhs) => lhs.partial_cmp(rhs).unwrap(),
                TypedTerm::BigInteger(rhs) => lhs.partial_cmp(rhs).unwrap(),
                _ => Less,
            },
            TypedTerm::BigInteger(lhs) => match other {
                TypedTerm::SmallInteger(rhs) => lhs.partial_cmp(rhs).unwrap(),
                TypedTerm::Float(rhs) => lhs.partial_cmp(rhs).unwrap(),
                TypedTerm::BigInteger(rhs) => lhs.cmp(rhs),
                _ => Less,
            },
            TypedTerm::Reference(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_) | TypedTerm::BigInteger(_) => Greater,
                TypedTerm::Reference(rhs) => lhs.cmp(rhs),
                TypedTerm::ExternalReference(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::Atom(_) => Greater,
                _ => Less,
            },
            TypedTerm::ExternalReference(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_) | TypedTerm::BigInteger(_) => Greater,
                TypedTerm::Reference(rhs) => rhs.as_ref().partial_cmp(lhs).unwrap().reverse(),
                TypedTerm::ExternalReference(rhs) => lhs.partial_cmp(rhs).unwrap(),
                TypedTerm::Atom(_) => Greater,
                _ => Less,
            },
            TypedTerm::Closure(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_)
                | TypedTerm::BigInteger(_)
                | TypedTerm::Reference(_)
                | TypedTerm::ExternalReference(_) => Greater,
                TypedTerm::Closure(rhs) => lhs.cmp(rhs),
                TypedTerm::Atom(_) => Greater,
                _ => Less,
            },
            TypedTerm::ExternalPid(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_)
                | TypedTerm::BigInteger(_)
                | TypedTerm::Reference(_)
                | TypedTerm::ExternalReference(_)
                | TypedTerm::Closure(_)
                | TypedTerm::ExternalPort(_) => Greater,
                TypedTerm::ExternalPid(rhs) => lhs.cmp(rhs),
                TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) => Greater,
                _ => Less,
            },
            TypedTerm::Tuple(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_)
                | TypedTerm::BigInteger(_)
                | TypedTerm::Reference(_)
                | TypedTerm::ExternalReference(_)
                | TypedTerm::Closure(_)
                | TypedTerm::ExternalPort(_)
                | TypedTerm::ExternalPid(_) => Greater,
                TypedTerm::Tuple(rhs) => lhs.cmp(rhs),
                TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) => Greater,
                _ => Less,
            },
            TypedTerm::Map(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_)
                | TypedTerm::BigInteger(_)
                | TypedTerm::Reference(_)
                | TypedTerm::Closure(_)
                | TypedTerm::ExternalPort(_)
                | TypedTerm::ExternalPid(_)
                | TypedTerm::Tuple(_) => Greater,
                TypedTerm::Map(rhs) => lhs.cmp(rhs),
                TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) => Greater,
                _ => Less,
            },
            // Bitstrings in likely order
            TypedTerm::HeapBinary(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.cmp(rhs),
                TypedTerm::ProcBin(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::BinaryLiteral(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::SubBinary(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::MatchContext(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                _ => Greater,
            },
            TypedTerm::ProcBin(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::ProcBin(rhs) => lhs.partial_cmp(rhs).unwrap(),
                TypedTerm::BinaryLiteral(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::SubBinary(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::MatchContext(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                _ => Greater,
            },
            TypedTerm::BinaryLiteral(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::ProcBin(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::BinaryLiteral(rhs) => lhs.partial_cmp(rhs).unwrap(),
                TypedTerm::SubBinary(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::MatchContext(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                _ => Greater,
            },
            TypedTerm::SubBinary(lhs) => match other {
                TypedTerm::HeapBinary(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::ProcBin(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::BinaryLiteral(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                TypedTerm::SubBinary(rhs) => lhs.cmp(rhs),
                TypedTerm::MatchContext(rhs) => lhs.as_ref().partial_cmp(rhs).unwrap(),
                _ => Greater,
            },
            TypedTerm::MatchContext(lhs) => {
                unimplemented!("match_context ({:?}) cmp {:?}", lhs, other)
            }
            TypedTerm::Atom(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_) | TypedTerm::BigInteger(_) => Greater,
                TypedTerm::Atom(rhs) => lhs.cmp(rhs),
                _ => Less,
            },
            TypedTerm::Port(lhs) => unimplemented!("Port {:?} cmp {:?}", lhs, other),
            TypedTerm::ExternalPort(lhs) => {
                unimplemented!("ExternalPort {:?} cmp {:?}", lhs, other)
            }
            TypedTerm::Pid(lhs) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Float(_)
                | TypedTerm::BigInteger(_)
                | TypedTerm::Reference(_)
                | TypedTerm::Closure(_)
                | TypedTerm::ExternalPort(_) => Greater,
                TypedTerm::Atom(_) | TypedTerm::Port(_) => Greater,
                TypedTerm::Pid(rhs) => lhs.cmp(rhs),
                _ => Less,
            },
            TypedTerm::Nil => match other {
                TypedTerm::SmallInteger(_)
                | TypedTerm::Float(_)
                | TypedTerm::BigInteger(_)
                | TypedTerm::Reference(_)
                | TypedTerm::Closure(_)
                | TypedTerm::ExternalPort(_)
                | TypedTerm::ExternalPid(_)
                | TypedTerm::Tuple(_)
                | TypedTerm::Map(_)
                | TypedTerm::Atom(_)
                | TypedTerm::Port(_)
                | TypedTerm::Pid(_) => Greater,
                TypedTerm::Nil => Equal,
                _ => Less,
            },
            TypedTerm::List(lhs) => match other {
                TypedTerm::SmallInteger(_)
                | TypedTerm::Float(_)
                | TypedTerm::BigInteger(_)
                | TypedTerm::Reference(_)
                | TypedTerm::Closure(_)
                | TypedTerm::ExternalPort(_)
                | TypedTerm::ExternalPid(_)
                | TypedTerm::Tuple(_)
                | TypedTerm::Map(_)
                | TypedTerm::Atom(_)
                | TypedTerm::Port(_)
                | TypedTerm::Pid(_)
                | TypedTerm::Nil => Greater,
                TypedTerm::List(rhs) => lhs.as_ref().cmp(rhs),
                _ => Less,
            },
            TypedTerm::ResourceReference(lhs) => {
                unimplemented!("ResourceReference {:?} cmp {:?}", lhs, other)
            }
        }
    }
}

/// All terms in Erlang and Elixir are completely ordered.
///
/// number < atom < reference < function < port < pid < tuple < map < list < bitstring
///
/// > When comparing two numbers of different types (a number being either an integer or a float), a
/// > conversion to the type with greater precision will always occur, unless the comparison
/// > operator used is either === or !==. A float will be considered more precise than an integer,
/// > unless the float is greater/less than +/-9007199254740992.0 respectively, at which point all
/// > the significant figures of the float are to the left of the decimal point. This behavior
/// > exists so that the comparison of large numbers remains transitive.
/// >
/// > The collection types are compared using the following rules:
/// >
/// > * Tuples are compared by size, then element by element.
/// > * Maps are compared by size, then by keys in ascending term order, then by values in key
/// >   order.   In the specific case of maps' key ordering, integers are always considered to be
/// >   less than floats.
/// > * Lists are compared element by element.
/// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
/// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
impl PartialOrd<TypedTerm> for TypedTerm {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl CloneToProcess for TypedTerm {
    fn clone_to_process(&self, process: &Process) -> Term {
        use TypedTerm::*;
        // Immediates are just copied and returned, all other terms
        // are expected to require allocation, so we delegate to those types
        match self {
            &Atom(term) => term.encode().unwrap(),
            &Pid(term) => term.encode().unwrap(),
            &Port(term) => term.encode().unwrap(),
            &SmallInteger(term) => term.encode().unwrap(),
            &Float(ref term) => term.clone_to_process(process),
            &List(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &Map(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &Reference(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &ExternalPid(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &ExternalPort(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &ExternalReference(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &ResourceReference(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &MatchContext(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &BinaryLiteral(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &SubBinary(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &ProcBin(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &BigInteger(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &Tuple(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &Closure(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &HeapBinary(term_ptr) => term_ptr.as_ref().clone_to_process(process),
            &Nil => Term::NIL,
        }
    }

    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        use TypedTerm::*;
        // Immediates are just copied and returned, all other terms
        // are expected to require allocation, so we delegate to those types
        match self {
            &Atom(term) => Ok(term.encode().unwrap()),
            &Pid(term) => Ok(term.encode().unwrap()),
            &Port(term) => Ok(term.encode().unwrap()),
            &SmallInteger(term) => Ok(term.encode().unwrap()),
            &Float(ref term) => term.clone_to_heap(heap),
            &List(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &Map(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &Reference(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &ExternalPid(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &ExternalPort(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &ExternalReference(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &ResourceReference(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &MatchContext(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &BinaryLiteral(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &SubBinary(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &ProcBin(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &BigInteger(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &Tuple(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &Closure(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &HeapBinary(term_ptr) => term_ptr.as_ref().clone_to_heap(heap),
            &Nil => Ok(Term::NIL),
        }
    }

    fn size_in_words(&self) -> usize {
        use TypedTerm::*;
        // Immediates are just copied and returned, all other terms
        // are expected to require allocation, so we delegate to those types
        match self {
            &Atom(_term) => 1,
            &Pid(_term) => 1,
            &Port(_term) => 1,
            &SmallInteger(_term) => 1,
            &Float(ref term) => term.size_in_words(),
            &List(term_ptr) => term_ptr.size_in_words(),
            &Map(term_ptr) => term_ptr.size_in_words(),
            &Reference(term_ptr) => term_ptr.size_in_words(),
            &ExternalPid(term_ptr) => term_ptr.size_in_words(),
            &ExternalPort(term_ptr) => term_ptr.size_in_words(),
            &ExternalReference(term_ptr) => term_ptr.size_in_words(),
            &ResourceReference(term_ptr) => term_ptr.size_in_words(),
            &MatchContext(term_ptr) => term_ptr.size_in_words(),
            &BinaryLiteral(term_ptr) => term_ptr.size_in_words(),
            &SubBinary(term_ptr) => term_ptr.size_in_words(),
            &ProcBin(term_ptr) => term_ptr.size_in_words(),
            &BigInteger(term_ptr) => term_ptr.size_in_words(),
            &Tuple(term_ptr) => term_ptr.size_in_words(),
            &Closure(term_ptr) => term_ptr.size_in_words(),
            &HeapBinary(term_ptr) => term_ptr.size_in_words(),
            &Nil => 1,
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
            TypedTerm::BigInteger(big_integer) => Ok(big_integer.as_ref().into()),
            TypedTerm::Float(float) => Ok(float.into()),
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
            _ => Err(TypeError),
        }
    }
}

impl TryInto<Vec<u8>> for TypedTerm {
    type Error = Exception;

    fn try_into(self) -> Result<Vec<u8>, Self::Error> {
        match self {
            TypedTerm::BinaryLiteral(bin_ptr) => bin_ptr.as_ref().try_into(),
            TypedTerm::HeapBinary(bin_ptr) => bin_ptr.as_ref().try_into(),
            TypedTerm::SubBinary(bin_ptr) => bin_ptr.as_ref().try_into(),
            TypedTerm::ProcBin(bin_ptr) => bin_ptr.as_ref().try_into(),
            TypedTerm::MatchContext(bin_ptr) => bin_ptr.as_ref().try_into(),
            _ => Err(badarg!().into()),
        }
    }
}
