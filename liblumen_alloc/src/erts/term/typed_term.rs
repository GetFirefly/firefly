use core::cmp;
use core::convert::TryInto;
use core::hash::{Hash, Hasher};

use alloc::string::String;

use num_bigint::{BigInt, Sign};

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
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
    HeapBinary(Boxed<HeapBin>),
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

impl Hash for TypedTerm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::List(cons) => cons.hash(state),
            Self::Tuple(tuple) => tuple.hash(state),
            Self::Map(map) => map.hash(state),
            Self::Boxed(boxed) => boxed.to_typed_term().unwrap().hash(state),
            Self::Literal(literal) => literal.hash(state),
            Self::Pid(pid) => pid.hash(state),
            Self::Port(port) => port.hash(state),
            Self::Reference(reference) => reference.hash(state),
            Self::ExternalPid(external_pid) => external_pid.hash(state),
            Self::ExternalPort(external_port) => external_port.hash(state),
            Self::ExternalReference(external_reference) => external_reference.hash(state),
            Self::SmallInteger(small_integer) => small_integer.hash(state),
            Self::BigInteger(big_integer) => big_integer.hash(state),
            Self::Float(float) => float.hash(state),
            Self::Atom(atom) => atom.hash(state),
            Self::ProcBin(process_binary) => process_binary.hash(state),
            Self::HeapBinary(heap_binary) => heap_binary.hash(state),
            Self::SubBinary(subbinary) => subbinary.hash(state),
            Self::MatchContext(match_context) => match_context.hash(state),
            Self::Closure(closure) => closure.hash(state),
            Self::Catch => Term::CATCH.as_usize().hash(state),
            Self::Nil => Term::NIL.as_usize().hash(state),
            Self::None => Term::NONE.as_usize().hash(state),
        };
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
        match (self, other) {
            (&Self::Catch, &Self::Catch) => true,
            (&Self::Nil, &Self::Nil) => true,
            (&Self::None, &Self::None) => true,
            (Self::Boxed(self_boxed), Self::Boxed(other_boxed)) => self_boxed
                .to_typed_term()
                .unwrap()
                .eq(&other_boxed.to_typed_term().unwrap()),
            // For BitString types do `eq` in order of complexity, so that we don't have to
            // implement it twice
            (Self::ProcBin(proc_bin), Self::HeapBinary(boxed_heap_binary)) => {
                proc_bin.eq(boxed_heap_binary.as_ref())
            }
            (Self::ProcBin(proc_bin), Self::SubBinary(subbinary)) => subbinary.eq(proc_bin),
            (Self::ProcBin(proc_bin), Self::MatchContext(match_context)) => {
                match_context.eq(proc_bin)
            }
            (
                Self::HeapBinary(self_boxed_heap_binary),
                Self::HeapBinary(other_boxed_heap_binary),
            ) => self_boxed_heap_binary
                .as_ref()
                .eq(other_boxed_heap_binary.as_ref()),
            (Self::HeapBinary(boxed_heap_binary), Self::ProcBin(proc_bin)) => {
                proc_bin.eq(boxed_heap_binary.as_ref())
            }
            (Self::HeapBinary(boxed_heap_binary), Self::SubBinary(subbinary)) => {
                subbinary.eq(boxed_heap_binary.as_ref())
            }
            (Self::HeapBinary(boxed_heap_binary), Self::MatchContext(match_context)) => {
                match_context.eq(boxed_heap_binary.as_ref())
            }
            (Self::SubBinary(subbinary), Self::ProcBin(proc_bin)) => subbinary.eq(proc_bin),
            (Self::SubBinary(subbinary), Self::HeapBinary(boxed_heap_binary)) => {
                subbinary.eq(boxed_heap_binary.as_ref())
            }
            (Self::SubBinary(subbinary), Self::MatchContext(match_context)) => {
                subbinary.eq(match_context)
            }
            (Self::MatchContext(match_context), Self::ProcBin(proc_bin)) => {
                match_context.eq(proc_bin)
            }
            (Self::MatchContext(match_context), Self::HeapBinary(boxed_heap_binary)) => {
                match_context.eq(boxed_heap_binary.as_ref())
            }
            (Self::MatchContext(match_context), Self::SubBinary(subbinary)) => {
                subbinary.eq(match_context)
            }
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
            TypedTerm::SmallInteger(self_small_integer) => match other {
                TypedTerm::SmallInteger(other_small_integer) => {
                    self_small_integer.cmp(other_small_integer)
                }
                TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap() {
                    // Flip order so that only type that will be conversion target needs to
                    // implement `PartialOrd` between types.
                    TypedTerm::Float(other_float) => other_float
                        .partial_cmp(self_small_integer)
                        .unwrap()
                        .reverse(),
                    TypedTerm::BigInteger(other_big_integer) => other_big_integer
                        .partial_cmp(self_small_integer)
                        .unwrap()
                        .reverse(),
                    _ => Less,
                },
                _ => Less,
            },
            // In place of first boxed: Float.
            TypedTerm::Boxed(self_boxed) => {
                let self_unboxed = self_boxed.to_typed_term().unwrap();

                match self_unboxed {
                    TypedTerm::Float(self_float) => match other {
                        TypedTerm::SmallInteger(other_small_integer) => {
                            self_float.partial_cmp(other_small_integer).unwrap()
                        }
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(other_float) => self_float.cmp(&other_float),
                            TypedTerm::BigInteger(other_big_integer) => other_big_integer
                                .partial_cmp(&self_float)
                                .unwrap()
                                .reverse(),
                            _ => Less,
                        },
                        _ => Less,
                    },
                    TypedTerm::BigInteger(self_big_integer) => match other {
                        TypedTerm::SmallInteger(other_small_integer) => {
                            self_big_integer.partial_cmp(other_small_integer).unwrap()
                        }
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(other_float) => {
                                self_big_integer.partial_cmp(&other_float).unwrap()
                            }
                            TypedTerm::BigInteger(other_big_integer) => {
                                self_big_integer.cmp(&other_big_integer)
                            }
                            _ => Less,
                        },
                        _ => Less,
                    },
                    TypedTerm::Reference(self_reference) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_) | TypedTerm::BigInteger(_) => Greater,
                            TypedTerm::Reference(other_reference) => {
                                self_reference.cmp(&other_reference)
                            }
                            TypedTerm::ExternalReference(other_external_reference) => {
                                other_external_reference
                                    .partial_cmp(&self_reference)
                                    .unwrap()
                                    .reverse()
                            }
                            _ => Less,
                        },
                        TypedTerm::Atom(_) => Greater,
                        _ => Less,
                    },
                    TypedTerm::Closure(self_closure) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_)
                            | TypedTerm::BigInteger(_)
                            | TypedTerm::Reference(_)
                            | TypedTerm::ExternalReference(_) => Greater,
                            TypedTerm::Closure(other_closure) => {
                                self_closure.cmp(other_closure.as_ref())
                            }
                            _ => Less,
                        },
                        TypedTerm::Atom(_) => Greater,
                        _ => Less,
                    },
                    TypedTerm::ExternalPid(self_external_pid) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_)
                            | TypedTerm::BigInteger(_)
                            | TypedTerm::Reference(_)
                            | TypedTerm::ExternalReference(_)
                            | TypedTerm::Closure(_)
                            | TypedTerm::ExternalPort(_) => Greater,
                            TypedTerm::ExternalPid(other_external_pid) => {
                                self_external_pid.cmp(&other_external_pid)
                            }
                            _ => Less,
                        },
                        TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) => Greater,
                        _ => Less,
                    },
                    TypedTerm::Tuple(self_tuple) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_)
                            | TypedTerm::BigInteger(_)
                            | TypedTerm::Reference(_)
                            | TypedTerm::ExternalReference(_)
                            | TypedTerm::Closure(_)
                            | TypedTerm::ExternalPort(_)
                            | TypedTerm::ExternalPid(_) => Greater,
                            TypedTerm::Tuple(other_tuple) => self_tuple.cmp(other_tuple.as_ref()),
                            _ => Less,
                        },
                        TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) => Greater,
                        _ => Less,
                    },
                    TypedTerm::Map(self_map) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_)
                            | TypedTerm::BigInteger(_)
                            | TypedTerm::Reference(_)
                            | TypedTerm::Closure(_)
                            | TypedTerm::ExternalPort(_)
                            | TypedTerm::ExternalPid(_)
                            | TypedTerm::Tuple(_) => Greater,
                            TypedTerm::Map(other_map) => self_map.cmp(other_map.as_ref()),
                            _ => Less,
                        },
                        TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) => Greater,
                        _ => Less,
                    },
                    // Bitstrings in likely order
                    TypedTerm::HeapBinary(self_heap_binary) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_)
                            | TypedTerm::BigInteger(_)
                            | TypedTerm::Reference(_)
                            | TypedTerm::ExternalReference(_)
                            | TypedTerm::Closure(_)
                            | TypedTerm::ExternalPort(_)
                            | TypedTerm::ExternalPid(_)
                            | TypedTerm::Tuple(_)
                            | TypedTerm::Map(_)
                            | TypedTerm::List(_) => Greater,
                            TypedTerm::HeapBinary(other_heap_binary) => {
                                self_heap_binary.as_ref().cmp(other_heap_binary.as_ref())
                            }
                            TypedTerm::ProcBin(other_process_binary) => other_process_binary
                                .partial_cmp(&self_heap_binary)
                                .unwrap()
                                .reverse(),
                            TypedTerm::SubBinary(other_subbinary) => other_subbinary
                                .partial_cmp(self_heap_binary.as_ref())
                                .unwrap()
                                .reverse(),
                            TypedTerm::MatchContext(other_match_context) => other_match_context
                                .partial_cmp(self_heap_binary.as_ref())
                                .unwrap()
                                .reverse(),
                            _ => unreachable!(),
                        },
                        TypedTerm::Atom(_) | TypedTerm::Pid(_) | TypedTerm::Nil => Greater,
                        _ => unreachable!(),
                    },
                    TypedTerm::ProcBin(self_process_binary) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_)
                            | TypedTerm::BigInteger(_)
                            | TypedTerm::Reference(_)
                            | TypedTerm::ExternalReference(_)
                            | TypedTerm::Closure(_)
                            | TypedTerm::ExternalPort(_)
                            | TypedTerm::ExternalPid(_)
                            | TypedTerm::Tuple(_)
                            | TypedTerm::Map(_)
                            | TypedTerm::List(_) => Greater,
                            TypedTerm::HeapBinary(other_heap_binary) => {
                                self_process_binary.partial_cmp(&other_heap_binary).unwrap()
                            }
                            TypedTerm::ProcBin(other_process_binary) => self_process_binary
                                .partial_cmp(&other_process_binary)
                                .unwrap(),
                            TypedTerm::SubBinary(other_subbinary) => other_subbinary
                                .partial_cmp(&self_process_binary)
                                .unwrap()
                                .reverse(),
                            TypedTerm::MatchContext(other_match_context) => other_match_context
                                .partial_cmp(&self_process_binary)
                                .unwrap()
                                .reverse(),
                            _ => unreachable!(),
                        },
                        TypedTerm::Atom(_) | TypedTerm::Pid(_) | TypedTerm::Nil => Greater,
                        _ => unreachable!(),
                    },
                    TypedTerm::SubBinary(self_subbinary) => match other {
                        TypedTerm::SmallInteger(_) => Greater,
                        TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap()
                        {
                            TypedTerm::Float(_)
                            | TypedTerm::BigInteger(_)
                            | TypedTerm::Reference(_)
                            | TypedTerm::ExternalReference(_)
                            | TypedTerm::Closure(_)
                            | TypedTerm::ExternalPort(_)
                            | TypedTerm::ExternalPid(_)
                            | TypedTerm::Tuple(_)
                            | TypedTerm::Map(_)
                            | TypedTerm::List(_) => Greater,
                            TypedTerm::HeapBinary(other_heap_binary) => self_subbinary
                                .partial_cmp(other_heap_binary.as_ref())
                                .unwrap(),
                            TypedTerm::ProcBin(other_process_binary) => {
                                self_subbinary.partial_cmp(&other_process_binary).unwrap()
                            }
                            TypedTerm::SubBinary(other_subbinary) => {
                                self_subbinary.cmp(&other_subbinary)
                            }
                            TypedTerm::MatchContext(other_match_context) => self_subbinary
                                .partial_cmp(&other_match_context)
                                .unwrap()
                                .reverse(),
                            _ => unreachable!(),
                        },
                        TypedTerm::Atom(_) | TypedTerm::Pid(_) | TypedTerm::Nil => Greater,
                        _ => unreachable!(),
                    },
                    TypedTerm::MatchContext(self_match_context) => match other {
                        _ => unimplemented!(
                            "unboxed match_context ({:?}) cmp {:?}",
                            self_match_context,
                            other
                        ),
                    },
                    _ => unimplemented!("unboxed {:?} cmp {:?}", self_unboxed, other),
                }
            }
            TypedTerm::Atom(self_atom) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap() {
                    TypedTerm::Float(_) | TypedTerm::BigInteger(_) => Greater,
                    _ => Less,
                },
                TypedTerm::Atom(other_atom) => self_atom.cmp(other_atom),
                _ => Less,
            },
            TypedTerm::Port(self_port) => match other {
                _ => unimplemented!("Port {:?} cmp {:?}", self_port, other),
            },
            TypedTerm::Pid(self_pid) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap() {
                    TypedTerm::Float(_)
                    | TypedTerm::BigInteger(_)
                    | TypedTerm::Reference(_)
                    | TypedTerm::Closure(_)
                    | TypedTerm::ExternalPort(_) => Greater,
                    _ => Less,
                },
                TypedTerm::Atom(_) | TypedTerm::Port(_) => Greater,
                TypedTerm::Pid(other_pid) => self_pid.cmp(other_pid),
                _ => Less,
            },
            TypedTerm::Nil => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap() {
                    TypedTerm::Float(_)
                    | TypedTerm::BigInteger(_)
                    | TypedTerm::Reference(_)
                    | TypedTerm::Closure(_)
                    | TypedTerm::ExternalPort(_)
                    | TypedTerm::ExternalPid(_)
                    | TypedTerm::Tuple(_)
                    | TypedTerm::Map(_) => Greater,
                    _ => Less,
                },
                TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) => Greater,
                TypedTerm::Nil => Equal,
                _ => Less,
            },
            TypedTerm::List(self_cons) => match other {
                TypedTerm::SmallInteger(_) => Greater,
                TypedTerm::Boxed(other_boxed) => match other_boxed.to_typed_term().unwrap() {
                    TypedTerm::Float(_)
                    | TypedTerm::BigInteger(_)
                    | TypedTerm::Reference(_)
                    | TypedTerm::Closure(_)
                    | TypedTerm::ExternalPort(_)
                    | TypedTerm::ExternalPid(_)
                    | TypedTerm::Tuple(_)
                    | TypedTerm::Map(_) => Greater,
                    _ => Less,
                },
                TypedTerm::Atom(_) | TypedTerm::Port(_) | TypedTerm::Pid(_) | TypedTerm::Nil => {
                    Greater
                }
                TypedTerm::List(other_cons) => self_cons.cmp(other_cons),
                _ => Less,
            },
            // rest are boxed or GC-only
            _ => unreachable!(),
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

    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
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
            &Self::Boxed(ref inner) => inner.to_typed_term().unwrap().size_in_words(),
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
