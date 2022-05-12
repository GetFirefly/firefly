mod atom;
mod binary;
mod closure;
mod float;
mod index;
mod integer;
mod list;
mod map;
mod node;
mod opaque;
mod pid;
mod port;
mod reference;
mod tuple;

pub use self::atom::Atom;
pub use self::binary::Binary;
pub use self::closure::Closure;
pub use self::float::Float;
pub use self::index::{NonPrimitiveIndex, OneBasedIndex, TupleIndex, ZeroBasedIndex};
pub use self::integer::BigInteger;
pub use self::list::Cons;
pub use self::map::Map;
pub use self::node::Node;
pub use self::opaque::OpaqueTerm;
pub use self::pid::{Pid, ProcessId};
pub use self::port::{Port, PortId};
pub use self::reference::{LocalRef, Reference};
pub use self::tuple::Tuple;

use alloc::alloc::Allocator;
use core::any::Any;
use core::ops::Deref;
use core::ptr::NonNull;

use static_assertions::assert_eq_size;

use liblumen_alloc::gc::GcBox;

assert_eq_size!(Term, (u32, u64));

#[derive(Debug, Copy, Clone, Hash)]
#[repr(C)]
pub enum Term {
    None,
    Nil,
    Bool(bool),
    Atom(Atom),
    Int(i64),
    BigInt(GcBox<BigInteger>),
    Float(Float),
    Cons(NonNull<Cons>),
    Tuple(NonNull<Tuple>),
    Map(GcBox<Map>),
    Closure(GcBox<Closure>),
    Pid(GcBox<Pid>),
    Port(GcBox<Port>),
    Reference(GcBox<Reference>),
    Binary(GcBox<Binary>),
}
impl Term {
    pub fn clone_in<A: Allocator>(&self, alloc: A) -> Self {
        match self {
            Self::None => Self::None,
            Self::Nil => Self::Nil,
            Self::Bool(b) => Self::Bool(*b),
            Self::Atom(a) => Self::Atom(*a),
            Self::Int(i) => Self::Int(*i),
            Self::Float(f) => Self::Float(*f),
            Self::BigInt(i) => {
                let mut empty = GcBox::new_uninit_in(alloc);
                empty.write((&**i).clone());
                Self::BigInt(unsafe { empty.assume_init() })
            }
            Self::Cons(boxed) => {
                let old = unsafe { boxed.as_ref() };
                let cons = Cons::new_in(alloc).unwrap();
                unsafe {
                    cons.as_ptr().write(*old);
                }
                Self::Cons(cons)
            }
            Self::Tuple(tup) => {
                let tup = unsafe { tup.as_ref() };
                let len = tup.len();
                let mut cloned = Tuple::new_in(len, alloc).unwrap();
                unsafe {
                    cloned.as_mut().copy_from_slice(tup.as_slice());
                }
                Self::Tuple(cloned)
            }
            Self::Map(map) => Self::Map(GcBox::new_in((&**map).clone(), alloc)),
            Self::Closure(fun) => {
                let len = fun.len();
                let mut cloned = GcBox::<Closure>::with_capacity_in(len, alloc);
                cloned.copy_from(&fun);
                Self::Closure(cloned)
            }
            Self::Pid(pid) => Self::Pid(GcBox::new_in((&**pid).clone(), alloc)),
            Self::Port(port) => Self::Port(GcBox::new_in((&**port).clone(), alloc)),
            Self::Reference(r) => Self::Reference(GcBox::new_in((&**r).clone(), alloc)),
            Self::Binary(b) => {
                let flags = b.flags();
                let len = b.len();
                let mut cloned = GcBox::<Binary>::with_capacity_in(len, alloc);
                unsafe {
                    cloned.set_flags(flags);
                }
                cloned.copy_from_slice(b.as_bytes());
                Self::Binary(cloned)
            }
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            _ => false,
        }
    }

    pub fn is_nil(&self) -> bool {
        match self {
            Self::Nil => true,
            _ => false,
        }
    }

    pub fn as_cons(&self) -> Option<&Cons> {
        match self {
            Self::Cons(ptr) => Some(unsafe { ptr.as_ref() }),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&Tuple> {
        match self {
            Self::Tuple(ptr) => Some(unsafe { ptr.as_ref() }),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&Map> {
        match self {
            Self::Map(map) => Some(map.deref()),
            _ => None,
        }
    }

    pub fn as_closure(&self) -> Option<&Closure> {
        match self {
            Self::Closure(fun) => Some(fun.deref()),
            _ => None,
        }
    }
    pub fn as_pid(&self) -> Option<&Pid> {
        match self {
            Self::Pid(pid) => Some(pid.deref()),
            _ => None,
        }
    }

    pub fn as_port(&self) -> Option<&Port> {
        match self {
            Self::Port(port) => Some(port.deref()),
            _ => None,
        }
    }

    pub fn as_reference(&self) -> Option<&Reference> {
        match self {
            Self::Reference(r) => Some(r.deref()),
            _ => None,
        }
    }

    pub fn as_binary(&self) -> Option<&Binary> {
        match self {
            Self::Binary(bin) => Some(bin.deref()),
            _ => None,
        }
    }

    pub fn exact_eq(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            return false;
        }
        self.eq(other)
    }
}
impl From<bool> for Term {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}
impl From<Atom> for Term {
    fn from(a: Atom) -> Self {
        Self::Atom(a)
    }
}
impl TryFrom<i64> for Term {
    type Error = ();
    fn try_from(i: i64) -> Result<Self, ()> {
        if OpaqueTerm::is_small_integer(i) {
            Ok(Self::Int(i))
        } else {
            Err(())
        }
    }
}
impl From<GcBox<BigInteger>> for Term {
    fn from(i: GcBox<BigInteger>) -> Self {
        Self::BigInt(i)
    }
}
impl From<f64> for Term {
    fn from(f: f64) -> Self {
        Self::Float(f.into())
    }
}
impl From<Float> for Term {
    fn from(f: Float) -> Self {
        Self::Float(f)
    }
}
impl From<NonNull<Cons>> for Term {
    fn from(term: NonNull<Cons>) -> Self {
        Self::Cons(term)
    }
}
impl From<NonNull<Tuple>> for Term {
    fn from(term: NonNull<Tuple>) -> Self {
        Self::Tuple(term)
    }
}
impl From<GcBox<Map>> for Term {
    fn from(term: GcBox<Map>) -> Self {
        Self::Map(term)
    }
}
impl From<GcBox<Closure>> for Term {
    fn from(term: GcBox<Closure>) -> Self {
        Self::Closure(term)
    }
}
impl From<GcBox<Pid>> for Term {
    fn from(term: GcBox<Pid>) -> Self {
        Self::Pid(term)
    }
}
impl From<GcBox<Port>> for Term {
    fn from(term: GcBox<Port>) -> Self {
        Self::Port(term)
    }
}
impl From<GcBox<Reference>> for Term {
    fn from(term: GcBox<Reference>) -> Self {
        Self::Reference(term)
    }
}
impl From<GcBox<Binary>> for Term {
    fn from(term: GcBox<Binary>) -> Self {
        Self::Binary(term)
    }
}
impl TryInto<bool> for Term {
    type Error = Term;
    fn try_into(self) -> Result<bool, Self::Error> {
        match self {
            Self::Bool(b) => Ok(b),
            Self::Atom(a) if a.is_boolean() => Ok(a.as_boolean()),
            other => Err(other),
        }
    }
}
impl TryInto<Atom> for Term {
    type Error = Term;
    fn try_into(self) -> Result<Atom, Self::Error> {
        match self {
            Self::Atom(a) => Ok(a),
            Self::Bool(b) => Ok(b.into()),
            other => Err(other),
        }
    }
}
impl TryInto<i64> for Term {
    type Error = Term;
    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            Self::Int(i) => Ok(i),
            Self::BigInt(i) => match i.as_i64() {
                Some(i) => Ok(i),
                None => Err(self),
            },
            other => Err(other),
        }
    }
}
impl TryInto<NonNull<Cons>> for Term {
    type Error = Term;
    fn try_into(self) -> Result<NonNull<Cons>, Self::Error> {
        match self {
            Self::Cons(c) => Ok(c),
            other => Err(other),
        }
    }
}
impl TryInto<NonNull<Tuple>> for Term {
    type Error = Term;
    fn try_into(self) -> Result<NonNull<Tuple>, Self::Error> {
        match self {
            Self::Tuple(t) => Ok(t),
            other => Err(other),
        }
    }
}
impl TryInto<GcBox<BigInteger>> for Term {
    type Error = Term;
    fn try_into(self) -> Result<GcBox<BigInteger>, Self::Error> {
        match self {
            Self::BigInt(i) => Ok(i),
            other => Err(other),
        }
    }
}
impl TryInto<f64> for Term {
    type Error = Term;
    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            Self::Float(f) => Ok(f.into()),
            other => Err(other),
        }
    }
}
impl TryInto<Float> for Term {
    type Error = Term;
    fn try_into(self) -> Result<Float, Self::Error> {
        match self {
            Self::Float(f) => Ok(f),
            other => Err(other),
        }
    }
}
impl Eq for Term {}
impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::None => other.is_none(),
            Self::Nil => other.is_nil(),
            Self::Bool(x) => match other {
                Self::Bool(y) => x == y,
                _ => false,
            },
            Self::Atom(x) => match other {
                Self::Atom(y) => x == y,
                _ => false,
            },
            Self::Int(x) => match other {
                Self::Int(y) => x == y,
                Self::BigInt(y) => (&**y) == x,
                Self::Float(y) => y == x,
                _ => false,
            },
            Self::BigInt(x) => match other {
                Self::Int(y) => (&**x) == y,
                Self::BigInt(y) => x == y,
                Self::Float(y) => y == (&**x),
                _ => false,
            },
            Self::Float(x) => match other {
                Self::Float(y) => x == y,
                Self::Int(y) => x == y,
                Self::BigInt(y) => x == (&**y),
                _ => false,
            },
            Self::Cons(x) => match other {
                Self::Cons(y) => unsafe { x.as_ref().eq(y.as_ref()) },
                _ => false,
            },
            Self::Tuple(x) => match other {
                Self::Tuple(y) => unsafe { x.as_ref().eq(y.as_ref()) },
                _ => false,
            },
            Self::Map(x) => match other {
                Self::Map(y) => x == y,
                _ => false,
            },
            Self::Closure(x) => match other {
                Self::Closure(y) => x == y,
                _ => false,
            },
            Self::Pid(x) => match other {
                Self::Pid(y) => x == y,
                _ => false,
            },
            Self::Port(x) => match other {
                Self::Port(y) => x == y,
                _ => false,
            },
            Self::Reference(x) => match other {
                Self::Reference(y) => x == y,
                _ => false,
            },
            Self::Binary(x) => match other {
                Self::Binary(y) => x == y,
                _ => false,
            },
        }
    }
}
impl PartialOrd for Term {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Term {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        match self {
            // None is always least
            Self::None => {
                if other.is_none() {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            }
            // Numbers are smaller than all other terms, using whichever type has the highest precision
            Self::Int(x) => match other {
                Self::None => Ordering::Greater,
                Self::Int(y) => x.cmp(y),
                Self::BigInt(y) => match (&**y).partial_cmp(x).unwrap() {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                    equal => equal,
                },
                Self::Float(y) => match y.partial_cmp(x).unwrap() {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                    equal => equal,
                },
                _ => Ordering::Less,
            },
            Self::BigInt(x) => match other {
                Self::None => Ordering::Greater,
                Self::Int(y) => (&**x).partial_cmp(y).unwrap(),
                Self::BigInt(y) => (&**x).cmp(&**y),
                Self::Float(y) => (&**x).partial_cmp(y).unwrap(),
                _ => Ordering::Less,
            },
            Self::Float(x) => match other {
                Self::None => Ordering::Greater,
                Self::Float(y) => x.partial_cmp(y).unwrap(),
                Self::Int(y) => x.partial_cmp(y).unwrap(),
                Self::BigInt(y) => x.partial_cmp(&**y).unwrap(),
                _ => Ordering::Less,
            },
            Self::Bool(x) => match other {
                Self::Bool(y) => x.cmp(y),
                Self::Atom(a) if a.is_boolean() => x.cmp(&a.as_boolean()),
                Self::Atom(_) => Ordering::Less,
                Self::None | Self::Int(_) | Self::BigInt(_) | Self::Float(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Atom(x) => match other {
                Self::Atom(y) => x.cmp(y),
                Self::Bool(y) if x.is_boolean() => x.as_boolean().cmp(y),
                Self::Bool(_) => Ordering::Greater,
                Self::None | Self::Int(_) | Self::BigInt(_) | Self::Float(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Reference(x) => match other {
                Self::Reference(y) => x.cmp(y),
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Closure(x) => match other {
                Self::Closure(y) => x.cmp(y),
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_)
                | Self::Reference(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Port(x) => match other {
                Self::Port(y) => x.cmp(y),
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_)
                | Self::Reference(_)
                | Self::Closure(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Pid(x) => match other {
                Self::Pid(y) => x.cmp(y),
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_)
                | Self::Reference(_)
                | Self::Closure(_)
                | Self::Port(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Tuple(x) => match other {
                Self::Tuple(y) => unsafe { x.as_ref().cmp(y.as_ref()) },
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_)
                | Self::Reference(_)
                | Self::Closure(_)
                | Self::Port(_)
                | Self::Pid(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Map(x) => match other {
                Self::Map(y) => x.cmp(y),
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_)
                | Self::Reference(_)
                | Self::Closure(_)
                | Self::Port(_)
                | Self::Pid(_)
                | Self::Tuple(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Nil => match other {
                Self::Nil => Ordering::Equal,
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_)
                | Self::Reference(_)
                | Self::Closure(_)
                | Self::Port(_)
                | Self::Pid(_)
                | Self::Tuple(_)
                | Self::Map(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Cons(x) => match other {
                Self::Cons(y) => unsafe { x.as_ref().cmp(y.as_ref()) },
                Self::None
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_)
                | Self::Bool(_)
                | Self::Atom(_)
                | Self::Reference(_)
                | Self::Closure(_)
                | Self::Port(_)
                | Self::Pid(_)
                | Self::Tuple(_)
                | Self::Map(_)
                | Self::Nil => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Binary(x) => match other {
                Self::Binary(y) => x.cmp(y),
                _ => Ordering::Greater,
            },
        }
    }
}

#[cfg(test)]
mod test {
    use core::alloc::Layout;
    use core::ptr::NonNull;

    use crate::process::ProcessHeap;

    use super::*;

    macro_rules! cons {
        ($heap:expr, $tail:expr) => {{
            cons!($heap, $tail, Term::Nil)
        }};

        ($heap:expr, $head:expr, $tail:expr) => {{
            let layout = Layout::new::<Cons>();
            let ptr: NonNull<Cons> = $heap.allocate(layout).unwrap().cast();
            ptr.as_ptr().write(Cons::cons($head, $tail));
            Term::Cons(ptr)
        }};

        ($heap:expr, $head:expr, $tail:expr, $($rest:expr,)+) => {{
            let rest = cons!($heap, $($rest),+);
            let tail = cons!($heap, $tail, tail);
            cons!($heap, $head, tail);
        }}
    }

    #[test]
    fn list_test() {
        let heap = ProcessHeap::new();

        let list = cons!(
            &heap,
            Term::Binary(Binary::from_str("foo")),
            Term::Float(f64::MIN.into()),
            Term::Int(42),
        );

        let opaque: OpaqueTerm = list.into();
        let value: Term = opaque.into();
        let cons: NonNull<Cons> = value.try_into().unwrap();
        let mut iter = cons.iter();

        assert_eq!(iter.next(), Some(Ok(Term::Binary(Binary::from_str("foo")))));
        assert_eq!(iter.next(), Some(Ok(Term::Float(f64::MIN.into()))));
        assert_eq!(iter.next(), Some(Ok(Term::Int(42))));
        assert_eq!(iter.next(), Some(Ok(Term::Nil)));
        assert_eq!(iter.next(), None);
    }
}
