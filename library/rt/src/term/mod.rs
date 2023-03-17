pub mod atom;
mod binary;
mod closure;
mod convert;
mod fragment;
mod header;
mod index;
mod integer;
mod layout;
mod list;
mod map;
mod opaque;
mod pid;
mod port;
mod reference;
mod tuple;
mod value;

pub use self::atom::{atoms, Atom, AtomData, AtomError};
pub use self::binary::*;
pub use self::closure::{Closure, ClosureFlags};
pub use self::convert::ToTerm;
pub use self::fragment::TermFragment;
pub use self::header::{Boxable, Header, Metadata, Tag};
pub use self::index::{NonPrimitiveIndex, OneBasedIndex, TupleIndex, ZeroBasedIndex};
pub use self::integer::BigInt;
pub use self::layout::LayoutBuilder;
pub use self::list::{Cons, ImproperList, ListBuilder};
pub use self::map::{Map, MapError, SmallMap};
pub use self::opaque::{OpaqueTerm, TermType};
pub use self::pid::Pid;
pub use self::port::{Port, PortId};
pub use self::reference::{Reference, ReferenceId};
pub use self::tuple::Tuple;
pub use self::value::Value;

use firefly_number::{DivisionError, InvalidArithmeticError, Sign, ToPrimitive};
pub use firefly_number::{Float, Int, Number};
use firefly_system::time;

use alloc::alloc::{AllocError, Layout};
use alloc::sync::Arc;
use core::convert::AsRef;
use core::fmt;
use core::ops::Deref;
use core::ptr;

use anyhow::anyhow;
use firefly_alloc::heap::Heap;
use firefly_binary::{Binary, Bitstring, Encoding};

use crate::cmp::ExactEq;
use crate::gc::Gc;

/// `Term` is two things:
///
/// * An enumeration of the types that can be represented as Erlang values,
/// unifying them under a single conceptual value type, i.e. term. As such,
/// Term is how comparisons/equality/etc. are defined between value types.
///
/// * The decoded form of `OpaqueTerm`. `OpaqueTerm` is a compact encoding intended to
/// guarantee that passing around a term value in Erlang code is always the same size (i.e.
/// u64 on 64-bit systems). This invariant is not needed in Rust, and it is more ergonomic
/// and performant to pass around a less compact represenetation. However, we want to preserve
/// the `Copy`-ability of the `OpaqueTerm` representation, so `Term` still requires some care
/// when performing certain operations such as cloning and dropping, where the underlying type
/// may have additional invariants that `Term` does not know about. For example, we use trait
/// objects for the various binary types to keep things simple when working with them as terms,
/// but when garbage collecting, we must make sure that we operate on the concrete type,
/// as some binaries are heap-allocated while others are reference-counted, and the semantics
/// of how those are collected are different.
///
/// See notes on the individual variants for why a specific representation was chosen for that
/// variant.
#[derive(Debug, Clone, Hash)]
pub enum Term {
    None,
    Catch(usize),
    Code(usize),
    Nil,
    Bool(bool),
    Atom(Atom),
    Int(i64),
    BigInt(Gc<BigInt>),
    Float(Float),
    Cons(Gc<Cons>),
    Tuple(Gc<Tuple>),
    Map(Gc<Map>),
    Closure(Gc<Closure>),
    Pid(Gc<Pid>),
    Port(Arc<Port>),
    Reference(Gc<Reference>),
    HeapBinary(Gc<BinaryData>),
    RcBinary(Arc<BinaryData>),
    RefBinary(Gc<BitSlice>),
    ConstantBinary(&'static BinaryData),
}
impl Term {
    pub fn clone_to_fragment(&self) -> Result<TermFragment, AllocError> {
        TermFragment::clone_from(self)
    }

    #[inline(always)]
    pub fn into_opaque(self) -> OpaqueTerm {
        self.into()
    }

    pub fn is_special(&self) -> bool {
        match self {
            Self::Catch(_) | Self::Code(_) => true,
            _ => false,
        }
    }

    pub fn is_refcounted(&self) -> bool {
        match self {
            Self::Port(_) | Self::RcBinary(_) => true,
            _ => false,
        }
    }

    /// Corresponds to `OpaqueTerm::is_box`
    pub fn is_box(&self) -> bool {
        match self {
            Self::BigInt(_)
            | Self::Cons(_)
            | Self::Tuple(_)
            | Self::Map(_)
            | Self::Closure(_)
            | Self::Pid(_)
            | Self::Port(_)
            | Self::Reference(_)
            | Self::HeapBinary(_)
            | Self::RcBinary(_)
            | Self::RefBinary(_) => true,
            _ => false,
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

    pub fn as_cons(&self) -> Option<Gc<Cons>> {
        match self {
            Self::Cons(ptr) => Some(*ptr),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<Gc<Tuple>> {
        match self {
            Self::Tuple(ptr) => Some(*ptr),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&Map> {
        match self {
            Self::Map(map) => Some(map.as_ref()),
            _ => None,
        }
    }

    pub fn as_closure(&self) -> Option<&Closure> {
        match self {
            Self::Closure(fun) => Some(fun.as_ref()),
            _ => None,
        }
    }
    pub fn as_pid(&self) -> Option<&Pid> {
        match self {
            Self::Pid(pid) => Some(pid.as_ref()),
            _ => None,
        }
    }

    pub fn as_port(&self) -> Option<&Port> {
        match self {
            Self::Port(port) => Some(port.as_ref()),
            _ => None,
        }
    }

    pub fn as_reference(&self) -> Option<&Reference> {
        match self {
            Self::Reference(r) => Some(r.as_ref()),
            _ => None,
        }
    }

    pub fn as_bitstring(&self) -> Option<&dyn Bitstring> {
        match self {
            Self::HeapBinary(boxed) => Some(boxed),
            Self::RcBinary(boxed) => Some(boxed),
            Self::RefBinary(boxed) => Some(boxed),
            Self::ConstantBinary(lit) => Some(lit),
            _ => None,
        }
    }

    pub fn as_binary(&self) -> Option<&dyn Bitstring> {
        match self {
            Self::HeapBinary(boxed) if boxed.is_binary() => Some(boxed),
            Self::RcBinary(boxed) if boxed.is_binary() => Some(boxed),
            Self::RefBinary(boxed) if boxed.is_binary() => Some(boxed),
            Self::ConstantBinary(lit) if lit.is_binary() => Some(lit),
            _ => None,
        }
    }

    pub fn is_bitstring(&self) -> bool {
        match self {
            Self::HeapBinary(_)
            | Self::RcBinary(_)
            | Self::RefBinary(_)
            | Self::ConstantBinary(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn as_char(self) -> Result<char, ()> {
        self.try_into()
    }

    pub fn exact_eq(&self, other: &Self) -> bool {
        // With exception of bitstring variants, if the discriminant is different, the
        // types can never be exactly equal
        if core::mem::discriminant(self) != core::mem::discriminant(other) {
            if self.is_bitstring() && other.is_bitstring() {
                return self.eq(other);
            }
            return false;
        }
        self.eq(other)
    }

    #[inline]
    pub fn layout(&self) -> Layout {
        const EMPTY: firefly_alloc::heap::EmptyHeap = firefly_alloc::heap::EmptyHeap;
        self.layout_excluding_heap(&EMPTY)
    }

    pub fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        match self {
            Self::None
            | Self::Catch(_)
            | Self::Code(_)
            | Self::Nil
            | Self::Bool(_)
            | Self::Atom(_)
            | Self::Int(_)
            | Self::Float(_)
            | Self::Port(_)
            | Self::Reference(_)
            | Self::RcBinary(_)
            | Self::ConstantBinary(_) => Layout::new::<()>(),
            Self::BigInt(boxed) => boxed.deref().layout_excluding_heap(heap),
            Self::Cons(boxed) => boxed.deref().layout_excluding_heap(heap),
            Self::Tuple(boxed) => boxed.deref().layout_excluding_heap(heap),
            Self::Map(boxed) => boxed.deref().layout_excluding_heap(heap),
            Self::Closure(boxed) => boxed.deref().layout_excluding_heap(heap),
            Self::Pid(boxed) => boxed.deref().layout_excluding_heap(heap),
            Self::HeapBinary(boxed) => boxed.deref().layout_excluding_heap(heap),
            Self::RefBinary(boxed) => boxed.layout_excluding_heap(heap),
        }
    }

    pub fn clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Result<Self, AllocError> {
        let layout = self.layout_excluding_heap(heap);
        if heap.heap_available() < layout.size() {
            return Err(AllocError);
        }
        Ok(unsafe { self.unsafe_clone_to_heap(heap) })
    }

    pub fn move_to_heap<H: ?Sized + Heap>(self, heap: &H) -> Result<OpaqueTerm, AllocError> {
        let layout = self.layout_excluding_heap(heap);
        if heap.heap_available() < layout.size() {
            return Err(AllocError);
        }
        Ok(unsafe { self.unsafe_move_to_heap(heap) })
    }

    pub unsafe fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Self {
        match self {
            term @ (Self::None
            | Self::Catch(_)
            | Self::Code(_)
            | Self::Nil
            | Self::Bool(_)
            | Self::Atom(_)
            | Self::Int(_)
            | Self::Float(_)) => term.clone(),
            Self::BigInt(boxed) => Self::BigInt(boxed.deref().unsafe_clone_to_heap(heap)),
            Self::Cons(boxed) => Self::Cons(boxed.unsafe_clone_to_heap(heap)),
            Self::Tuple(boxed) => Self::Tuple(boxed.deref().unsafe_clone_to_heap(heap)),
            Self::Map(boxed) => Self::Map(boxed.deref().unsafe_clone_to_heap(heap)),
            Self::Closure(boxed) => Self::Closure(boxed.deref().unsafe_clone_to_heap(heap)),
            Self::Pid(boxed) => Self::Pid(boxed.deref().unsafe_clone_to_heap(heap)),
            Self::Port(ref port) => Self::Port(Arc::clone(port)),
            Self::Reference(boxed) => Self::Reference(boxed.deref().unsafe_clone_to_heap(heap)),
            Self::HeapBinary(boxed) => Self::HeapBinary(boxed.deref().unsafe_clone_to_heap(heap)),
            Self::RcBinary(ref bin) => Self::RcBinary(Arc::clone(bin)),
            Self::RefBinary(boxed) => {
                if heap.contains(Gc::as_ptr(boxed).cast()) || boxed.is_owner_refcounted() {
                    Self::RefBinary(boxed.clone())
                } else {
                    let byte_size = boxed.byte_size();
                    let mut cloned = Gc::<BinaryData>::with_capacity_in(byte_size, heap).unwrap();
                    cloned.copy_from_selection(boxed.as_selection());
                    Self::HeapBinary(cloned)
                }
            }
            Self::ConstantBinary(bytes) => Self::ConstantBinary(bytes),
        }
    }

    pub unsafe fn unsafe_move_to_heap<H: ?Sized + Heap>(self, heap: &H) -> OpaqueTerm {
        match self {
            term @ (Self::None
            | Self::Catch(_)
            | Self::Code(_)
            | Self::Nil
            | Self::Bool(_)
            | Self::Atom(_)
            | Self::Int(_)
            | Self::Float(_)
            | Self::ConstantBinary(_)) => term.into(),
            Self::Cons(mut boxed) => boxed.unsafe_move_to_heap(heap).into(),
            Self::BigInt(boxed) => {
                let cloned = boxed.deref().unsafe_clone_to_heap(heap);
                core::ptr::drop_in_place(boxed.as_non_null_ptr().as_ptr());
                cloned.into()
            }
            Self::Tuple(boxed) => boxed.deref().unsafe_move_to_heap(heap).into(),
            Self::Map(boxed) => boxed.deref().unsafe_move_to_heap(heap).into(),
            Self::Closure(boxed) => boxed.deref().unsafe_move_to_heap(heap).into(),
            Self::Pid(boxed) => {
                let cloned = boxed.deref().unsafe_clone_to_heap(heap);
                core::ptr::drop_in_place(boxed.as_non_null_ptr().as_ptr());
                cloned.into()
            }
            port @ Self::Port(_) => port.into(),
            Self::Reference(boxed) => {
                let cloned: Gc<Reference> = boxed.deref().unsafe_clone_to_heap(heap).into();
                core::ptr::drop_in_place(boxed.as_non_null_ptr().as_ptr());
                cloned.into()
            }
            Self::HeapBinary(boxed) => boxed.deref().unsafe_clone_to_heap(heap).into(),
            rc @ Self::RcBinary(_) => rc.into(),
            Self::RefBinary(boxed) => {
                if heap.contains(Gc::as_ptr(&boxed).cast()) || boxed.is_owner_refcounted() {
                    return Self::RefBinary(boxed).into();
                } else {
                    let byte_size = boxed.byte_size();
                    let mut cloned = Gc::<BinaryData>::with_capacity_in(byte_size, heap).unwrap();
                    cloned.copy_from_selection(boxed.as_selection());
                    Self::HeapBinary(cloned).into()
                }
            }
        }
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
impl TryFrom<usize> for Term {
    type Error = ();
    #[inline]
    fn try_from(i: usize) -> Result<Self, ()> {
        let i: i64 = i.try_into().map_err(|_| ())?;
        i.try_into()
    }
}
impl TryFrom<isize> for Term {
    type Error = ();
    #[inline]
    fn try_from(i: isize) -> Result<Self, ()> {
        (i as i64).try_into()
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
impl From<Gc<Cons>> for Term {
    fn from(term: Gc<Cons>) -> Self {
        Self::Cons(term)
    }
}
impl From<Gc<Tuple>> for Term {
    fn from(term: Gc<Tuple>) -> Self {
        Self::Tuple(term)
    }
}
impl From<Gc<BigInt>> for Term {
    fn from(i: Gc<BigInt>) -> Self {
        Self::BigInt(i)
    }
}
impl From<Gc<Map>> for Term {
    fn from(term: Gc<Map>) -> Self {
        Self::Map(term)
    }
}
impl From<Gc<Closure>> for Term {
    fn from(term: Gc<Closure>) -> Self {
        Self::Closure(term)
    }
}
impl From<Gc<Pid>> for Term {
    fn from(term: Gc<Pid>) -> Self {
        Self::Pid(term)
    }
}
impl From<Gc<Reference>> for Term {
    fn from(term: Gc<Reference>) -> Self {
        Self::Reference(term)
    }
}
impl From<Gc<BinaryData>> for Term {
    fn from(term: Gc<BinaryData>) -> Self {
        Self::HeapBinary(term)
    }
}
impl From<Gc<BitSlice>> for Term {
    fn from(term: Gc<BitSlice>) -> Self {
        Self::RefBinary(term)
    }
}
impl From<Arc<Port>> for Term {
    fn from(term: Arc<Port>) -> Self {
        Self::Port(term)
    }
}
impl From<Arc<BinaryData>> for Term {
    fn from(term: Arc<BinaryData>) -> Self {
        Self::RcBinary(term)
    }
}
impl From<&'static BinaryData> for Term {
    fn from(term: &'static BinaryData) -> Self {
        Self::ConstantBinary(term)
    }
}
impl TryInto<bool> for Term {
    type Error = ();
    fn try_into(self) -> Result<bool, Self::Error> {
        match self {
            Self::Bool(b) => Ok(b),
            Self::Atom(a) if a.is_boolean() => Ok(a.as_boolean()),
            _ => Err(()),
        }
    }
}
impl TryInto<Atom> for Term {
    type Error = ();
    fn try_into(self) -> Result<Atom, Self::Error> {
        match self {
            Self::Atom(a) => Ok(a),
            Self::Bool(b) => Ok(b.into()),
            _ => Err(()),
        }
    }
}
impl TryInto<char> for Term {
    type Error = ();
    fn try_into(self) -> Result<char, Self::Error> {
        const MAX: i64 = char::MAX as u32 as i64;

        let i: i64 = self.try_into()?;

        if i >= 0 && i <= MAX {
            (i as u32).try_into().map_err(|_| ())
        } else {
            Err(())
        }
    }
}
impl TryInto<i64> for Term {
    type Error = ();
    #[inline]
    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            Self::Int(i) => Ok(i),
            Self::BigInt(i) => match i.to_i64() {
                Some(i) => Ok(i),
                None => Err(()),
            },
            _ => Err(()),
        }
    }
}
impl TryInto<Int> for Term {
    type Error = ();
    #[inline]
    fn try_into(self) -> Result<Int, Self::Error> {
        match self {
            Self::Int(i) => Ok(Int::Small(i)),
            Self::BigInt(i) => Ok(Int::Big((**i).clone())),
            _ => Err(()),
        }
    }
}
impl TryInto<Number> for Term {
    type Error = ();
    #[inline]
    fn try_into(self) -> Result<Number, Self::Error> {
        match self {
            Self::Int(i) => Ok(Number::Integer(Int::Small(i))),
            Self::BigInt(i) => Ok(Number::Integer(Int::Big((**i).clone()))),
            Self::Float(f) => Ok(Number::Float(f)),
            _ => Err(()),
        }
    }
}
impl TryInto<Gc<Cons>> for Term {
    type Error = ();
    fn try_into(self) -> Result<Gc<Cons>, Self::Error> {
        match self {
            Self::Cons(c) => Ok(c),
            _ => Err(()),
        }
    }
}
impl TryInto<Gc<Tuple>> for Term {
    type Error = ();
    fn try_into(self) -> Result<Gc<Tuple>, Self::Error> {
        match self {
            Self::Tuple(t) => Ok(t),
            _ => Err(()),
        }
    }
}
impl TryInto<Gc<BigInt>> for Term {
    type Error = ();
    fn try_into(self) -> Result<Gc<BigInt>, Self::Error> {
        match self {
            Self::BigInt(i) => Ok(i),
            _ => Err(()),
        }
    }
}
impl TryInto<f64> for Term {
    type Error = ();
    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            Self::Float(f) => Ok(f.into()),
            _ => Err(()),
        }
    }
}
impl TryInto<Float> for Term {
    type Error = ();
    fn try_into(self) -> Result<Float, Self::Error> {
        match self {
            Self::Float(f) => Ok(f),
            _ => Err(()),
        }
    }
}
// Support converting from atom terms to `Encoding` type
impl TryInto<Encoding> for Term {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Encoding, Self::Error> {
        match self {
            Self::Atom(a) => a.as_str().parse(),
            other => Err(anyhow!(
                "invalid encoding name: expected atom; got {}",
                &other
            )),
        }
    }
}
impl AsRef<Term> for Term {
    #[inline(always)]
    fn as_ref(&self) -> &Term {
        self
    }
}
impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => write!(f, "NONE"),
            Self::Catch(_) => write!(f, "CATCH"),
            Self::Code(_) => write!(f, "CP"),
            Self::Nil => f.write_str("[]"),
            Self::Bool(term) => write!(f, "{}", term),
            Self::Atom(term) => write!(f, "{}", term),
            Self::Int(term) => write!(f, "{}", term),
            Self::BigInt(term) => write!(f, "{}", term),
            Self::Float(term) => write!(f, "{}", term),
            Self::Cons(term) => write!(f, "{}", term),
            Self::Tuple(term) => write!(f, "{}", term),
            Self::Map(boxed) => write!(f, "{}", boxed),
            Self::Closure(boxed) => write!(f, "{}", boxed),
            Self::Pid(boxed) => write!(f, "{}", boxed),
            Self::Port(boxed) => write!(f, "{}", boxed),
            Self::Reference(boxed) => write!(f, "{}", boxed),
            Self::HeapBinary(boxed) => write!(f, "{}", boxed),
            Self::RcBinary(boxed) => write!(f, "{}", boxed),
            Self::RefBinary(boxed) => write!(f, "{}", boxed),
            Self::ConstantBinary(bytes) => write!(f, "{}", bytes),
        }
    }
}
impl Eq for Term {}
impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::None => other.is_none(),
            Self::Catch(x) => match other {
                Self::Catch(y) => x == y,
                _ => false,
            },
            Self::Code(x) => match other {
                Self::Code(y) => x == y,
                _ => false,
            },
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
                Self::BigInt(y) => match y.to_i64() {
                    Some(ref y) => x == y,
                    None => false,
                },
                Self::Float(y) => y == x,
                _ => false,
            },
            Self::BigInt(x) => match other {
                Self::Int(y) => match x.to_i64() {
                    Some(ref x) => x == y,
                    None => false,
                },
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
                Self::Cons(y) => x == y,
                _ => false,
            },
            Self::Tuple(x) => match other {
                Self::Tuple(y) => x == y,
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
            Self::HeapBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_ref().eq(y),
                Self::HeapBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RcBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RefBinary(y) => x.as_ref().eq(y.as_ref()),
                _ => false,
            },
            Self::RcBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_ref().eq(y),
                Self::HeapBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RcBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RefBinary(y) => x.as_ref().eq(y.as_ref()),
                _ => false,
            },
            Self::RefBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_ref().eq(y),
                Self::HeapBinary(y) => x.as_ref().eq(y),
                Self::RcBinary(y) => x.as_ref().eq(y),
                Self::RefBinary(y) => x.as_ref().eq(y),
                _ => false,
            },
            Self::ConstantBinary(x) => match other {
                Self::ConstantBinary(y) => x.eq(y),
                Self::HeapBinary(y) => x.as_bytes().eq(y.as_bytes()),
                Self::RcBinary(y) => x.as_bytes().eq(y.as_bytes()),
                Self::RefBinary(y) => y.as_ref().eq(x),
                _ => false,
            },
        }
    }
}
impl ExactEq for Term {
    fn exact_eq(&self, other: &Self) -> bool {
        match self {
            Self::None => other.is_none(),
            Self::Nil => other.is_nil(),
            Self::Catch(x) => match other {
                Self::Catch(y) => x == y,
                _ => false,
            },
            Self::Code(x) => match other {
                Self::Code(y) => x == y,
                _ => false,
            },
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
                Self::BigInt(y) => match y.to_i64() {
                    Some(ref y) => x == y,
                    None => false,
                },
                _ => false,
            },
            Self::BigInt(x) => match other {
                Self::Int(y) => match x.to_i64() {
                    Some(ref x) => x == y,
                    None => false,
                },
                Self::BigInt(y) => x == y,
                _ => false,
            },
            Self::Float(x) => match other {
                Self::Float(y) => x == y,
                _ => false,
            },
            Self::Cons(x) => match other {
                Self::Cons(y) => x.deref().exact_eq(y.deref()),
                _ => false,
            },
            Self::Tuple(x) => match other {
                Self::Tuple(y) => x.deref().exact_eq(y.deref()),
                _ => false,
            },
            Self::Map(x) => match other {
                Self::Map(y) => x.as_ref().exact_eq(y.as_ref()),
                _ => false,
            },
            Self::Closure(x) => match other {
                Self::Closure(y) => x.as_ref().exact_eq(y.as_ref()),
                _ => false,
            },
            Self::Pid(x) => match other {
                Self::Pid(y) => x.as_ref().exact_eq(y.as_ref()),
                _ => false,
            },
            Self::Port(x) => match other {
                Self::Port(y) => x.as_ref().exact_eq(y.as_ref()),
                _ => false,
            },
            Self::Reference(x) => match other {
                Self::Reference(y) => x.as_ref().exact_eq(y.as_ref()),
                _ => false,
            },
            Self::HeapBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_ref().eq(y),
                Self::HeapBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RcBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RefBinary(y) => x.as_ref().eq(y.as_ref()),
                _ => false,
            },
            Self::RcBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_ref().eq(y),
                Self::HeapBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RcBinary(y) => x.as_ref().eq(y.as_ref()),
                Self::RefBinary(y) => x.as_ref().eq(y.as_ref()),
                _ => false,
            },
            Self::RefBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_ref().eq(y),
                Self::HeapBinary(y) => x.as_ref().eq(y),
                Self::RcBinary(y) => x.as_ref().eq(y),
                Self::RefBinary(y) => x.as_ref().eq(y),
                _ => false,
            },
            Self::ConstantBinary(x) => match other {
                Self::ConstantBinary(y) => x.eq(y),
                Self::HeapBinary(y) => x.as_bytes().eq(y.as_bytes()),
                Self::RcBinary(y) => x.as_bytes().eq(y.as_bytes()),
                Self::RefBinary(y) => y.as_ref().eq(x),
                _ => false,
            },
        }
    }

    #[inline]
    fn exact_ne(&self, other: &Self) -> bool {
        !self.exact_eq(other)
    }
}
impl PartialEq<Atom> for Term {
    fn eq(&self, other: &Atom) -> bool {
        match self {
            Self::Atom(this) => this.eq(other),
            _ => false,
        }
    }
}
impl PartialEq<Term> for Atom {
    fn eq(&self, other: &Term) -> bool {
        match other {
            Term::Atom(atom) => self.eq(atom),
            _ => false,
        }
    }
}
impl PartialOrd for Term {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Term {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match self {
            // None and special variants are always ordered last
            Self::None => {
                if other.is_none() {
                    Ordering::Equal
                } else if other.is_special() {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            // Followed by catch/code
            Self::Catch(x) => match other {
                Self::None => Ordering::Greater,
                Self::Catch(y) => x.cmp(y),
                Self::Code(_) => Ordering::Less,
                _ => Ordering::Greater,
            },
            Self::Code(x) => match other {
                Self::None | Self::Catch(_) => Ordering::Greater,
                Self::Code(y) => x.cmp(y),
                _ => Ordering::Greater,
            },
            // Numbers are smaller than all other terms, using whichever type has the highest
            // precision. We need comparison order to preserve the ExactEq semantics, so
            // equality between integers/floats is broken by sorting floats first due to
            // their greater precision in most cases
            Self::Int(x) => match other {
                Self::None | Self::Catch(_) | Self::Code(_) => Ordering::Greater,
                Self::Int(y) => x.cmp(y),
                Self::BigInt(y) => match y.to_i64() {
                    Some(y) => x.cmp(&y),
                    None if y.sign() == Sign::Minus => Ordering::Greater,
                    None => Ordering::Less,
                },
                Self::Float(y) => match y.partial_cmp(x).unwrap().reverse() {
                    Ordering::Equal => Ordering::Greater,
                    other => other,
                },
                _ => Ordering::Less,
            },
            Self::BigInt(x) => match other {
                Self::None | Self::Catch(_) | Self::Code(_) => Ordering::Greater,
                Self::Int(y) => match x.to_i64() {
                    Some(x) => x.cmp(&y),
                    None if x.sign() == Sign::Minus => Ordering::Less,
                    None => Ordering::Greater,
                },
                Self::BigInt(y) => (&**x).cmp(&**y),
                Self::Float(y) => match y.partial_cmp(&**x).unwrap().reverse() {
                    Ordering::Equal => Ordering::Greater,
                    other => other,
                },
                _ => Ordering::Less,
            },
            Self::Float(x) => match other {
                Self::None | Self::Catch(_) | Self::Code(_) => Ordering::Greater,
                Self::Float(y) => x.partial_cmp(y).unwrap(),
                Self::Int(y) => match x.partial_cmp(y).unwrap() {
                    Ordering::Equal => Ordering::Less,
                    other => other,
                },
                Self::BigInt(y) => match x.partial_cmp(&**y).unwrap() {
                    Ordering::Equal => Ordering::Less,
                    other => other,
                },
                _ => Ordering::Less,
            },
            Self::Bool(x) => match other {
                Self::Bool(y) => x.cmp(y),
                Self::Atom(a) if a.is_boolean() => x.cmp(&a.as_boolean()),
                Self::Atom(_) => Ordering::Less,
                Self::None
                | Self::Catch(_)
                | Self::Code(_)
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Atom(x) => match other {
                Self::Atom(y) => x.cmp(y),
                Self::Bool(y) if x.is_boolean() => x.as_boolean().cmp(y),
                Self::Bool(_) => Ordering::Greater,
                Self::None
                | Self::Catch(_)
                | Self::Code(_)
                | Self::Int(_)
                | Self::BigInt(_)
                | Self::Float(_) => Ordering::Greater,
                _ => Ordering::Less,
            },
            Self::Reference(x) => match other {
                Self::Reference(y) => x.cmp(y),
                Self::None
                | Self::Catch(_)
                | Self::Code(_)
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
                | Self::Catch(_)
                | Self::Code(_)
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
                | Self::Catch(_)
                | Self::Code(_)
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
                | Self::Catch(_)
                | Self::Code(_)
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
                Self::Tuple(y) => x.cmp(y),
                Self::None
                | Self::Catch(_)
                | Self::Code(_)
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
                | Self::Catch(_)
                | Self::Code(_)
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
                | Self::Catch(_)
                | Self::Code(_)
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
                Self::Cons(y) => x.cmp(y),
                Self::None
                | Self::Catch(_)
                | Self::Code(_)
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
            Self::HeapBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_bytes().cmp(y.as_bytes()),
                Self::HeapBinary(y) => x.cmp(y),
                Self::RcBinary(y) => (&**x).partial_cmp(y).unwrap(),
                Self::RefBinary(y) => (&**x).partial_cmp(y).unwrap(),
                _ => Ordering::Greater,
            },
            Self::RcBinary(x) => match other {
                Self::ConstantBinary(y) => x.as_bytes().cmp(y.as_bytes()),
                Self::HeapBinary(y) => (&**x).partial_cmp(y).unwrap(),
                Self::RcBinary(y) => x.cmp(y),
                Self::RefBinary(y) => (&**x).partial_cmp(y).unwrap(),
                _ => Ordering::Greater,
            },
            Self::RefBinary(x) => match other {
                Self::ConstantBinary(y) => (&**x).partial_cmp(y).unwrap(),
                Self::HeapBinary(y) => (&**x).partial_cmp(y).unwrap(),
                Self::RcBinary(y) => (&**x).partial_cmp(y).unwrap(),
                Self::RefBinary(y) => x.cmp(y),
                _ => Ordering::Greater,
            },
            Self::ConstantBinary(x) => match other {
                Self::ConstantBinary(y) => x.cmp(y),
                Self::HeapBinary(y) => x.as_bytes().cmp(y.as_bytes()),
                Self::RcBinary(y) => x.as_bytes().cmp(y.as_bytes()),
                Self::RefBinary(y) => (&**y).partial_cmp(x).unwrap().reverse(),
                _ => Ordering::Greater,
            },
        }
    }
}
impl core::ops::Add for Term {
    type Output = Result<Number, InvalidArithmeticError>;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs: Number = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Number = rhs.try_into().map_err(|_| InvalidArithmeticError)?;
        (lhs + rhs).map_err(|_| InvalidArithmeticError)
    }
}
impl core::ops::Sub for Term {
    type Output = Result<Number, InvalidArithmeticError>;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs: Number = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Number = rhs.try_into().map_err(|_| InvalidArithmeticError)?;
        (lhs - rhs).map_err(|_| InvalidArithmeticError)
    }
}
impl core::ops::Mul for Term {
    type Output = Result<Number, InvalidArithmeticError>;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs: Number = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Number = rhs.try_into().map_err(|_| InvalidArithmeticError)?;
        (lhs * rhs).map_err(|_| InvalidArithmeticError)
    }
}
impl core::ops::Div for Term {
    type Output = Result<Result<Number, DivisionError>, InvalidArithmeticError>;

    fn div(self, rhs: Self) -> Self::Output {
        let lhs: Number = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Number = rhs.try_into().map_err(|_| InvalidArithmeticError)?;

        match (lhs, rhs) {
            (Number::Integer(lhs), Number::Integer(rhs)) => Ok((lhs / rhs).map(Number::Integer)),
            (Number::Float(lhs), Number::Float(rhs)) => Ok((lhs / rhs).map(Number::Float)),
            (Number::Float(lhs), Number::Integer(rhs)) => Ok((lhs / rhs).map(Number::Float)),
            _ => Err(InvalidArithmeticError),
        }
    }
}
impl core::ops::Rem for Term {
    type Output = Result<Result<Int, DivisionError>, InvalidArithmeticError>;

    fn rem(self, rhs: Self) -> Self::Output {
        let lhs: Int = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Int = rhs.try_into().map_err(|_| InvalidArithmeticError)?;

        Ok(lhs % rhs)
    }
}
impl core::ops::Neg for Term {
    type Output = Result<Number, InvalidArithmeticError>;

    fn neg(self) -> Self::Output {
        let lhs: Number = self.try_into().map_err(|_| InvalidArithmeticError)?;
        Ok(-lhs)
    }
}

impl core::ops::Shl for Term {
    type Output = Result<Int, InvalidArithmeticError>;

    fn shl(self, rhs: Self) -> Self::Output {
        let lhs: Int = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Int = rhs.try_into().map_err(|_| InvalidArithmeticError)?;

        Ok(lhs << rhs)
    }
}
impl core::ops::Shr for Term {
    type Output = Result<Int, InvalidArithmeticError>;

    fn shr(self, rhs: Self) -> Self::Output {
        let lhs: Int = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Int = rhs.try_into().map_err(|_| InvalidArithmeticError)?;

        Ok(lhs >> rhs)
    }
}
impl core::ops::BitAnd for Term {
    type Output = Result<Int, InvalidArithmeticError>;

    fn bitand(self, rhs: Self) -> Self::Output {
        let lhs: Int = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Int = rhs.try_into().map_err(|_| InvalidArithmeticError)?;

        Ok(lhs & rhs)
    }
}
impl core::ops::BitOr for Term {
    type Output = Result<Int, InvalidArithmeticError>;

    fn bitor(self, rhs: Self) -> Self::Output {
        let lhs: Int = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Int = rhs.try_into().map_err(|_| InvalidArithmeticError)?;

        Ok(lhs | rhs)
    }
}
impl core::ops::BitXor for Term {
    type Output = Result<Int, InvalidArithmeticError>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let lhs: Int = self.try_into().map_err(|_| InvalidArithmeticError)?;
        let rhs: Int = rhs.try_into().map_err(|_| InvalidArithmeticError)?;

        Ok(lhs ^ rhs)
    }
}

impl From<time::TimeUnit> for Term {
    fn from(unit: time::TimeUnit) -> Self {
        use firefly_system::time::TimeUnit;

        match unit {
            TimeUnit::Hertz(hertz) => Self::Int(hertz.get().try_into().unwrap()),
            TimeUnit::Second => Self::Atom(atoms::Second),
            TimeUnit::Millisecond => Self::Atom(atoms::Millisecond),
            TimeUnit::Microsecond => Self::Atom(atoms::Microsecond),
            TimeUnit::Nanosecond => Self::Atom(atoms::Nanosecond),
            TimeUnit::Native => Self::Atom(atoms::Native),
            TimeUnit::PerformanceCounter => Self::Atom(atoms::PerfCounter),
        }
    }
}
impl TryInto<time::TimeUnit> for Term {
    type Error = time::TimeUnitConversionError;

    fn try_into(self) -> Result<time::TimeUnit, Self::Error> {
        use firefly_system::time::TimeUnitConversionError;

        match self {
            Self::Int(i) => i.try_into(),
            Self::BigInt(big) => {
                let hertz = big
                    .to_usize()
                    .ok_or(TimeUnitConversionError::InvalidHertzValue)?;
                hertz.try_into()
            }
            Self::Atom(atom) => atom.as_str().parse(),
            _ => Err(TimeUnitConversionError::InvalidConversion),
        }
    }
}
