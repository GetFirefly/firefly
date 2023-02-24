use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};

use firefly_binary::{BitVec, Bitstring};
use firefly_intern::{symbols, Symbol};
use firefly_number::{Float, Int};
use firefly_syntax_base::{PrimitiveType, TermType, Type};

use cranelift_entity::entity_impl;

/// A handle that references a constant in the current context
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Constant(u32);
entity_impl!(Constant, "const");
impl Constant {
    fn new(value: u32) -> Self {
        Self(value)
    }
}

/// Represents the possible values encodable as immediate terms
#[derive(Debug, Copy, Clone)]
pub enum ImmediateTerm {
    Bool(bool),
    Atom(Symbol),
    Integer(i64),
    Float(f64),
    Nil,
    None,
}
impl ImmediateTerm {
    pub fn ty(&self) -> Type {
        match self {
            Self::Bool(_) => Type::Term(TermType::Bool),
            Self::Atom(_) => Type::Term(TermType::Atom),
            Self::Integer(_) => Type::Term(TermType::Integer),
            Self::Float(_) => Type::Term(TermType::Float),
            Self::Nil => Type::Term(TermType::Nil),
            Self::None => Type::Term(TermType::Any),
        }
    }
}
impl fmt::Display for ImmediateTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{}", b),
            Self::Atom(a) => write!(f, "{}", a.as_interned_str()),
            Self::Integer(i) => write!(f, "{}", i),
            Self::Float(flt) => write!(f, "{}", flt),
            Self::Nil => write!(f, "[]"),
            Self::None => write!(f, "none"),
        }
    }
}
impl Hash for ImmediateTerm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let d = std::mem::discriminant(self);
        d.hash(state);
        match self {
            Self::Bool(b) => b.hash(state),
            Self::Atom(a) => a.hash(state),
            Self::Integer(i) => i.hash(state),
            Self::Float(f) => {
                let bytes = f.to_be_bytes();
                bytes.hash(state)
            }
            Self::Nil | Self::None => (),
        }
    }
}
impl Eq for ImmediateTerm {}
impl PartialEq for ImmediateTerm {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::None, Self::None) => true,
            (Self::Nil, Self::Nil) => true,
            (Self::Integer(x), Self::Integer(y)) => x == y,
            (Self::Integer(x), Self::Float(y)) => (x as f64) == y,
            (Self::Float(x), Self::Float(y)) => x == y,
            (Self::Float(x), Self::Integer(y)) => x == (y as f64),
            (Self::Bool(x), Self::Bool(y)) => x == y,
            (Self::Atom(x), Self::Atom(y)) => x.as_interned_str() == y.as_interned_str(),
            (Self::Bool(x), Self::Atom(y)) => {
                if y == symbols::True {
                    x == true
                } else if y == symbols::False {
                    x == false
                } else {
                    false
                }
            }
            (Self::Atom(x), Self::Bool(y)) => {
                if x == symbols::True {
                    y == true
                } else if x == symbols::False {
                    y == false
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}
impl PartialOrd for ImmediateTerm {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (Self::None, _) => Some(Ordering::Less),
            (_, Self::None) => Some(Ordering::Greater),
            (Self::Integer(x), Self::Integer(y)) => x.partial_cmp(y),
            (Self::Integer(x), Self::Float(y)) => (*x as f64).partial_cmp(y),
            (Self::Integer(_), _) => Some(Ordering::Less),
            (Self::Float(x), Self::Integer(y)) => x.partial_cmp(&(*y as f64)),
            (_, Self::Integer(_)) => Some(Ordering::Greater),
            (Self::Float(x), Self::Float(y)) => x.partial_cmp(y),
            (Self::Float(_), _) => Some(Ordering::Less),
            (_, Self::Float(_)) => Some(Ordering::Greater),
            (Self::Bool(x), Self::Bool(y)) => x.partial_cmp(y),
            (Self::Bool(x), Self::Atom(y)) => {
                let x = if *x { symbols::True } else { symbols::False };
                let xstr = x.as_interned_str();
                let ystr = y.as_interned_str();
                xstr.partial_cmp(&ystr)
            }
            (Self::Atom(x), Self::Bool(y)) => {
                let y = if *y { symbols::True } else { symbols::False };
                let xstr = x.as_interned_str();
                let ystr = y.as_interned_str();
                xstr.partial_cmp(&ystr)
            }
            (Self::Atom(x), Self::Atom(y)) => {
                let xstr = x.as_interned_str();
                let ystr = y.as_interned_str();
                xstr.partial_cmp(&ystr)
            }
            (Self::Nil, _) => Some(Ordering::Greater),
            (_, Self::Nil) => Some(Ordering::Less),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Immediate {
    Term(ImmediateTerm),
    I1(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Isize(isize),
    F64(f64),
}
impl Immediate {
    pub fn ty(&self) -> Type {
        match self {
            Self::Term(t) => t.ty(),
            Self::I1(_) => Type::Primitive(PrimitiveType::I1),
            Self::I8(_) => Type::Primitive(PrimitiveType::I8),
            Self::I16(_) => Type::Primitive(PrimitiveType::I16),
            Self::I32(_) => Type::Primitive(PrimitiveType::I32),
            Self::I64(_) => Type::Primitive(PrimitiveType::I64),
            Self::Isize(_) => Type::Primitive(PrimitiveType::Isize),
            Self::F64(_) => Type::Primitive(PrimitiveType::F64),
        }
    }

    pub fn is_primitive(&self) -> bool {
        match self {
            Self::Term(_) => false,
            _ => true,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Term(ImmediateTerm::Integer(i)) => Some(*i),
            Self::Term(_) => None,
            Self::I1(b) => Some(*b as i64),
            Self::I8(i) => Some(*i as i64),
            Self::I16(i) => Some(*i as i64),
            Self::I32(i) => Some(*i as i64),
            Self::I64(i) => Some(*i),
            Self::Isize(i) => Some(*i as i64),
            Self::F64(_) => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Term(ImmediateTerm::Float(f)) | Self::F64(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Term(ImmediateTerm::Bool(b)) | Self::I1(b) => Some(*b),
            _ => None,
        }
    }

    pub fn rank(&self) -> u8 {
        match self {
            Self::Term(_) => 0,
            Self::I1(_) => 1,
            Self::I8(_) => 2,
            Self::I16(_) => 3,
            Self::I32(_) => 4,
            Self::I64(_) => 5,
            Self::Isize(_) => 6,
            Self::F64(_) => 7,
        }
    }
}
impl fmt::Display for Immediate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Term(t) => write!(f, "{}", t),
            Self::I1(i) => write!(f, "{}", i),
            Self::I8(i) => write!(f, "{}", i),
            Self::I16(i) => write!(f, "{}", i),
            Self::I32(i) => write!(f, "{}", i),
            Self::I64(i) => write!(f, "{}", i),
            Self::Isize(i) => write!(f, "{}", i),
            Self::F64(n) => write!(f, "{}", n),
        }
    }
}
impl Hash for Immediate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let d = std::mem::discriminant(self);
        d.hash(state);
        match self {
            Self::Term(t) => t.hash(state),
            Self::I1(i) => i.hash(state),
            Self::I8(i) => i.hash(state),
            Self::I16(i) => i.hash(state),
            Self::I32(i) => i.hash(state),
            Self::I64(i) => i.hash(state),
            Self::Isize(i) => i.hash(state),
            Self::F64(f) => {
                let bytes = f.to_be_bytes();
                bytes.hash(state)
            }
        }
    }
}
impl Eq for Immediate {}
impl PartialEq for Immediate {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Term(x), Self::Term(y)) => x == y,
            (Self::Term(_), _) => false,
            (_, Self::Term(_)) => false,
            (Self::I8(x), Self::I8(y)) => x == y,
            (Self::I16(x), Self::I16(y)) => x == y,
            (Self::I32(x), Self::I32(y)) => x == y,
            (Self::I64(x), Self::I64(y)) => x == y,
            (Self::Isize(x), Self::Isize(y)) => x == y,
            (Self::F64(x), Self::F64(y)) => x == y,
            _ => false,
        }
    }
}
impl PartialOrd for Immediate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (Self::Term(x), Self::Term(y)) => x.partial_cmp(y),
            (Self::F64(x), Self::F64(y)) => x.partial_cmp(y),
            (Self::F64(x), y) => {
                let y = y.as_i64().unwrap() as f64;
                match x.total_cmp(&y) {
                    Ordering::Equal => self.rank().partial_cmp(&other.rank()),
                    ord => Some(ord),
                }
            }
            (x, Self::F64(y)) => {
                let x = x.as_i64().unwrap() as f64;
                match x.total_cmp(&y) {
                    Ordering::Equal => self.rank().partial_cmp(&other.rank()),
                    ord => Some(ord),
                }
            }
            (x, y) => {
                let x = x.as_i64().unwrap();
                let y = y.as_i64().unwrap();
                match x.cmp(&y) {
                    Ordering::Equal => self.rank().partial_cmp(&other.rank()),
                    ord => Some(ord),
                }
            }
        }
    }
}
impl From<bool> for Immediate {
    #[inline(always)]
    fn from(value: bool) -> Self {
        Self::I1(value)
    }
}
impl From<i8> for Immediate {
    #[inline(always)]
    fn from(value: i8) -> Self {
        Self::I8(value)
    }
}
impl From<i16> for Immediate {
    #[inline(always)]
    fn from(value: i16) -> Self {
        Self::I16(value)
    }
}
impl From<i32> for Immediate {
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}
impl From<i64> for Immediate {
    #[inline(always)]
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}
impl From<isize> for Immediate {
    #[inline(always)]
    fn from(value: isize) -> Self {
        Self::Isize(value)
    }
}
impl From<f64> for Immediate {
    #[inline(always)]
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}
impl From<char> for Immediate {
    #[inline(always)]
    fn from(value: char) -> Self {
        Self::Term(ImmediateTerm::Integer(value as i64))
    }
}
impl From<Symbol> for Immediate {
    #[inline(always)]
    fn from(value: Symbol) -> Self {
        Self::Term(ImmediateTerm::Atom(value))
    }
}
impl From<Float> for Immediate {
    #[inline(always)]
    fn from(value: Float) -> Self {
        Self::Term(ImmediateTerm::Float(value.inner()))
    }
}

/// A trait that represents a value which can be represented as a vector of bytes
pub trait IntoBytes {
    fn into_bytes(self) -> Vec<u8>;
}
impl IntoBytes for u8 {
    fn into_bytes(self) -> Vec<u8> {
        vec![self]
    }
}
impl IntoBytes for i8 {
    fn into_bytes(self) -> Vec<u8> {
        vec![self as u8]
    }
}
impl IntoBytes for u16 {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for i16 {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for u32 {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for i32 {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for u64 {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for i64 {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for usize {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for isize {
    fn into_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl IntoBytes for Vec<u8> {
    fn into_bytes(self) -> Vec<u8> {
        self
    }
}
impl IntoBytes for &str {
    fn into_bytes(self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

/// Represents a raw constant value as a vector of bytes
///
/// It is expected that metadata about the constant is tracked elsewhere
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstantData(Vec<u8>);
impl<I: IntoBytes> From<I> for ConstantData {
    fn from(item: I) -> Self {
        Self(item.into_bytes())
    }
}
impl ConstantData {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.0
    }

    pub fn iter(&self) -> core::slice::Iter<u8> {
        self.0.iter()
    }

    pub fn append(mut self, bytes: impl IntoBytes) -> Self {
        let mut to_add = bytes.into_bytes();
        self.0.append(&mut to_add);
        self
    }

    // Zero-extend the data by appending zeros to the high-order byte slots
    pub fn zext(mut self, width: usize) -> Self {
        assert!(
            self.len() <= width,
            "constant data is already larger than {} bytes",
            width
        );
        self.0.resize(width, 0);
        self
    }
}
impl fmt::Display for ConstantData {
    /// Print the constant data in hexadecimal format, in big-endian order.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bytes = self.0.as_slice();
        if !bytes.is_empty() {
            write!(f, "0x")?;
            for b in bytes.iter().rev() {
                write!(f, "{:02x}", b)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ConstantItem {
    Integer(Int),
    Float(f64),
    Bool(bool),
    Atom(Symbol),
    Bytes(ConstantData),
    Bitstring(BitVec),
    String(String),
    InternedStr(Symbol),
}
impl Eq for ConstantItem {}
impl PartialEq for ConstantItem {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(x), Self::Integer(y)) => x.eq(y),
            (Self::Integer(_), _) => false,
            (Self::Float(x), Self::Float(y)) => x.eq(y),
            (Self::Float(_), _) => false,
            (Self::Bool(x), Self::Bool(y)) => x.eq(y),
            (Self::Bool(_), _) => false,
            (Self::Atom(x), Self::Atom(y)) => x.eq(y),
            (Self::Atom(_), _) => false,
            (Self::Bitstring(x), other) => match other {
                Self::Bitstring(y) => x.eq(y),
                Self::Bytes(y) => x.eq(y.as_slice()),
                Self::String(y) => x.eq(y.as_bytes()),
                Self::InternedStr(y) => x.eq(y.as_str().get().as_bytes()),
                _ => false,
            },
            (Self::Bytes(x), other) => match other {
                Self::Bytes(y) => x.eq(y),
                Self::Bitstring(y) => y.eq(x.as_slice()),
                Self::String(y) => x.as_slice().eq(y.as_bytes()),
                Self::InternedStr(y) => x.as_slice().eq(y.as_str().get().as_bytes()),
                _ => false,
            },
            (Self::String(x), other) => match other {
                Self::Bytes(y) => x.as_bytes().eq(y.as_slice()),
                Self::Bitstring(y) => y.eq(x.as_bytes()),
                Self::String(y) => x.eq(y),
                Self::InternedStr(y) => x.as_str().eq(y.as_str().get()),
                _ => false,
            },
            (Self::InternedStr(x), other) => match other {
                Self::Bytes(y) => x.as_str().get().as_bytes().eq(y.as_slice()),
                Self::Bitstring(y) => y.eq(x.as_str().get().as_bytes()),
                Self::String(y) => x.as_str().get().eq(y.as_str()),
                Self::InternedStr(y) => x.eq(y),
                _ => false,
            },
        }
    }
}
impl Hash for ConstantItem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let d = std::mem::discriminant(self);
        d.hash(state);
        match self {
            Self::Integer(i) => i.hash(state),
            Self::Float(f) => {
                let bytes = f.to_be_bytes();
                bytes.hash(state)
            }
            Self::Bool(b) => b.hash(state),
            Self::Atom(a) => a.hash(state),
            Self::Bytes(b) => b.as_slice().hash(state),
            Self::Bitstring(b) => b.hash(state),
            Self::String(b) => b.as_bytes().hash(state),
            Self::InternedStr(b) => b.as_str().get().as_bytes().hash(state),
        }
    }
}
impl ConstantItem {
    pub fn ty(&self) -> Type {
        match self {
            Self::Integer(_) => Type::Term(TermType::Integer),
            Self::Float(_) => Type::Term(TermType::Float),
            Self::Bool(_) => Type::Term(TermType::Bool),
            Self::Atom(_) => Type::Term(TermType::Atom),
            Self::Bitstring(_) | Self::Bytes(_) | Self::String(_) | Self::InternedStr(_) => {
                Type::Term(TermType::Bitstring)
            }
        }
    }

    fn byte_size(&self) -> usize {
        match self {
            Self::Atom(_) | Self::Bool(_) | Self::Float(_) | Self::Integer(Int::Small(_)) => 8,
            Self::Integer(Int::Big(b)) => {
                let bytes = b.to_signed_bytes_le();
                bytes.len()
            }
            Self::Bytes(b) => b.len(),
            Self::Bitstring(b) => b.byte_size(),
            Self::String(b) => b.as_bytes().len(),
            Self::InternedStr(b) => b.as_str().get().as_bytes().len(),
        }
    }
}
impl fmt::Display for ConstantItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Integer(i) => write!(f, "{}", i),
            Self::Float(flt) => write!(f, "{}", flt),
            Self::Bool(b) => write!(f, "{}", b),
            Self::Atom(s) => write!(f, "{}", s.as_interned_str()),
            Self::Bytes(bytes) => {
                if !bytes.is_empty() {
                    write!(f, "0x")?;
                    for b in bytes.iter().rev() {
                        write!(f, "{:02x}", b)?;
                    }
                    Ok(())
                } else {
                    write!(f, "0x0")
                }
            }
            Self::Bitstring(b) => write!(f, "{}", b.display()),
            Self::String(s) => {
                write!(f, "\"")?;
                for c in s.escape_debug() {
                    write!(f, "{}", c)?;
                }
                write!(f, "\"")
            }
            Self::InternedStr(b) => {
                let s = b.as_str().get();
                write!(f, "\"")?;
                for c in s.escape_debug() {
                    write!(f, "{}", c)?;
                }
                write!(f, "\"")
            }
        }
    }
}

/// A pool of constants.
///
/// The pool associates handles to constant data, and allows looking up
/// data via the handle, as well as looking up handles from data.
///
/// The pool de-duplicates data so all constants that have the same data
/// representation will share a handle
#[derive(Debug, Clone)]
pub struct ConstantPool {
    handles_to_values: BTreeMap<Constant, ConstantItem>,
    values_to_handles: HashMap<ConstantItem, Constant>,
}
impl ConstantPool {
    pub fn new() -> Self {
        Self {
            handles_to_values: BTreeMap::new(),
            values_to_handles: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.handles_to_values.clear();
        self.values_to_handles.clear();
    }

    pub fn insert(&mut self, constant: ConstantItem) -> Constant {
        if self.values_to_handles.contains_key(&constant) {
            *self.values_to_handles.get(&constant).unwrap()
        } else {
            let handle = Constant::new(self.len() as u32);
            self.set(handle, constant);
            handle
        }
    }

    pub fn get(&self, constant: Constant) -> &ConstantItem {
        self.handles_to_values.get(&constant).unwrap()
    }

    /// Link a constant handle to its value. This does not de-duplicate data but does avoid
    /// replacing any existing constant values. use `set` to tie a specific `const42` to its value;
    /// use `insert` to add a value and return the next available `const` entity.
    pub fn set(&mut self, constant_handle: Constant, constant_value: ConstantItem) {
        let replaced = self
            .handles_to_values
            .insert(constant_handle, constant_value.clone());
        assert!(
            replaced.is_none(),
            "attempted to overwrite an existing constant {:?}: {:?} => {:?}",
            constant_handle,
            &constant_value,
            replaced.unwrap()
        );
        self.values_to_handles
            .insert(constant_value, constant_handle);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Constant, &ConstantItem)> {
        self.handles_to_values.iter()
    }

    pub fn values(&self) -> impl Iterator<Item = &ConstantItem> {
        self.handles_to_values.values()
    }

    pub fn len(&self) -> usize {
        self.handles_to_values.len()
    }

    pub fn byte_size(&self) -> usize {
        self.values_to_handles.keys().map(|c| c.byte_size()).sum()
    }
}
