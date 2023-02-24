use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use firefly_binary::BitVec;
use firefly_diagnostics::{SourceSpan, Spanned};
use firefly_intern::Symbol;
use firefly_number::{Float, Int};

use crate::*;

#[derive(Debug, Clone, Spanned)]
pub struct Literal {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub value: Lit,
}
annotated!(Literal);
impl Literal {
    pub fn atom(span: SourceSpan, sym: Symbol) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Atom(sym),
        }
    }

    pub fn integer<I: Into<Int>>(span: SourceSpan, i: I) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Integer(i.into()),
        }
    }

    pub fn float<F: Into<Float>>(span: SourceSpan, f: F) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Float(f.into()),
        }
    }

    pub fn nil(span: SourceSpan) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Nil,
        }
    }

    pub fn cons(span: SourceSpan, head: Literal, tail: Literal) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Cons(Box::new(head), Box::new(tail)),
        }
    }

    pub fn tuple(span: SourceSpan, elements: Vec<Literal>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Tuple(elements),
        }
    }

    pub fn map(span: SourceSpan, mut elements: Vec<(Literal, Literal)>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Map(elements.drain(..).collect()),
        }
    }

    pub fn binary(span: SourceSpan, data: BitVec) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Binary(data),
        }
    }

    pub fn is_integer(&self) -> bool {
        match self.value {
            Lit::Integer(_) => true,
            _ => false,
        }
    }

    pub fn as_integer(&self) -> Option<&Int> {
        match &self.value {
            Lit::Integer(ref i) => Some(i),
            _ => None,
        }
    }

    pub fn as_atom(&self) -> Option<Symbol> {
        match &self.value {
            Lit::Atom(a) => Some(*a),
            _ => None,
        }
    }
}
impl PartialEq for Literal {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl Eq for Literal {}
impl Hash for Literal {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}
impl PartialOrd for Literal {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
impl Ord for Literal {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Lit {
    Atom(Symbol),
    Integer(Int),
    Float(Float),
    Nil,
    Cons(Box<Literal>, Box<Literal>),
    Tuple(Vec<Literal>),
    Map(BTreeMap<Literal, Literal>),
    Binary(BitVec),
}
impl fmt::Debug for Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        crate::printing::print_lit(f, self)
    }
}
impl Lit {
    pub fn is_number(&self) -> bool {
        match self {
            Self::Integer(_) | Self::Float(_) => true,
            _ => false,
        }
    }
}
impl Hash for Lit {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Atom(x) => x.hash(state),
            Self::Float(f) => f.hash(state),
            Self::Integer(i) => i.hash(state),
            Self::Nil => (),
            Self::Cons(h, t) => {
                h.hash(state);
                t.hash(state);
            }
            Self::Tuple(elements) => Hash::hash_slice(elements.as_slice(), state),
            Self::Map(map) => map.hash(state),
            Self::Binary(bin) => bin.hash(state),
        }
    }
}
impl PartialOrd for Lit {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Lit {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Float(x), Self::Float(y)) => x.cmp(y),
            (Self::Float(x), Self::Integer(y)) => x.partial_cmp(y).unwrap(),
            (Self::Float(_), _) => Ordering::Less,
            (Self::Integer(x), Self::Integer(y)) => x.cmp(y),
            (Self::Integer(x), Self::Float(y)) => x.partial_cmp(y).unwrap(),
            (Self::Integer(_), _) => Ordering::Less,
            (Self::Atom(_), Self::Float(_)) | (Self::Atom(_), Self::Integer(_)) => {
                Ordering::Greater
            }
            (Self::Atom(x), Self::Atom(y)) => x.cmp(y),
            (Self::Atom(_), _) => Ordering::Less,
            (Self::Tuple(_), Self::Float(_))
            | (Self::Tuple(_), Self::Integer(_))
            | (Self::Tuple(_), Self::Atom(_)) => Ordering::Greater,
            (Self::Tuple(xs), Self::Tuple(ys)) => xs.cmp(ys),
            (Self::Tuple(_), _) => Ordering::Less,
            (Self::Map(_), Self::Float(_))
            | (Self::Map(_), Self::Integer(_))
            | (Self::Map(_), Self::Atom(_))
            | (Self::Map(_), Self::Tuple(_)) => Ordering::Greater,
            (Self::Map(x), Self::Map(y)) => x.cmp(y),
            (Self::Map(_), _) => Ordering::Less,
            (Self::Nil, Self::Nil) => Ordering::Equal,
            (Self::Nil, Self::Cons(_, _)) => Ordering::Less,
            (Self::Nil, _) => Ordering::Greater,
            (Self::Cons(h1, t1), Self::Cons(h2, t2)) => match h1.cmp(&h2) {
                Ordering::Equal => t1.cmp(&t2),
                other => other,
            },
            (Self::Cons(_, _), Self::Binary(_)) => Ordering::Less,
            (Self::Cons(_, _), _) => Ordering::Greater,
            (Self::Binary(x), Self::Binary(y)) => x.cmp(y),
            (Self::Binary(_), _) => Ordering::Greater,
        }
    }
}
