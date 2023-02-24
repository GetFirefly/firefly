pub use firefly_beam::ast::*;

use firefly_beam::serialization::etf;
use firefly_beam::AbstractCode;
use firefly_diagnostics::{SourceSpan, Span, Spanned};
use firefly_intern::Symbol;

/// This is the root of a syntax tree parsed from Abstract Erlang forms
#[derive(Debug, Clone)]
pub struct Ast {
    pub forms: Vec<Form>,
}
impl From<AbstractCode> for Ast {
    fn from(code: AbstractCode) -> Self {
        Self { forms: code.forms }
    }
}

/// This is the root of a document containing a single term
#[derive(Debug, Clone, Spanned)]
pub struct Root {
    #[span]
    pub term: Term,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Spanned)]
pub enum Term {
    Atom(Span<Symbol>),
    String(Span<Symbol>),
    Char(Span<char>),
    Integer(Span<firefly_number::Int>),
    Float(Span<firefly_number::Float>),
    Tuple(Span<Vec<Term>>),
    Nil(SourceSpan),
    Cons(Span<(Box<Term>, Box<Term>)>),
    Map(Span<Vec<(Term, Term)>>),
}
impl Term {
    pub fn as_atom(self) -> Result<Span<Symbol>, Self> {
        match self {
            Self::Atom(s) => Ok(s),
            other => Err(other),
        }
    }

    pub fn as_string(self) -> Result<String, Self> {
        match self {
            Self::String(s) => Ok(s.as_str().get().to_string()),
            other => Err(other),
        }
    }

    pub fn as_string_symbol(self) -> Result<Span<Symbol>, Self> {
        match self {
            Self::String(s) => Ok(s),
            other => Err(other),
        }
    }

    pub fn as_list(self) -> Result<Span<Vec<Term>>, Self> {
        if !self.is_list() {
            return Err(self);
        }
        match self {
            Self::Nil(span) => Ok(Span::new(span, vec![])),
            Self::Cons(cons) => {
                let span = cons.span();
                let cons = cons.item;
                let mut elements = vec![*cons.0];
                let mut next = Some(*cons.1);
                while let Some(current) = next.take() {
                    match current {
                        Self::Nil(_) => break,
                        Self::Cons(cons) => {
                            let cons = cons.item;
                            elements.push(*cons.0);
                            next = Some(*cons.1);
                        }
                        _ => unreachable!(),
                    }
                }
                Ok(Span::new(span, elements))
            }
            _ => unreachable!(),
        }
    }

    pub fn as_tuple(self) -> Result<Span<Vec<Term>>, Self> {
        match self {
            Self::Tuple(tuple) => Ok(tuple),
            other => Err(other),
        }
    }

    pub fn is_list(&self) -> bool {
        match self {
            Self::Nil(_) => true,
            Self::Cons(cons) => cons.item.1.is_list(),
            _ => false,
        }
    }
}
impl Into<etf::Term> for Term {
    fn into(self) -> etf::Term {
        match self {
            Self::Atom(a) => etf::Term::Atom(etf::Atom { name: a.item }),
            Self::String(a) => etf::Term::String(etf::Str { value: a.item }),
            Self::Char(c) => etf::Term::Integer(c.item.into()),
            Self::Integer(i) => etf::Term::Integer(i.item.into()),
            Self::Float(f) => etf::Term::Float(f.item.into()),
            Self::Nil(_) => etf::Term::List(etf::List::nil()),
            Self::Tuple(mut t) => etf::Term::Tuple(
                t.item
                    .drain(..)
                    .map(|t| t.into())
                    .collect::<Vec<etf::Term>>()
                    .into(),
            ),
            Self::Map(mut m) => etf::Term::Map(
                m.item
                    .drain(..)
                    .map(|(k, v)| (k.into(), v.into()))
                    .collect::<Vec<(etf::Term, etf::Term)>>()
                    .into(),
            ),
            Self::Cons(cons) => {
                let cons = cons.item;
                let mut elements: Vec<etf::Term> = vec![(*cons.0).into()];
                let mut last: Option<Box<etf::Term>> = None;
                let mut next = Some(*cons.1);
                while let Some(current) = next.take() {
                    match current {
                        Self::Nil(_) => elements.push(etf::Term::List(etf::List::nil())),
                        Self::Cons(cons) => {
                            let cons = cons.item;
                            elements.push((*cons.0).into());
                            next = Some(*cons.1);
                        }
                        other => {
                            last = Some(Box::new(other.into()));
                            break;
                        }
                    }
                }
                match last {
                    None => etf::Term::List(etf::List { elements }),
                    Some(last) => etf::Term::ImproperList(etf::ImproperList { elements, last }),
                }
            }
        }
    }
}
