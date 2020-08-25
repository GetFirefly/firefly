use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use anyhow::*;
use thiserror::Error;

use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::trace::Trace;
use crate::erts::term::prelude::*;

use super::ArcError;

#[derive(Error, Clone)]
pub struct Throw {
    reason: Term,
    stacktrace: Option<Arc<Trace>>,
    source: Option<ArcError>,
}
impl Throw {
    pub fn new(reason: Term, source: ArcError) -> Self {
        Self {
            reason,
            stacktrace: None,
            source: Some(source),
        }
    }
    pub fn new_with_trace(reason: Term, trace: Arc<Trace>) -> Self {
        Self {
            reason,
            stacktrace: Some(trace),
            source: None,
        }
    }
    pub fn as_error_tuple<A>(&self, heap: &mut A) -> super::AllocResult<Term>
    where
        A: TermAlloc,
    {
        let class = Atom::THROW.as_term();
        let trace = if let Some(ref trace) = self.stacktrace {
            trace.as_term()?
        } else {
            Term::NIL
        };
        heap.tuple_from_slice(&[class, self.reason, trace])
            .map(|t| t.into())
    }
    pub fn class(&self) -> Class {
        Class::Throw
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Option<Arc<Trace>> {
        self.stacktrace.clone()
    }
    pub fn source(&self) -> Option<ArcError> {
        self.source.clone()
    }
}
impl PartialEq for Throw {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason
    }
}
impl fmt::Debug for Throw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Throw")
            .field("reason", &self.reason.decode())
            .field("has_stacktrace", &self.stacktrace.is_some())
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Throw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(ref trace) = self.stacktrace {
            trace
                .format(f, Atom::str_to_term("throw"), self.reason)
                .map_err(|_| fmt::Error)?;
        } else {
            // Fallback display
            write!(f, "** throw: {}\n", self.reason)?;

            // use Debug format for source, so that backtrace is included
            if let Some(ref source) = self.source {
                write!(f, "{:?}", source)?;
            }
        }

        Ok(())
    }
}

#[derive(Error, Clone)]
pub struct Error {
    reason: Term,
    arguments: Option<Term>,
    stacktrace: Option<Arc<Trace>>,
    source: Option<ArcError>,
}
impl Error {
    pub fn new(reason: Term, arguments: Option<Term>, source: ArcError) -> Self {
        Self {
            reason,
            arguments,
            stacktrace: None,
            source: Some(source),
        }
    }
    pub fn new_with_trace(reason: Term, arguments: Option<Term>, trace: Arc<Trace>) -> Self {
        Self {
            reason,
            arguments,
            stacktrace: Some(trace),
            source: None,
        }
    }
    pub fn as_error_tuple<A>(&self, heap: &mut A) -> super::AllocResult<Term>
    where
        A: TermAlloc,
    {
        let class = Atom::ERROR.as_term();
        let trace = if let Some(ref trace) = self.stacktrace {
            trace.as_term()?
        } else {
            Term::NIL
        };
        heap.tuple_from_slice(&[class, self.reason, trace])
            .map(|t| t.into())
    }
    pub fn arguments(&self) -> Option<Term> {
        self.arguments
    }
    pub fn class(&self) -> Class {
        Class::Error {
            arguments: self.arguments,
        }
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Option<Arc<Trace>> {
        self.stacktrace.clone()
    }
    pub fn source(&self) -> Option<ArcError> {
        self.source.clone()
    }
}
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason
    }
}
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Error")
            .field("reason", &self.reason.decode())
            .field("arguments", &self.arguments.map(|t| t.decode()))
            .field("has_stacktrace", &self.stacktrace.is_some())
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(ref trace) = self.stacktrace {
            trace
                .format(f, Atom::str_to_term("throw"), self.reason)
                .map_err(|_| fmt::Error)?;
        } else {
            // Fallback display
            write!(f, "** error: {}\n", self.reason)?;

            // use Debug format for source, so that backtrace is included
            if let Some(ref source) = self.source {
                write!(f, "{:?}", source)?;
            }
        }

        Ok(())
    }
}

#[derive(Error, Clone)]
pub struct Exit {
    reason: Term,
    stacktrace: Option<Arc<Trace>>,
    source: Option<ArcError>,
}
impl Exit {
    pub fn new(reason: Term, source: ArcError) -> Self {
        Self {
            reason,
            stacktrace: None,
            source: Some(source),
        }
    }
    pub fn new_with_trace(reason: Term, trace: Arc<Trace>) -> Self {
        Self {
            reason,
            stacktrace: Some(trace),
            source: None,
        }
    }
    pub fn as_error_tuple<A>(&self, heap: &mut A) -> super::AllocResult<Term>
    where
        A: TermAlloc,
    {
        let class = Atom::EXIT.as_term();
        let trace = if let Some(ref trace) = self.stacktrace {
            trace.as_term()?
        } else {
            Term::NIL
        };
        heap.tuple_from_slice(&[class, self.reason, trace])
            .map(|t| t.into())
    }
    pub fn class(&self) -> Class {
        Class::Exit
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Option<Arc<Trace>> {
        self.stacktrace.clone()
    }
    pub fn source(&self) -> Option<ArcError> {
        self.source.clone()
    }
}
impl PartialEq for Exit {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason
    }
}
impl fmt::Debug for Exit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Exit")
            .field("reason", &self.reason.decode())
            .field("has_stacktrace", &self.stacktrace.is_some())
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Exit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(ref trace) = self.stacktrace {
            trace
                .format(f, Atom::str_to_term("throw"), self.reason)
                .map_err(|_| fmt::Error)?;
        } else {
            // Fallback display
            write!(f, "** exit: {}\n", self.reason)?;

            // use Debug format for source, so that backtrace is included
            if let Some(ref source) = self.source {
                write!(f, "{:?}", source)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Error { arguments: Option<Term> },
    Exit,
    Throw,
}
impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Class::Error { .. } => f.write_str("error"),
            Class::Exit => f.write_str("exit"),
            Class::Throw => f.write_str("throw"),
        }
    }
}
impl TryFrom<Term> for Class {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> anyhow::Result<Class> {
        use self::Class::*;

        let atom: Atom = term
            .try_into()
            .with_context(|| format!("class ({}) is not an atom", term))?;

        match atom.name() {
            "error" => Ok(Error { arguments: None }),
            "exit" => Ok(Exit),
            "throw" => Ok(Throw),
            name => Err(TryAtomFromTermError(name))
                .context("supported exception classes are error, exit, or throw"),
        }
    }
}
