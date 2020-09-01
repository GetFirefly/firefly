use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::mem;
use std::sync::Arc;

use anyhow::*;
use thiserror::Error;

use crate::borrow::CloneToProcess;
use crate::erts::exception::ErlangException;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::trace::Trace;
use crate::erts::term::prelude::*;

use super::ArcError;

#[derive(Error, Clone)]
pub struct Throw {
    reason: Term,
    stacktrace: Arc<Trace>,
    source: Option<ArcError>,
}
impl Throw {
    pub fn new(reason: Term, stacktrace: Arc<Trace>, source: Option<ArcError>) -> Self {
        Self {
            reason,
            stacktrace,
            source,
        }
    }
    pub fn as_error_tuple<A>(&self, heap: &mut A) -> super::AllocResult<Term>
    where
        A: TermAlloc,
    {
        let class = Atom::THROW.as_term();
        // NOTE: The trace is allocated in a fragment, so this is a single word always
        let trace = self.stacktrace.as_term()?;
        let tuple = if self.reason.is_immediate() {
            let mut tuple = heap.mut_tuple(3)?;
            tuple.set_element(0, class).unwrap();
            tuple.set_element(1, self.reason).unwrap();
            tuple.set_element(2, trace).unwrap();
            tuple
        } else {
            let reason = self.reason.clone_to_heap(heap)?;
            let mut tuple =
                heap.mut_tuple(1 + self.reason.size_in_words() * mem::size_of::<Term>())?;
            tuple.set_element(0, class).unwrap();
            tuple.set_element(1, reason).unwrap();
            tuple.set_element(2, trace).unwrap();
            tuple
        };
        Ok(tuple.into())
    }
    pub fn as_erlang_exception(&self) -> Box<ErlangException> {
        let class = Atom::THROW.as_term();
        ErlangException::new(class, self.reason, self.stacktrace.clone())
    }
    pub fn class(&self) -> Class {
        Class::Throw
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Arc<Trace> {
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
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Throw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.stacktrace
            .format(
                f,
                Atom::str_to_term("throw"),
                self.reason,
                self.source.clone(),
            )
            .map_err(|_| fmt::Error)
    }
}

#[derive(Error, Clone)]
pub struct Error {
    reason: Term,
    arguments: Option<Term>,
    stacktrace: Arc<Trace>,
    source: Option<ArcError>,
}
impl Error {
    pub fn new(
        reason: Term,
        arguments: Option<Term>,
        stacktrace: Arc<Trace>,
        source: Option<ArcError>,
    ) -> Self {
        Self {
            reason,
            arguments,
            stacktrace,
            source,
        }
    }
    pub fn as_error_tuple<A>(&self, heap: &mut A) -> super::AllocResult<Term>
    where
        A: TermAlloc,
    {
        let class = Atom::ERROR.as_term();
        // NOTE: The trace is allocated in a fragment, so this is a single word always
        let trace = self.stacktrace.as_term()?;
        let tuple = if self.reason.is_immediate() {
            let mut tuple = heap.mut_tuple(3)?;
            tuple.set_element(0, class).unwrap();
            tuple.set_element(1, self.reason).unwrap();
            tuple.set_element(2, trace).unwrap();
            tuple
        } else {
            let reason = self.reason.clone_to_heap(heap)?;
            let mut tuple =
                heap.mut_tuple(1 + self.reason.size_in_words() * mem::size_of::<Term>())?;
            tuple.set_element(0, class).unwrap();
            tuple.set_element(1, reason).unwrap();
            tuple.set_element(2, trace).unwrap();
            tuple
        };
        Ok(tuple.into())
    }
    pub fn as_erlang_exception(&self) -> Box<ErlangException> {
        let class = Atom::ERROR.as_term();
        ErlangException::new(class, self.reason, self.stacktrace.clone())
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
    pub fn stacktrace(&self) -> Arc<Trace> {
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
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.stacktrace
            .format(
                f,
                Atom::str_to_term("error"),
                self.reason,
                self.source.clone(),
            )
            .map_err(|_| fmt::Error)
    }
}

#[derive(Error, Clone)]
pub struct Exit {
    reason: Term,
    stacktrace: Arc<Trace>,
    source: Option<ArcError>,
}
impl Exit {
    pub fn new(reason: Term, stacktrace: Arc<Trace>, source: Option<ArcError>) -> Self {
        Self {
            reason,
            stacktrace,
            source,
        }
    }
    pub fn as_error_tuple<A>(&self, heap: &mut A) -> super::AllocResult<Term>
    where
        A: TermAlloc,
    {
        let class = Atom::EXIT.as_term();
        // NOTE: The trace is allocated in a fragment, so this is a single word always
        let trace = self.stacktrace.as_term()?;
        let tuple = if self.reason.is_immediate() {
            let mut tuple = heap.mut_tuple(3)?;
            tuple.set_element(0, class).unwrap();
            tuple.set_element(1, self.reason).unwrap();
            tuple.set_element(2, trace).unwrap();
            tuple
        } else {
            let reason = self.reason.clone_to_heap(heap)?;
            let mut tuple =
                heap.mut_tuple(1 + self.reason.size_in_words() * mem::size_of::<Term>())?;
            tuple.set_element(0, class).unwrap();
            tuple.set_element(1, reason).unwrap();
            tuple.set_element(2, trace).unwrap();
            tuple
        };
        Ok(tuple.into())
    }
    pub fn as_erlang_exception(&self) -> Box<ErlangException> {
        let class = Atom::EXIT.as_term();
        ErlangException::new(class, self.reason, self.stacktrace.clone())
    }
    pub fn class(&self) -> Class {
        Class::Exit
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Arc<Trace> {
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
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Exit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.stacktrace
            .format(
                f,
                Atom::str_to_term("exit"),
                self.reason,
                self.source.clone(),
            )
            .map_err(|_| fmt::Error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Error { arguments: Option<Term> },
    Exit,
    Throw,
}
impl Class {
    pub fn as_atom(&self) -> Atom {
        match self {
            Self::Error { .. } => Atom::ERROR,
            Self::Exit => Atom::EXIT,
            Self::Throw => Atom::THROW,
        }
    }

    pub fn as_term(&self) -> Term {
        self.as_atom().as_term()
    }
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
