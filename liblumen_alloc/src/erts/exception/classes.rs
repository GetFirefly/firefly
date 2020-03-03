use core::convert::{TryFrom, TryInto};
use core::fmt;

use anyhow::*;
use thiserror::Error;

use crate::erts::term::prelude::*;

use super::ArcError;

#[derive(Error, Clone)]
pub struct Throw {
    reason: Term,
    stacktrace: Option<Term>,
    source: ArcError,
}
impl Throw {
    pub fn new(reason: Term, source: ArcError) -> Self {
        Self::new_with_trace(reason, None, source)
    }
    pub fn new_with_trace(reason: Term, trace: Option<Term>, source: ArcError) -> Self {
        Self {
            reason,
            stacktrace: trace,
            source,
        }
    }
    pub fn class(&self) -> Class {
        Class::Throw
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Option<Term> {
        self.stacktrace
    }
    pub fn source(&self) -> ArcError {
        self.source.clone()
    }
}
impl PartialEq for Throw {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason && self.stacktrace == other.stacktrace
    }
}
impl fmt::Debug for Throw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Throw")
            .field("reason", &self.reason.decode())
            .field("stacktrace", &self.stacktrace.map(|t| t.decode()))
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Throw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "** throw: {}\n", self.reason)?;

        if let Some(stacktrace) = self.stacktrace {
            write!(f, "{}", stacktrace)?;
        }

        // use Debug format for source, so that backtrace is included
        write!(f, "{:?}", self.source)
    }
}

#[derive(Error, Clone)]
pub struct Error {
    reason: Term,
    arguments: Option<Term>,
    stacktrace: Option<Term>,
    source: ArcError,
}
impl Error {
    pub fn new(reason: Term, arguments: Option<Term>, source: ArcError) -> Self {
        Self::new_with_trace(reason, arguments, None, source)
    }
    pub fn new_with_trace(
        reason: Term,
        arguments: Option<Term>,
        trace: Option<Term>,
        source: ArcError,
    ) -> Self {
        Self {
            reason,
            arguments,
            stacktrace: trace,
            source,
        }
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
    pub fn stacktrace(&self) -> Option<Term> {
        self.stacktrace
    }
    pub fn source(&self) -> ArcError {
        self.source.clone()
    }
}
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason && self.stacktrace == other.stacktrace
    }
}
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Error")
            .field("reason", &self.reason.decode())
            .field("arguments", &self.arguments.map(|t| t.decode()))
            .field("stacktrace", &self.stacktrace.map(|t| t.decode()))
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "** error: {}", self.reason)?;

        if let Some(stacktrace) = self.stacktrace {
            write!(f, "{}", stacktrace)?;
        }

        // use Debug format for source, so that backtrace is included
        write!(f, "{:?}", self.source)
    }
}

#[derive(Error, Clone)]
pub struct Exit {
    reason: Term,
    stacktrace: Option<Term>,
    source: ArcError,
}
impl Exit {
    pub fn new(reason: Term, source: ArcError) -> Self {
        Self::new_with_trace(reason, None, source)
    }
    pub fn new_with_trace(reason: Term, trace: Option<Term>, source: ArcError) -> Self {
        Self {
            reason,
            stacktrace: trace,
            source,
        }
    }
    pub fn class(&self) -> Class {
        Class::Exit
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Option<Term> {
        self.stacktrace
    }
    pub fn source(&self) -> ArcError {
        self.source.clone()
    }
}
impl PartialEq for Exit {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason && self.stacktrace == other.stacktrace
    }
}
impl fmt::Debug for Exit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Exit")
            .field("reason", &self.reason.decode())
            .field("stacktrace", &self.stacktrace.map(|t| t.decode()))
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for Exit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "** exit: {}\n", self.reason)?;

        if let Some(stacktrace) = self.stacktrace {
            write!(f, "{}", stacktrace)?;
        }

        // use Debug format for source, so that backtrace is included
        write!(f, "{:?}", self.source)
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
