use core::convert::TryFrom;
use core::fmt;

use thiserror::Error;

use crate::erts::term::prelude::{Encoded, Term, TypedTerm};

use super::stacktrace::Stacktrace;
use super::Location;

#[derive(Error, Clone)]
#[error("** throw: {reason} at {location}")]
pub struct Throw {
    reason: Term,
    stacktrace: Option<Term>,
    location: Location,
}
impl Throw {
    pub fn new(reason: Term, location: Location) -> Self {
        Self::new_with_trace(reason, location, None)
    }
    pub fn new_with_trace(reason: Term, location: Location, trace: Option<Term>) -> Self {
        Self {
            reason,
            location,
            stacktrace: trace,
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
    pub fn location(&self) -> Location {
        self.location
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
            .field("location", &self.location)
            .finish()
    }
}

#[derive(Error, Clone)]
#[error("** error: {reason} at {location}")]
pub struct Error {
    reason: Term,
    arguments: Option<Term>,
    stacktrace: Option<Stacktrace>,
    location: Location,
}
impl Error {
    pub fn new(reason: Term, arguments: Option<Term>, location: Location) -> Self {
        Self::new_with_trace(reason, arguments, location, None)
    }
    pub fn new_with_trace(
        reason: Term,
        arguments: Option<Term>,
        location: Location,
        trace: Option<Stacktrace>,
    ) -> Self {
        Self {
            reason,
            arguments,
            location,
            stacktrace: trace,
        }
    }
    pub fn class(&self) -> Class {
        Class::Error {
            arguments: self.arguments,
        }
    }
    pub fn reason(&self) -> Term {
        self.reason
    }
    pub fn stacktrace(&self) -> Option<Stacktrace> {
        self.stacktrace.clone()
    }
    pub fn location(&self) -> Location {
        self.location
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
            .field("stacktrace", &self.stacktrace)
            .field("location", &self.location)
            .finish()
    }
}

#[derive(Error, Clone)]
#[error("** exit: {reason} at {location}")]
pub struct Exit {
    reason: Term,
    stacktrace: Option<Term>,
    location: Location,
}
impl Exit {
    pub fn new(reason: Term, location: Location) -> Self {
        Self::new_with_trace(reason, location, None)
    }
    pub fn new_with_trace(reason: Term, location: Location, trace: Option<Term>) -> Self {
        Self {
            reason,
            location,
            stacktrace: trace,
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
    pub fn location(&self) -> Location {
        self.location
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
            .field("location", &self.location)
            .finish()
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
    type Error = TryFromTermError;

    fn try_from(term: Term) -> Result<Class, TryFromTermError> {
        use self::Class::*;

        match term.decode().unwrap() {
            TypedTerm::Atom(atom) => match atom.name() {
                "error" => Ok(Error { arguments: None }),
                "exit" => Ok(Exit),
                "throw" => Ok(Throw),
                name => Err(TryFromTermError::AtomName(name)),
            },
            _ => Err(TryFromTermError::Type),
        }
    }
}

#[derive(Debug, Error)]
pub enum TryFromTermError {
    #[error("atom ({0}) is not in class names (error, exit, or throw)")]
    AtomName(&'static str),
    #[error("class is not an atom")]
    Type,
}
