use core::fmt;
use core::convert::TryFrom;

use thiserror::Error;

use crate::erts::term::prelude::{Term, TypedTerm, Encoded};

use super::Location;
use super::RuntimeException;


#[derive(Error, Debug, Clone)]
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

#[derive(Error, Debug, Clone)]
#[error("** error: {reason} at {location}")]
pub struct Error {
    reason: Term,
    arguments: Option<Term>,
    stacktrace: Option<Term>,
    location: Location,
}
impl Error {
    pub fn new(reason: Term, arguments: Option<Term>, location: Location) -> Self {
        Self::new_with_trace(reason, arguments, location, None)
    }
    pub fn new_with_trace(reason: Term, arguments: Option<Term>, location: Location, trace: Option<Term>) -> Self {
        Self {
            reason,
            arguments,
            location,
            stacktrace: trace,
        }
    }
    pub fn class(&self) -> Class {
        Class::Error { arguments: self.arguments }
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
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason && self.stacktrace == other.stacktrace
    }
}

#[derive(Error, Debug, Clone)]
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
impl PartialEq for Exit {
    fn eq(&self, other: &Self) -> bool {
        self.reason == other.reason && self.stacktrace == other.stacktrace
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
    type Error = RuntimeException;

    fn try_from(term: Term) -> Result<Class, RuntimeException> {
        use self::Class::*;

        match term.decode().unwrap() {
            TypedTerm::Atom(atom) => match atom.name() {
                "error" => Ok(Error { arguments: None }),
                "exit" => Ok(Exit),
                "throw" => Ok(Throw),
                _ => Err(super::badarg(location!())),
            },
            _ => Err(super::badarg(location!())),
        }
    }
}
