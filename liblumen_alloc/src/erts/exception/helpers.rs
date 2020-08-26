use std::sync::Arc;

use crate::erts::process::trace::Trace;
use crate::erts::process::Process;
use crate::erts::term::prelude::Term;

use super::{ArcError, Exception, RuntimeException};

#[inline]
pub fn badarg(trace: Arc<Trace>) -> RuntimeException {
    self::error(atom("badarg"), None, trace)
}

#[inline]
pub fn badarg_with_source(source: ArcError) -> RuntimeException {
    self::error_with_source(atom("badarg"), None, source)
}

#[inline]
pub fn badarith(trace: Arc<Trace>) -> RuntimeException {
    self::error(atom("badarith"), None, trace)
}

pub fn badarity(process: &Process, fun: Term, args: Term, trace: Arc<Trace>) -> Exception {
    let fun_args = process.tuple_from_slice(&[fun, args]);
    let tag = atom("badarity");
    let reason = process.tuple_from_slice(&[tag, fun_args]);

    Exception::Runtime(self::error(reason, None, trace))
}

pub fn badfun(process: &Process, fun: Term, source: ArcError) -> Exception {
    let tag = atom("badfun");
    match process.tuple_from_slice(&[tag, fun]) {
        Ok(reason) => Exception::Runtime(self::error_with_source(reason, None, source)),
        Err(err) => err.into(),
    }
}

pub fn badkey(process: &Process, key: Term, source: ArcError) -> Exception {
    let tag = atom("badkey");
    match process.tuple_from_slice(&[tag, key]) {
        Ok(reason) => Exception::Runtime(self::error_with_source(reason, None, source)),
        Err(err) => err.into(),
    }
}

pub fn badmap(process: &Process, map: Term, source: ArcError) -> Exception {
    let tag = atom("badmap");
    match process.tuple_from_slice(&[tag, map]) {
        Ok(reason) => Exception::Runtime(self::error_with_source(reason, None, source)),
        Err(err) => err.into(),
    }
}

#[inline]
pub fn undef(trace: Arc<Trace>) -> Exception {
    Exception::Runtime(self::exit(atom("undef"), trace))
}

#[inline]
pub fn raise(class: super::Class, reason: Term, trace: Arc<Trace>) -> RuntimeException {
    use super::Class;

    match class {
        Class::Exit => self::exit(reason, trace),
        Class::Throw => self::throw(reason, trace),
        Class::Error { arguments } => self::error(reason, arguments, trace),
    }
}

#[inline]
pub fn raise_with_source(class: super::Class, reason: Term, source: ArcError) -> RuntimeException {
    use super::Class;

    match class {
        Class::Exit => self::exit_with_source(reason, source),
        Class::Throw => self::throw_with_source(reason, source),
        Class::Error { arguments } => self::error_with_source(reason, arguments, source),
    }
}

#[inline]
pub fn exit(reason: Term, trace: Arc<Trace>) -> RuntimeException {
    use super::Exit;
    RuntimeException::Exit(Exit::new_with_trace(reason, trace))
}

#[inline]
pub fn exit_with_source(reason: Term, source: ArcError) -> RuntimeException {
    use super::Exit;

    RuntimeException::Exit(Exit::new(reason, source))
}

#[inline]
pub fn error(reason: Term, args: Option<Term>, trace: Arc<Trace>) -> RuntimeException {
    use super::Error;

    RuntimeException::Error(Error::new_with_trace(reason, args, trace))
}

#[inline]
pub fn error_with_source(reason: Term, args: Option<Term>, source: ArcError) -> RuntimeException {
    use super::Error;

    RuntimeException::Error(Error::new(reason, args, source))
}

#[inline]
pub fn throw(reason: Term, trace: Arc<Trace>) -> RuntimeException {
    use super::Throw;

    RuntimeException::Throw(Throw::new_with_trace(reason, trace))
}

#[inline]
pub fn throw_with_source(reason: Term, source: ArcError) -> RuntimeException {
    use super::Throw;

    RuntimeException::Throw(Throw::new(reason, source))
}

#[inline(always)]
fn atom(s: &'static str) -> Term {
    use crate::erts::term::prelude::Atom;

    Atom::str_to_term(s)
}
