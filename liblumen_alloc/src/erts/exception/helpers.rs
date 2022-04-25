use std::sync::Arc;

use crate::erts::process::trace::Trace;
use crate::erts::process::Process;
use crate::erts::term::prelude::Term;

use super::{ArcError, Exception, RuntimeException};

#[inline]
pub fn badarg(trace: Arc<Trace>, source: Option<ArcError>) -> RuntimeException {
    self::error(atom!(badarg), None, trace, source)
}

#[inline]
pub fn badarith(trace: Arc<Trace>, source: Option<ArcError>) -> RuntimeException {
    self::error(atom!(badarith), None, trace, source)
}

pub fn badarity(
    process: &Process,
    fun: Term,
    args: Term,
    trace: Arc<Trace>,
    source: Option<ArcError>,
) -> RuntimeException {
    let fun_args = process.tuple_from_slice(&[fun, args]);
    let tag = atom!(badarity);
    let reason = process.tuple_from_slice(&[tag, fun_args]);

    self::error(reason, None, trace, source)
}

pub fn badfun(process: &Process, fun: Term, trace: Arc<Trace>, source: ArcError) -> Exception {
    let tag = atom!(badfun);
    let reason = process.tuple_from_slice(&[tag, fun]);
    Exception::Runtime(self::error(reason, None, trace, Some(source)))
}

pub fn badkey(process: &Process, key: Term, trace: Arc<Trace>, source: ArcError) -> Exception {
    let tag = atom!(badkey);
    let reason = process.tuple_from_slice(&[tag, key]);
    Exception::Runtime(self::error(reason, None, trace, Some(source)))
}

pub fn badmap(
    process: &Process,
    map: Term,
    trace: Arc<Trace>,
    source: Option<ArcError>,
) -> RuntimeException {
    let tag = atom!(badmap);
    let reason = process.tuple_from_slice(&[tag, map]);
    self::error(reason, None, trace, source)
}

#[inline]
pub fn undef(trace: Arc<Trace>, source: Option<ArcError>) -> Exception {
    Exception::Runtime(self::exit(atom!(undef), trace, source))
}

#[inline]
pub fn raise(
    class: super::Class,
    reason: Term,
    trace: Arc<Trace>,
    source: Option<ArcError>,
) -> RuntimeException {
    use super::Class;

    match class {
        Class::Exit => self::exit(reason, trace, source),
        Class::Throw => self::throw(reason, trace, source),
        Class::Error { arguments } => self::error(reason, arguments, trace, source),
    }
}

#[inline]
pub fn exit(reason: Term, trace: Arc<Trace>, source: Option<ArcError>) -> RuntimeException {
    use super::Exit;
    RuntimeException::Exit(Exit::new(reason, trace, source))
}

#[inline]
pub fn error(
    reason: Term,
    args: Option<Term>,
    trace: Arc<Trace>,
    source: Option<ArcError>,
) -> RuntimeException {
    use super::Error;

    RuntimeException::Error(Error::new(reason, args, trace, source))
}

#[inline]
pub fn throw(reason: Term, trace: Arc<Trace>, source: Option<ArcError>) -> RuntimeException {
    use super::Throw;

    RuntimeException::Throw(Throw::new(reason, trace, source))
}
