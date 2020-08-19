use crate::erts::process::Process;
use crate::erts::term::prelude::Term;

use super::{ArcError, Exception, RuntimeException};

#[inline]
pub fn badarg(stacktrace: Option<Term>, source: ArcError) -> RuntimeException {
    self::error(atom("badarg"), None, stacktrace, source)
}

#[inline]
pub fn badarith(source: ArcError) -> RuntimeException {
    self::error(atom("badarith"), None, None, source)
}

pub fn badarity(process: &Process, fun: Term, args: Term, source: ArcError) -> Exception {
    let fun_args = process.tuple_from_slice(&[fun, args]);
    let tag = atom("badarity");
    let reason = process.tuple_from_slice(&[tag, fun_args]);

    Exception::Runtime(self::error(reason, None, None, source))
}

pub fn badfun(process: &Process, fun: Term, source: ArcError) -> Exception {
    let tag = atom("badfun");
    let reason = process.tuple_from_slice(&[tag, fun]);

    Exception::Runtime(self::error(reason, None, None, source))
}

pub fn badkey(process: &Process, key: Term, source: ArcError) -> Exception {
    let tag = atom("badkey");
    let reason = process.tuple_from_slice(&[tag, key]);

    Exception::Runtime(self::error(reason, None, None, source))
}

pub fn badmap(process: &Process, map: Term, source: ArcError) -> Exception {
    let tag = atom("badmap");
    let reason = process.tuple_from_slice(&[tag, map]);

    Exception::Runtime(self::error(reason, None, None, source))
}

pub fn undef(
    process: &Process,
    m: Term,
    f: Term,
    a: Term,
    stacktrace_tail: Term,
    source: ArcError,
) -> Exception {
    let reason = atom("undef");
    // TODO empty list should be the location `[file: charlist(), line: integer()]`
    let top = process.tuple_from_slice(&[m, f, a, Term::NIL]);
    let stacktrace = process.cons(top, stacktrace_tail);

    Exception::Runtime(self::exit(reason, Some(stacktrace), source))
}

#[inline]
pub fn raise(
    class: super::Class,
    reason: Term,
    stacktrace: Option<Term>,
    source: ArcError,
) -> RuntimeException {
    use super::Class;

    match class {
        Class::Exit => self::exit(reason, stacktrace, source),
        Class::Throw => self::throw(reason, stacktrace, source),
        Class::Error { arguments } => self::error(reason, arguments, stacktrace, source),
    }
}

#[inline]
pub fn exit(reason: Term, stacktrace: Option<Term>, source: ArcError) -> RuntimeException {
    use super::Exit;

    RuntimeException::Exit(Exit::new_with_trace(reason, stacktrace, source))
}

#[inline]
pub fn error(
    reason: Term,
    args: Option<Term>,
    stacktrace: Option<Term>,
    source: ArcError,
) -> RuntimeException {
    use super::Error;

    RuntimeException::Error(Error::new_with_trace(reason, args, stacktrace, source))
}

#[inline]
pub fn throw(reason: Term, stacktrace: Option<Term>, source: ArcError) -> RuntimeException {
    use super::Throw;

    RuntimeException::Throw(Throw::new_with_trace(reason, stacktrace, source))
}

#[inline(always)]
fn atom(s: &'static str) -> Term {
    use crate::erts::term::prelude::Atom;

    Atom::str_to_term(s)
}
