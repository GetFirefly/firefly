use crate::erts::process::Process;
use crate::erts::term::prelude::Term;

use super::{Exception, Location, RuntimeException, Stacktrace};

#[inline]
pub fn badarg<S: Into<Stacktrace>>(stacktrace: S, location: Location) -> RuntimeException {
    self::error(atom!("badarg"), None, location, stacktrace)
}

#[inline]
pub fn badarith<S: Into<Stacktrace>>(stacktrace: S, location: Location) -> RuntimeException {
    self::error(atom!("badarith"), None, location, stacktrace)
}

pub fn badarity<S: Into<Stacktrace>>(
    process: &Process,
    fun: Term,
    args: Term,
    location: Location,
    stacktrace: S,
) -> Exception {
    match process.tuple_from_slice(&[fun, args]) {
        Ok(fun_args) => {
            let tag = atom("badarity");
            match process.tuple_from_slice(&[tag, fun_args]) {
                Ok(reason) => Exception::Runtime(self::error(reason, None, location, stacktrace)),
                Err(err) => err.into(),
            }
        }
        Err(err) => err.into(),
    }
}

pub fn badfun<S: Into<Stacktrace>>(
    process: &Process,
    fun: Term,
    location: Location,
    stacktrace: S,
) -> Exception {
    let tag = atom("badfun");
    match process.tuple_from_slice(&[tag, fun]) {
        Ok(reason) => Exception::Runtime(self::error(reason, None, location, stacktrace)),
        Err(err) => err.into(),
    }
}

pub fn badkey<S: Into<Stacktrace>>(
    process: &Process,
    key: Term,
    location: Location,
    stacktrace: S,
) -> Exception {
    let tag = atom("badkey");
    match process.tuple_from_slice(&[tag, key]) {
        Ok(reason) => Exception::Runtime(self::error(reason, None, location, stacktrace)),
        Err(err) => err.into(),
    }
}

pub fn badmap<S: Into<Stacktrace>>(
    process: &Process,
    map: Term,
    location: Location,
    stacktrace: S,
) -> Exception {
    let tag = atom("badmap");
    match process.tuple_from_slice(&[tag, map]) {
        Ok(reason) => Exception::Runtime(self::error(reason, None, location, stacktrace)),
        Err(err) => err.into(),
    }
}

pub fn undef(
    process: &Process,
    m: Term,
    f: Term,
    a: Term,
    location: Location,
    stacktrace_tail: Term,
) -> Exception {
    let reason = atom("undef");
    // I'm not sure what this final empty list holds
    match process.tuple_from_slice(&[m, f, a, Term::NIL /* ? */]) {
        Ok(top) => match process.cons(top, stacktrace_tail) {
            Ok(stacktrace) => Exception::Runtime(self::exit(reason, location, stacktrace)),
            Err(err) => err.into(),
        },
        Err(err) => err.into(),
    }
}

pub fn raise<S: Into<Stacktrace>>(
    class: super::Class,
    reason: Term,
    location: Location,
    stacktrace: S,
) -> RuntimeException {
    use super::Class;

    match class {
        Class::Exit => self::exit(reason, location, stacktrace),
        Class::Throw => self::throw(reason, location, stacktrace),
        Class::Error { arguments } => self::error(reason, arguments, location, stacktrace),
    }
}

pub fn exit<S: Into<Stacktrace>>(
    reason: Term,
    location: Location,
    stacktrace: S,
) -> RuntimeException {
    use super::Exit;

    RuntimeException::Exit(Exit::new(reason, location, stacktrace))
}

#[inline]
pub fn error<S: Into<Stacktrace>>(
    reason: Term,
    args: Option<Term>,
    location: Location,
    stacktrace: S,
) -> RuntimeException {
    use super::Error;

    RuntimeException::Error(Error::new(reason, args, location, stacktrace))
}

#[inline]
pub fn throw<S: Into<Stacktrace>>(
    reason: Term,
    location: Location,
    stacktrace: S,
) -> RuntimeException {
    use super::Throw;

    RuntimeException::Throw(Throw::new(reason, location, stacktrace))
}

#[inline(always)]
fn atom(s: &'static str) -> Term {
    use crate::erts::term::prelude::Atom;

    Atom::str_to_term(s)
}
