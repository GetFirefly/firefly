use core::convert::TryFrom;
use core::num::TryFromIntError;
use core::result::Result;

use crate::erts::exception::system::Alloc;
use crate::erts::process::Process;
use crate::erts::string::InvalidEncodingNameError;
use crate::erts::term::index::IndexError;
use crate::erts::term::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Error { arguments: Option<Term> },
    Exit,
    Throw,
}

impl TryFrom<Term> for Class {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Class, Exception> {
        use self::Class::*;

        match term.decode().unwrap() {
            TypedTerm::Atom(atom) => match atom.name() {
                "error" => Ok(Error { arguments: None }),
                "exit" => Ok(Exit),
                "throw" => Ok(Throw),
                _ => Err(badarg!()),
            },
            _ => Err(badarg!()),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Exception {
    pub class: Class,
    pub reason: Term,
    pub stacktrace: Option<Term>,
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

impl Exception {
    pub fn badarg(file: &'static str, line: u32, column: u32) -> Self {
        Self::error(Self::badarg_reason(), None, None, file, line, column)
    }

    pub fn badarith(file: &'static str, line: u32, column: u32) -> Self {
        Self::error(Self::badarith_reason(), None, None, file, line, column)
    }

    pub fn badarity(
        process: &Process,
        function: Term,
        arguments: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Result<Self, Alloc> {
        let reason = Self::badarity_reason(process, function, arguments)?;
        let error = Self::error(reason, None, None, file, line, column);

        Ok(error)
    }

    pub fn badfun(
        process: &Process,
        function: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Result<Self, Alloc> {
        let tag = Atom::str_to_term("badfun");
        let reason = process.tuple_from_slice(&[tag, function])?;

        let error = Self::error(reason, None, None, file, line, column);

        Ok(error)
    }

    pub fn badkey(
        process: &Process,
        key: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Result<Self, Alloc> {
        let tag = Atom::str_to_term("badkey");
        let reason = process.tuple_from_slice(&[tag, key])?;
        let error = Self::error(reason, None, None, file, line, column);

        Ok(error)
    }

    pub fn badmap(
        process: &Process,
        map: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Result<Self, Alloc> {
        let tag = Atom::str_to_term("badmap");
        let reason = process.tuple_from_slice(&[tag, map])?;
        let error = Self::error(reason, None, None, file, line, column);

        Ok(error)
    }

    pub fn exit(
        reason: Term,
        stacktrace: Option<Term>,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        let class = Class::Exit;
        Self::new(class, reason, stacktrace, file, line, column)
    }

    pub fn undef(
        process: &Process,
        module: Term,
        function: Term,
        arguments: Term,
        stacktrace_tail: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Result<Self, Alloc> {
        let reason = Self::undef_reason();
        let stacktrace =
            Self::undef_stacktrace(process, module, function, arguments, stacktrace_tail)?;
        let exit = Self::exit(reason, Some(stacktrace), file, line, column);

        Ok(exit)
    }

    // Private

    fn badarg_reason() -> Term {
        Atom::str_to_term("badarg")
    }

    fn badarith_reason() -> Term {
        Atom::str_to_term("badarith")
    }

    fn badarity_reason(process: &Process, function: Term, arguments: Term) -> Result<Term, Alloc> {
        let function_arguments = process.tuple_from_slice(&[function, arguments])?;

        process.tuple_from_slice(&[Self::badarity_tag(), function_arguments])
    }

    fn badarity_tag() -> Term {
        Atom::str_to_term("badarity")
    }

    fn error(
        reason: Term,
        arguments: Option<Term>,
        stacktrace: Option<Term>,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        let class = Class::Error { arguments };
        Self::new(class, reason, stacktrace, file, line, column)
    }

    fn new(
        class: Class,
        reason: Term,
        stacktrace: Option<Term>,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        Exception {
            class,
            reason,
            stacktrace,
            file,
            line,
            column,
        }
    }

    fn undef_reason() -> Term {
        Atom::str_to_term("undef")
    }

    fn undef_stacktrace(
        process: &Process,
        module: Term,
        function: Term,
        arguments: Term,
        tail: Term,
    ) -> Result<Term, Alloc> {
        let top = process.tuple_from_slice(&[
            module,
            function,
            arguments,
            // I'm not sure what this final empty list holds
            Term::NIL,
        ])?;

        process.cons(top, tail)
    }
}

impl Eq for Exception {}

impl From<core::convert::Infallible> for Exception {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<AtomError> for Exception {
    fn from(_: AtomError) -> Self {
        badarg!()
    }
}

impl From<BoolError> for Exception {
    fn from(_: BoolError) -> Self {
        badarg!()
    }
}

impl From<InvalidEncodingNameError> for Exception {
    fn from(_: InvalidEncodingNameError) -> Self {
        badarg!()
    }
}

impl From<ImproperList> for Exception {
    fn from(_: ImproperList) -> Self {
        badarg!()
    }
}

impl From<IndexError> for Exception {
    fn from(_: IndexError) -> Self {
        badarg!()
    }
}

impl From<TryFromIntError> for Exception {
    fn from(_: TryFromIntError) -> Self {
        badarg!()
    }
}

impl From<TryIntoIntegerError> for Exception {
    fn from(_: TryIntoIntegerError) -> Self {
        badarg!()
    }
}

impl From<TypeError> for Exception {
    fn from(_: TypeError) -> Self {
        badarg!()
    }
}

impl PartialEq for Exception {
    /// `file`, `line`, and `column` don't count for equality as they are for `Debug` only to help
    /// track down exceptions.
    fn eq(&self, other: &Exception) -> bool {
        (self.class == other.class)
            & (self.reason == other.reason)
            & (self.stacktrace == other.stacktrace)
    }
}

impl TryFrom<super::Exception> for Exception {
    type Error = TypeError;

    fn try_from(exception: super::Exception) -> Result<Self, Self::Error> {
        match exception {
            super::Exception::Runtime(runtime_exception) => Ok(runtime_exception),
            _ => Err(TypeError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod error {
        use super::Class::*;
        use super::*;

        #[test]
        fn without_arguments_stores_none() {
            let reason = Atom::str_to_term("badarg");

            let error = error!(reason);

            assert_eq!(error.reason, reason);
            assert_eq!(error.class, Error { arguments: None });
        }

        #[test]
        fn without_arguments_stores_some() {
            let reason = Atom::str_to_term("badarg");
            let arguments = Term::NIL;

            let error = error!(reason, Some(arguments));

            assert_eq!(error.reason, reason);
            assert_eq!(
                error.class,
                Error {
                    arguments: Some(arguments)
                }
            );
        }
    }
}
