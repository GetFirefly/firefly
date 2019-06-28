use crate::atom::Existence::DoNotCare;
use crate::process::Process;
use crate::term::Term;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Clone))]
pub enum Class {
    Error { arguments: Option<Term> },
    Exit,
    Throw,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Clone))]
pub struct Exception {
    pub class: Class,
    pub reason: Term,
    pub stacktrace: Option<Term>,
    #[cfg(debug_assertions)]
    pub file: &'static str,
    #[cfg(debug_assertions)]
    pub line: u32,
    #[cfg(debug_assertions)]
    pub column: u32,
}

impl Exception {
    #[cfg(debug_assertions)]
    pub fn badarg(file: &'static str, line: u32, column: u32) -> Self {
        Self::error(Self::badarg_reason(), None, None, file, line, column)
    }

    #[cfg(not(debug_assertions))]
    pub fn badarg() -> Self {
        Self::error(Self::badarg_reason(), None, None)
    }

    #[cfg(debug_assertions)]
    pub fn badarith(file: &'static str, line: u32, column: u32) -> Self {
        Self::error(Self::badarith_reason(), None, None, file, line, column)
    }

    #[cfg(not(debug_assertions))]
    pub fn badarith() -> Self {
        Self::error(Self::badarith_reason(), None, None)
    }

    #[cfg(debug_assertions)]
    pub fn badarity(
        function: Term,
        arguments: Term,
        process: &Process,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        let reason = Self::badarity_reason(function, arguments, process);

        Self::error(reason, None, None, file, line, column)
    }

    #[cfg(not(debug_assertions))]
    pub fn badarity(function: Term, arguments: Term, process: &Process) -> Self {
        let reason = Self::badarity_reason(function, arguments, process);

        Self::error(reason, None, None)
    }

    // Private

    fn badarg_reason() -> Term {
        Term::str_to_atom("badarg", DoNotCare).unwrap()
    }

    fn badarith_reason() -> Term {
        Term::str_to_atom("badarith", DoNotCare).unwrap()
    }

    fn badarity_reason(function: Term, arguments: Term, process: &Process) -> Term {
        Term::slice_to_tuple(
            &[
                Self::badarity_tag(),
                Term::slice_to_tuple(&[function, arguments], process),
            ],
            process,
        )
    }

    fn badarity_tag() -> Term {
        Term::str_to_atom("badarity", DoNotCare).unwrap()
    }

    #[cfg(debug_assertions)]
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

    #[cfg(not(debug_assertions))]
    fn error(reason: Term, arguments: Option<Term>, stacktrace: Option<Term>) -> Self {
        let class = Class::Error { arguments };
        Self::new(class, reason, stacktrace)
    }

    #[cfg(debug_assertions)]
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

    #[cfg(not(debug_assertions))]
    fn new(class: Class, reason: Term, stacktrace: Option<Term>) -> Self {
        Exception {
            class,
            reason,
            stacktrace,
        }
    }
}

impl Eq for Exception {}

impl PartialEq for Exception {
    /// `file`, `line`, and `column` don't count for equality as they are for `Debug` only to help
    /// track down exceptions.
    fn eq(&self, other: &Exception) -> bool {
        (self.class == other.class)
            && (self.reason == other.reason)
            && (self.stacktrace == other.stacktrace)
    }
}

pub type Result = std::result::Result<Term, Exception>;

#[cfg(test)]
mod tests {
    use super::*;

    mod error {
        use super::Class::*;
        use super::*;

        use crate::atom::Existence::DoNotCare;

        #[test]
        fn without_arguments_stores_none() {
            let reason = Term::str_to_atom("badarg", DoNotCare).unwrap();

            let error = error!(reason);

            assert_eq!(error.reason, reason);
            assert_eq!(error.class, Error { arguments: None });
        }

        #[test]
        fn without_arguments_stores_some() {
            let reason = Term::str_to_atom("badarg", DoNotCare).unwrap();
            let arguments = Term::EMPTY_LIST;

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
