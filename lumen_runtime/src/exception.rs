use crate::term::Term;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(Debug))]
pub enum Class {
    Error { arguments: Option<Term> },
    Exit,
    Throw,
}

#[cfg_attr(test, derive(Debug))]
pub struct Exception {
    pub class: Class,
    pub reason: Term,
    pub stacktrace: Option<Term>,
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

impl Eq for Exception {}

impl PartialEq for Exception {
    /// `file`, `line`, and `column` don't count for equality as they are for `Debug` only to help
    /// track down exceptions.
    fn eq(&self, other: &Exception) -> bool {
        (self.class == other.class)
            & (self.reason == other.reason)
            & (self.stacktrace == other.stacktrace)
    }
}

pub type Result = std::result::Result<Term, Exception>;

#[cfg(test)]
macro_rules! assert_badarg {
    ($left:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        assert_error!($left, Term::str_to_atom("badarg", DoNotCare).unwrap())
    }};
}

#[cfg(test)]
macro_rules! assert_badarith {
    ($left:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        assert_error!($left, Term::str_to_atom("badarith", DoNotCare).unwrap())
    }};
}

#[cfg(test)]
macro_rules! assert_badkey {
    ($left:expr, $key:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badkey = Term::str_to_atom("badkey", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(&[badkey, $key], $process);

        assert_error!($left, reason)
    }};
    ($left:expr, $key:expr, $process:expr,) => {{
        assert_badkey!($left, $key, $process)
    }};
}

#[cfg(test)]
macro_rules! assert_badmap {
    ($left:expr, $map:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badmap = Term::str_to_atom("badmap", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(&[badmap, $map], $process);

        assert_error!($left, reason)
    }};
    ($left:expr, $map:expr, $process:expr,) => {{
        assert_badmap($left, $map, $process)
    }};
}

#[cfg(test)]
macro_rules! assert_error {
    ($left:expr, $reason:expr) => {
        assert_eq!($left, Err(error!($reason)))
    };
    ($left:expr, $reason:expr,) => {
        assert_eq!($left, Err(error!($reason)))
    };
    ($left:expr, $reason:expr, $arguments:expr) => {
        assert_eq!($left, Err(error!($reason, $arguments)))
    };
    ($left:expr, $reason:expr, $arguments:expr,) => {
        assert_eq!($left, Err(error!($reason, $arguments)))
    };
}

#[cfg(test)]
macro_rules! assert_exit {
    ($left:expr, $reason:expr) => {
        assert_eq!($left, Err(exit!($reason)))
    };
    ($left:expr, $reason:expr,) => {
        assert_exit($left, $reason)
    };
}

#[cfg(test)]
macro_rules! assert_raises {
    ($left:expr, $class:expr, $reason:expr, $stacktrace:expr) => {
        assert_eq!($left, Err(raise!($class, $reason, $stacktrace)))
    };
    ($left:expr, $class:expr, $reason:expr, $stacktrace:expr,) => {
        assert_raises!($left, $class, $reason, $stacktrace)
    };
}

#[cfg(test)]
macro_rules! assert_throw {
    ($left:expr, $reason:expr) => {
        assert_eq!($left, Err(throw!($reason)))
    };
    ($left:expr, $reason:expr,) => {
        assert_throw($left, $reason)
    };
}

macro_rules! badarg {
    () => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        error!(Term::str_to_atom("badarg", DoNotCare).unwrap())
    }};
}

macro_rules! badarith {
    () => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        error!(Term::str_to_atom("badarith", DoNotCare).unwrap())
    }};
}

macro_rules! badmap {
    ($map:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badmap = Term::str_to_atom("badmap", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(&[badmap, $map], $process);

        error!(reason)
    }};
    ($map:expr, $process:expr,) => {{
        bad_map!($map, $process)
    }};
}

macro_rules! error {
    ($reason:expr) => {{
        error!($reason, None)
    }};
    ($reason:expr, $arguments:expr) => {{
        use crate::exception::Class::Error;

        exception!(
            Error {
                arguments: $arguments
            },
            $reason
        )
    }};
    ($reason:expr, $arguments:expr,) => {{
        error!($reason, $arguments)
    }};
}

macro_rules! exception {
    ($class:expr, $reason:expr) => {
        exception!($class, $reason, None)
    };
    ($class:expr, $reason:expr,) => {
        exception!($class, $reason)
    };
    ($class:expr, $reason:expr, $stacktrace:expr) => {{
        use crate::exception::Exception;

        Exception {
            class: $class,
            reason: $reason,
            stacktrace: $stacktrace,
            file: file!(),
            line: line!(),
            column: column!(),
        }
    }};
    ($class:expr, $reason:expr, $stacktrace:expr) => {
        exception!($class, $reason, $stacktrace)
    };
}

macro_rules! exit {
    ($reason:expr) => {{
        use crate::exception::Class::Exit;

        exception!(Exit, $reason)
    }};
}

macro_rules! raise {
    ($class:expr, $reason:expr, $stacktrace:expr) => {
        exception!($class, $reason, $stacktrace)
    };
    ($class:expr, $reason:expr, $stacktrace:expr,) => {
        exception!($class, $reason, $stacktrace)
    };
}

macro_rules! throw {
    ($reason:expr) => {{
        use crate::exception::Class::Throw;

        exception!(Throw, $reason)
    }};
}

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
