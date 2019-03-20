use std::fmt::{self, Debug};

use crate::term::Term;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Error,
    Exit,
    Throw,
}

pub struct Exception {
    pub class: Class,
    pub reason: Term,
    pub arguments: Option<Term>,
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

// Needed to support `std::result::Result.unwrap`
impl Debug for Exception {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Exception {{ class: {:?}, reason: Term {{ tagged: {:#b} }}, file: {:?}, line: {:?}, column: {:?} }}",
            self.class,
            self.reason.tagged,
            self.file,
            self.line,
            self.column
        )
    }
}

pub type Result = std::result::Result<Term, Exception>;

#[macro_export]
macro_rules! assert_bad_argument {
    ($left:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        assert_error!(
            $left,
            Term::str_to_atom("badarg", DoNotCare, $process).unwrap(),
            $process
        )
    }};
    ($left:expr, $process:expr,) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        assert_error!(
            $left,
            Term::str_to_atom("badarg", DoNotCare, $process).unwrap(),
            $process
        )
    }};
}

#[macro_export]
macro_rules! assert_error {
    ($left:expr, $reason:expr, $process:expr) => {{
        assert_eq_in_process!($left, Err(error!($reason)), $process)
    }};
    ($left:expr, $reason:expr, $process:expr,) => {{
        assert_eq_in_process!($left, Err(error!($reason)), $process)
    }};
    ($left:expr, $reason:expr, $arguments:expr, $process:expr) => {{
        assert_eq_in_process!($left, Err(error!($reason, $arguments)), $process)
    }};
    ($left:expr, $reason:expr, $arguments:expr, $process:expr,) => {{
        assert_eq_in_process!($left, Err(error!($reason, $arguments)), $process)
    }};
}

#[macro_export]
macro_rules! bad_argument {
    ($process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        error!(Term::str_to_atom("badarg", DoNotCare, $process).unwrap())
    }};
}

#[macro_export]
macro_rules! error {
    ($reason:expr) => {{
        use crate::exception::{Class::Error, Exception};

        Exception {
            class: Error,
            reason: $reason,
            arguments: None,
            file: file!(),
            line: line!(),
            column: column!(),
        }
    }};
    ($reason:expr, $arguments:expr) => {{
        use crate::exception::{Class::Error, Exception};

        Exception {
            class: Error,
            reason: $reason,
            arguments: Some($arguments),
            file: file!(),
            line: line!(),
            column: column!(),
        }
    }};
    ($reason:expr, $arguments:expr,) => {{
        error!($reason, $arguments)
    }};
}
