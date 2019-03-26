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
        assert_bad_argument!($left, $process)
    }};
}

#[macro_export]
macro_rules! assert_bad_key {
    ($left:expr, $key:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badkey = Term::str_to_atom("badkey", DoNotCare, $process).unwrap();
        let reason = Term::slice_to_tuple(&[badkey, $key], $process);

        assert_error!($left, reason, $process)
    }};
    ($left:expr, $key:expr, $process:expr,) => {{
        assert_bad_map($left, $key, $process)
    }};
}

#[macro_export]
macro_rules! assert_bad_map {
    ($left:expr, $map:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badmap = Term::str_to_atom("badmap", DoNotCare, $process).unwrap();
        let reason = Term::slice_to_tuple(&[badmap, $map], $process);

        assert_error!($left, reason, $process)
    }};
    ($left:expr, $map:expr, $process:expr,) => {{
        assert_bad_map($left, $map, $process)
    }};
}

#[macro_export]
macro_rules! assert_error {
    ($left:expr, $reason:expr, $process:expr) => {
        assert_eq_in_process!($left, Err(error!($reason)), $process)
    };
    ($left:expr, $reason:expr, $process:expr,) => {
        assert_eq_in_process!($left, Err(error!($reason)), $process)
    };
    ($left:expr, $reason:expr, $arguments:expr, $process:expr) => {
        assert_eq_in_process!($left, Err(error!($reason, $arguments)), $process)
    };
    ($left:expr, $reason:expr, $arguments:expr, $process:expr,) => {
        assert_eq_in_process!($left, Err(error!($reason, $arguments)), $process)
    };
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
macro_rules! bad_map {
    ($map:expr, $process:expr) => {{
        use crate::atom::DoNotCare;
        use crate::term::Term;

        let badmap = Term::str_to_atom("badmap", DoNotCare, $process).unwrap();
        let reason = Term::slice_to_tuple(&[badmap, map], $process);

        error!(reason)
    }};
    ($map:expr, $process:expr,) => {{
        bad_map!($map, $process)
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

#[cfg(test)]
mod tests {
    use super::*;

    mod error {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::atom::Existence::DoNotCare;
        use crate::environment::{self, Environment};

        #[test]
        fn without_arguments_stores_none() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let reason = Term::str_to_atom("badarg", DoNotCare, &mut process).unwrap();

            let error = error!(reason);

            assert_eq_in_process!(error.reason, reason, &mut process);
            assert_eq_in_process!(error.arguments, None, &mut process);
        }

        #[test]
        fn without_arguments_stores_some() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let reason = Term::str_to_atom("badarg", DoNotCare, &mut process).unwrap();
            let arguments = Term::EMPTY_LIST;

            let error = error!(reason, arguments);

            assert_eq_in_process!(error.reason, reason, &mut process);
            assert_eq_in_process!(error.arguments, Some(arguments), &mut process);
        }
    }
}
