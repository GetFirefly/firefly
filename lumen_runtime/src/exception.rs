use crate::term::Term;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Error { arguments: Option<Term> },
    Exit,
    Throw,
}

#[derive(Debug)]
pub struct Exception {
    pub class: Class,
    pub reason: Term,
    pub stacktrace: Option<Term>,
    //    #[cfg(debug_assertions)]
    pub file: &'static str,
    //    #[cfg(debug_assertions)]
    pub line: u32,
    //    #[cfg(debug_assertions)]
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
