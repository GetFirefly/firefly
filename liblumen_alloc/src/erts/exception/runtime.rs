use std::convert::TryFrom;
use std::sync::Arc;

use thiserror::Error;

use crate::erts::process::alloc::TermAlloc;
use crate::erts::process::trace::Trace;
use crate::erts::term::prelude::*;

use super::{ArcError, ErlangException, Exception, SystemException, UnexpectedExceptionError};

#[derive(Error, Debug, Clone)]
pub enum RuntimeException {
    #[error("{0}")]
    Throw(#[from] super::Throw),
    #[error("{0}")]
    Error(#[from] super::Error),
    #[error("{0}")]
    Exit(#[from] super::Exit),
}
impl Eq for RuntimeException {}
impl PartialEq for RuntimeException {
    fn eq(&self, other: &Self) -> bool {
        use RuntimeException::*;
        match (self, other) {
            (Throw(ref lhs), Throw(ref rhs)) => lhs.eq(rhs),
            (Error(ref lhs), Error(ref rhs)) => lhs.eq(rhs),
            (Exit(ref lhs), Exit(ref rhs)) => lhs.eq(rhs),
            _ => false,
        }
    }
}
impl RuntimeException {
    pub fn class(&self) -> super::Class {
        match self {
            RuntimeException::Throw(e) => e.class(),
            RuntimeException::Exit(e) => e.class(),
            RuntimeException::Error(e) => e.class(),
        }
    }

    pub fn reason(&self) -> Term {
        match self {
            RuntimeException::Throw(e) => e.reason(),
            RuntimeException::Exit(e) => e.reason(),
            RuntimeException::Error(e) => e.reason(),
        }
    }

    pub fn stacktrace(&self) -> Arc<Trace> {
        match self {
            RuntimeException::Throw(e) => e.stacktrace(),
            RuntimeException::Exit(e) => e.stacktrace(),
            RuntimeException::Error(e) => e.stacktrace(),
        }
    }

    #[inline]
    pub fn layout(&self) -> std::alloc::Layout {
        use crate::borrow::CloneToProcess;
        use std::alloc::Layout;

        // The class and trace are 1 word each and stored inline with the tuple,
        // the reason is the only potentially dynamic sized term. Simply calculate
        // the tuple + a block of memory capable of holding the size in words of
        // the reason
        let words = self.reason().size_in_words();
        let layout = Layout::new::<ErlangException>();
        let (layout, _) = layout
            .extend(Layout::array::<Term>(words).unwrap())
            .unwrap();
        layout.pad_to_align()
    }

    #[inline]
    pub fn as_error_tuple<A>(&self, heap: &mut A) -> super::AllocResult<Term>
    where
        A: TermAlloc,
    {
        match self {
            RuntimeException::Throw(e) => e.as_error_tuple(heap),
            RuntimeException::Exit(e) => e.as_error_tuple(heap),
            RuntimeException::Error(e) => e.as_error_tuple(heap),
        }
    }

    #[inline]
    pub fn as_erlang_exception(&self) -> Box<ErlangException> {
        match self {
            RuntimeException::Throw(e) => e.as_erlang_exception(),
            RuntimeException::Exit(e) => e.as_erlang_exception(),
            RuntimeException::Error(e) => e.as_erlang_exception(),
        }
    }

    pub fn source(&self) -> Option<ArcError> {
        match self {
            RuntimeException::Throw(e) => e.source(),
            RuntimeException::Exit(e) => e.source(),
            RuntimeException::Error(e) => e.source(),
        }
    }
}

impl From<anyhow::Error> for RuntimeException {
    fn from(err: anyhow::Error) -> Self {
        badarg!(Trace::capture(), ArcError::new(err))
    }
}

impl TryFrom<Exception> for RuntimeException {
    type Error = UnexpectedExceptionError<RuntimeException, SystemException>;

    fn try_from(err: Exception) -> Result<Self, <Self as TryFrom<Exception>>::Error> {
        match err {
            Exception::Runtime(e) => Ok(e),
            Exception::System(_) => Err(UnexpectedExceptionError::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom;
    use crate::erts::exception::Class;

    mod error {
        use super::*;

        #[test]
        fn without_arguments_stores_none() {
            let reason = atom!("badarg");
            let error = error!(reason, Trace::capture());

            assert_eq!(error.reason(), reason);
            assert_eq!(error.class(), Class::Error { arguments: None });
        }

        #[test]
        fn with_arguments_stores_some() {
            let reason = atom!("badarg");
            let arguments = Term::NIL;
            let error = error!(reason, arguments, Trace::capture());

            assert_eq!(error.reason(), reason);
            assert_eq!(
                error.class(),
                Class::Error {
                    arguments: Some(arguments)
                }
            );
        }
    }
}
