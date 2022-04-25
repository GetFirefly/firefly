use std::fmt::{self, Display};
use std::ops::Deref;
use std::sync::Arc;

use crate::erts::term::pid::InvalidPidError;
use crate::erts::term::prelude::{TermDecodingError, TermEncodingError, TypeError};

#[derive(Clone)]
pub struct ArcError(Arc<anyhow::Error>);
impl ArcError {
    pub fn new(err: anyhow::Error) -> Self {
        Self(Arc::new(err))
    }

    pub fn from_err<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self(Arc::new(anyhow::Error::new(err)))
    }

    pub fn context<C>(&self, context: C) -> Self
    where
        C: Display + Send + Sync + 'static,
    {
        Self::new(anyhow::Error::new(self.clone()).context(context))
    }
}
impl Deref for ArcError {
    type Target = anyhow::Error;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}
impl fmt::Debug for ArcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl fmt::Display for ArcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ArcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}
impl From<anyhow::Error> for ArcError {
    fn from(err: anyhow::Error) -> Self {
        Self::new(err)
    }
}
impl From<InvalidPidError> for ArcError {
    fn from(err: InvalidPidError) -> Self {
        Self::from_err(err)
    }
}
impl From<TermDecodingError> for ArcError {
    fn from(err: TermDecodingError) -> Self {
        Self::from_err(err)
    }
}
impl From<TermEncodingError> for ArcError {
    fn from(err: TermEncodingError) -> Self {
        Self::from_err(err)
    }
}
impl From<TypeError> for ArcError {
    fn from(err: TypeError) -> Self {
        Self::from_err(err)
    }
}
