use core::fmt;
use core::ops::Deref;

use alloc::sync::Arc;

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
