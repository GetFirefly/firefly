use core::convert::Infallible;
use core::mem;

use thiserror::Error;

use crate::erts::term::prelude::{TermDecodingError, TermEncodingError};

#[derive(Error, Debug, Clone)]
pub enum SystemException {
    #[error("allocation failed")]
    Alloc(#[from] super::Alloc),
    #[error("term encoding failed: {0:?}")]
    TermEncodingFailed(#[from] TermEncodingError),
    #[error("term encoding failed: {0:?}")]
    TermDecodingFailed(#[from] TermDecodingError),
}

impl Eq for SystemException {}
impl PartialEq for SystemException {
    fn eq(&self, other: &Self) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
}

impl From<Infallible> for SystemException {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}
