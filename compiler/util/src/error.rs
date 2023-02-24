use std::fmt;
use std::hash;
use std::panic;
use std::sync::Arc;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    Debug,
    Info,
    Warning,
    Error,
    Silent,
}
impl Verbosity {
    pub fn from_level(level: isize) -> Self {
        if level < 0 {
            return Verbosity::Silent;
        }

        match level {
            0 => Verbosity::Warning,
            1 => Verbosity::Info,
            _ => Verbosity::Debug,
        }
    }

    pub fn is_silent(&self) -> bool {
        match self {
            Self::Silent => true,
            _ => false,
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("{0:?}")]
pub struct HelpRequested(pub &'static str, pub Option<Vec<&'static str>>);
impl HelpRequested {
    pub fn primary(&self) -> &'static str {
        self.0
    }

    pub fn subcommands(&self) -> Option<&Vec<&'static str>> {
        self.1.as_ref()
    }
}

#[derive(Error, Clone)]
#[error("{err}")]
pub struct ArcError {
    #[from]
    err: Arc<anyhow::Error>,
}
impl ArcError {
    pub fn new(err: anyhow::Error) -> Self {
        Self { err: Arc::new(err) }
    }
}
impl fmt::Debug for ArcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if cfg!(debug_assertions) {
            write!(f, "{:#?}", self.err)
        } else {
            write!(f, "{:?}", self.err)
        }
    }
}
impl Eq for ArcError {}
impl PartialEq for ArcError {
    fn eq(&self, other: &Self) -> bool {
        self.err.to_string() == other.err.to_string()
    }
}
impl hash::Hash for ArcError {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.err.to_string().hash(state);
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ErrorReported;

pub struct FatalErrorMarker;

/// Used as a return value to signify a fatal error occurred
#[derive(Copy, Clone, Debug)]
#[must_use]
pub struct FatalError;
impl FatalError {
    pub fn raise(self) -> ! {
        panic::resume_unwind(Box::new(FatalErrorMarker))
    }
}
// Don't implement Send on FatalError. This makes it impossible to panic!(FatalError).
// We don't want to invoke the panic handler and print a backtrace for fatal errors.
impl !Send for FatalError {}
impl fmt::Display for FatalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "The compiler has encountered a fatal error")
    }
}
