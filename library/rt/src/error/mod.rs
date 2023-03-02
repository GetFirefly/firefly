mod erlang;
#[cfg(feature = "std")]
pub mod printer;

pub use self::erlang::{ErlangException, ExceptionClass};

use alloc::sync::Arc;
use core::fmt;

use crate::backtrace::Trace;
use crate::term::{atoms, Atom, OpaqueTerm, Term};

bitflags::bitflags! {
    /// These flags are used to by [`Process`] to track and control the behaviour
    /// of exceptions at various points during a program's execution.
    pub struct ExceptionFlags: u32 {
        /// Exception class is 'error'
        const IS_ERROR = 1;
        /// Exception class is 'exit'
        const IS_EXIT = 1 << 1;
        /// Exception class is 'throw'
        const IS_THROW = 1 << 2;
        /// Ignore catch handlers regardless of their presence
        const PANIC = 1 << 3;
        /// A non-local return
        const THROWN = 1 << 4;
        /// Write to logger on termination
        const LOG = 1 << 5;
        /// Occurred in native code
        const NATIVE = 1 << 6;
        /// Save the stack trace in internal form (i.e. not reified as a term)
        const SAVETRACE = 1 << 7;
        /// Restore original bif/nif
        const RESTORE_NIF = 1 << 8;
        /// Stack trace has arglist for top frame
        const ARGS = 1 << 9;
        /// Has extended error info
        const EXTENDED = 1 << 10;

        /// A bitmask for the exception class
        const CLASS_MASK = Self::IS_ERROR.bits | Self::IS_EXIT.bits | Self::IS_THROW.bits;
        /// A bitmask for the exception bits which are valid on primary exceptions
        const PRIMARY_MASK = Self::PANIC.bits | Self::THROWN.bits | Self::LOG.bits | Self::NATIVE.bits;
        /// Default flags for primary exceptions
        const PRIMARY = Self::SAVETRACE.bits;

        /// The default exception flags for an `error`
        const ERROR = Self::PRIMARY.bits | Self::IS_ERROR.bits | Self::LOG.bits;
        /// The default exception flags for an `error` with an arglist term
        const ERROR2 = Self::ERROR.bits | Self::ARGS.bits;
        /// The default exception flags for an `error` with an arglist term and extended info
        const ERROR3 = Self::ERROR.bits | Self::ARGS.bits | Self::EXTENDED.bits;
        /// The default exception flags for an `exit`
        const EXIT = Self::PRIMARY.bits | Self::IS_EXIT.bits;
        /// The default exception flags for a `throw`
        const THROW = Self::PRIMARY.bits | Self::IS_THROW.bits | Self::THROWN.bits;
    }
}
impl Default for ExceptionFlags {
    fn default() -> Self {
        Self::empty()
    }
}
impl ExceptionFlags {
    /// Stabilizes exception flags for primary exceptions
    pub fn to_primary(self) -> Self {
        self.intersection(Self::PRIMARY_MASK | Self::CLASS_MASK)
    }
}

/// This enumeration is used to represent a specific named error code of an exception
///
/// A few of these are built-in errors which have special behavior when handling errors.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ErrorCode {
    /// A primary error code is one which does not change once set
    Primary(Atom),
    CaseClause,
    TryClause,
    BadMatch,
    BadFun,
    BadArity,
    BadMap,
    BadKey,
    BadRecord,
    /// An error code of this type may be a built-in error, but has no special behavior
    Other(Atom),
}
impl ErrorCode {
    /// Lock in this error code as primary
    #[inline]
    pub fn to_primary(self) -> Self {
        Self::Primary(self.into())
    }
}
impl From<Atom> for ErrorCode {
    fn from(atom: Atom) -> Self {
        if atom == atoms::CaseClause {
            Self::CaseClause
        } else if atom == atoms::TryClause {
            Self::TryClause
        } else if atom == atoms::Badmatch {
            Self::BadMatch
        } else if atom == atoms::Badfun {
            Self::BadFun
        } else if atom == atoms::Badarity {
            Self::BadArity
        } else if atom == atoms::Badmap {
            Self::BadMap
        } else if atom == atoms::BadKey {
            Self::BadKey
        } else if atom == atoms::Badrecord {
            Self::BadRecord
        } else {
            Self::Other(atom)
        }
    }
}
impl Into<Atom> for ErrorCode {
    fn into(self) -> Atom {
        match self {
            Self::Primary(n) | Self::Other(n) => n,
            Self::CaseClause => atoms::CaseClause,
            Self::TryClause => atoms::TryClause,
            Self::BadMatch => atoms::Badmatch,
            Self::BadFun => atoms::Badfun,
            Self::BadArity => atoms::Badarity,
            Self::BadMap => atoms::Badmap,
            Self::BadKey => atoms::BadKey,
            Self::BadRecord => atoms::Badrecord,
        }
    }
}

#[derive(Copy, Clone)]
pub struct ExceptionCause {
    cause: OpaqueTerm,
    module: Atom,
    function: Atom,
}
impl fmt::Debug for ExceptionCause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExceptionCause")
            .field("cause", &format_args!("{}", self.cause))
            .field("module", &self.module)
            .field("function", &self.function)
            .finish()
    }
}
impl TryFrom<Term> for ExceptionCause {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Map(map) => {
                let Some(cause) = map.get(atoms::Cause) else { return Err(()); };
                let Some(module) = map.get(atoms::Module) else { return Err(()); };
                let Some(function) = map.get(atoms::Function) else { return Err(()); };
                if !module.is_atom() || !function.is_atom() {
                    return Err(());
                }
                Ok(Self {
                    cause,
                    module: module.as_atom(),
                    function: function.as_atom(),
                })
            }
            _ => Err(()),
        }
    }
}

/// Represents metadata about an active exception occurring in a process
pub struct ExceptionInfo {
    pub flags: ExceptionFlags,
    pub reason: ErrorCode,
    pub value: OpaqueTerm,
    pub args: Option<OpaqueTerm>,
    pub trace: Option<Arc<Trace>>,
    pub cause: Option<ExceptionCause>,
}
impl fmt::Debug for ExceptionInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExceptionInfo")
            .field("flags", &self.flags)
            .field("reason", &self.reason)
            .field("value", &format_args!("{}", self.value))
            .field(
                "args",
                &format_args!("{}", self.args.unwrap_or(OpaqueTerm::NIL)),
            )
            .field("cause", &self.cause)
            .finish()
    }
}
impl Default for ExceptionInfo {
    fn default() -> Self {
        Self {
            flags: ExceptionFlags::default(),
            reason: ErrorCode::Other(atoms::Undefined),
            value: OpaqueTerm::NIL,
            args: None,
            trace: None,
            cause: None,
        }
    }
}
impl ExceptionInfo {
    pub fn error(value: OpaqueTerm) -> Self {
        let reason = match value.into() {
            Term::Atom(a) => a.into(),
            Term::Tuple(tuple) => match tuple[0].into() {
                Term::Atom(a) => a.into(),
                _ => atoms::Error.into(),
            },
            _ => atoms::Error.into(),
        };
        Self {
            flags: ExceptionFlags::ERROR,
            reason,
            value,
            args: None,
            trace: None,
            cause: None,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    #[inline]
    pub fn is_error(&self) -> bool {
        self.flags.contains(ExceptionFlags::IS_ERROR)
    }

    #[inline]
    pub fn is_thrown(&self) -> bool {
        self.flags.contains(ExceptionFlags::IS_THROW)
    }

    #[inline]
    pub fn is_exit(&self) -> bool {
        self.flags.contains(ExceptionFlags::IS_EXIT)
    }

    pub fn class(&self) -> Option<ExceptionClass> {
        let class = self.flags & ExceptionFlags::CLASS_MASK;
        match class {
            ExceptionFlags::IS_ERROR => Some(ExceptionClass::Error),
            ExceptionFlags::IS_EXIT => Some(ExceptionClass::Exit),
            ExceptionFlags::IS_THROW => Some(ExceptionClass::Throw),
            _ => None,
        }
    }
}
