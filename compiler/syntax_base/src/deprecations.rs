use core::fmt;
use core::hash::{Hash, Hasher};

use firefly_diagnostics::{SourceSpan, Span, Spanned};
use firefly_intern::{symbols, Ident, Symbol};

use crate::FunctionName;

/// Represents a deprecated function or module
#[derive(Debug, Copy, Clone, Spanned)]
pub enum Deprecation {
    /// Represents deprecation of an entire module
    Module {
        #[span]
        span: SourceSpan,
        flag: DeprecatedFlag,
    },
    /// Represents deprecation of a specific function
    Function {
        #[span]
        span: SourceSpan,
        function: Span<FunctionName>,
        flag: DeprecatedFlag,
    },
    /// Represents deprecation of a function `name` of any arity
    FunctionAnyArity {
        #[span]
        span: SourceSpan,
        name: Symbol,
        flag: DeprecatedFlag,
    },
}
impl PartialEq for Deprecation {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Module { .. }, Self::Module { .. }) => true,
            // We ignore the flag because it used only for display,
            // the function/arity determines equality
            (
                Self::Function {
                    function: ref x1, ..
                },
                Self::Function {
                    function: ref y1, ..
                },
            ) => x1 == y1,
            (Self::FunctionAnyArity { name: x, .. }, Self::FunctionAnyArity { name: y, .. }) => {
                x == y
            }
            _ => false,
        }
    }
}
impl Eq for Deprecation {}
impl Hash for Deprecation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let discriminant = std::mem::discriminant(self);
        discriminant.hash(state);
        match self {
            Self::Module { flag, .. } => flag.hash(state),
            Self::Function {
                ref function, flag, ..
            } => {
                flag.hash(state);
                function.hash(state)
            }
            Self::FunctionAnyArity { name, flag, .. } => {
                flag.hash(state);
                name.hash(state)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeprecatedFlag {
    Eventually,
    NextVersion,
    NextMajorRelease,
    Description(Ident),
}
impl From<Ident> for DeprecatedFlag {
    fn from(ident: Ident) -> Self {
        match ident.name {
            symbols::Eventually => Self::Eventually,
            symbols::NextVersion => Self::NextVersion,
            symbols::NextMajorRelease => Self::NextMajorRelease,
            _ => Self::Description(ident),
        }
    }
}
impl fmt::Display for DeprecatedFlag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Eventually => write!(f, "eventually"),
            Self::NextVersion => write!(f, "in the next version"),
            Self::NextMajorRelease => write!(f, "in the next major release"),
            Self::Description(descr) => write!(f, "{}", descr.name),
        }
    }
}
