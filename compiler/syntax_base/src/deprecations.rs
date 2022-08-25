use core::fmt;
use core::hash::{Hash, Hasher};

use liblumen_diagnostics::{SourceSpan, Span, Spanned};
use liblumen_intern::Ident;

use crate::FunctionName;

/// Represents a deprecated function or module
#[derive(Debug, Copy, Clone, Spanned)]
pub enum Deprecation {
    Module {
        #[span]
        span: SourceSpan,
        flag: DeprecatedFlag,
    },
    Function {
        #[span]
        span: SourceSpan,
        function: Span<FunctionName>,
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
