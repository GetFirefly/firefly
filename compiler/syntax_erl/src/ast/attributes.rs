use std::fmt;
use std::hash::{Hash, Hasher};

use liblumen_diagnostics::{SourceSpan, Span, Spanned};
use liblumen_syntax_core as syntax_core;

use super::{Expr, Ident, Name, Type};

/// Type definitions
///
/// ## Examples
///
/// ```text
/// %% Simple types
/// -type foo() :: bar.
/// -opaque foo() :: bar.
///
/// %% Generic/parameterized types
/// -type foo(T) :: [T].
/// -opaque foo(T) :: [T].
/// ```
#[derive(Debug, Clone, Spanned)]
pub struct TypeDef {
    #[span]
    pub span: SourceSpan,
    pub opaque: bool,
    pub name: Ident,
    pub params: Vec<Name>,
    pub ty: Type,
}
impl PartialEq for TypeDef {
    fn eq(&self, other: &Self) -> bool {
        if self.opaque != other.opaque {
            return false;
        }
        if self.name != other.name {
            return false;
        }
        if self.params != other.params {
            return false;
        }
        if self.ty != other.ty {
            return false;
        }
        return true;
    }
}

/// Function type specifications, used for both function specs and callback specs
///
/// ## Example
///
/// ```text
/// %% Monomorphic function
/// -spec foo(A :: map(), Opts :: list({atom(), term()})) -> {ok, map()} | {error, term()}.
///
/// %% Polymorphic function
/// -spec foo(A, Opts :: list({atom, term()})) -> {ok, A} | {error, term()}.
///
/// %% Multiple dispatch function
/// -spec foo(map(), Opts) -> {ok, map()} | {error, term()};
///   foo(list(), Opts) -> {ok, list()} | {error, term()}.
///
/// %% Using `when` to express subtype constraints
/// -spec foo(map(), Opts) -> {ok, map()} | {error, term()}
///   when Opts :: list({atom, term});
/// ```
#[derive(Debug, Clone, Spanned)]
pub struct TypeSpec {
    #[span]
    pub span: SourceSpan,
    pub module: Option<Ident>,
    pub function: Ident,
    pub sigs: Vec<TypeSig>,
}
impl PartialEq for TypeSpec {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.function == other.function && self.sigs == other.sigs
    }
}

/// A callback declaration, which is functionally identical to `TypeSpec` in
/// its syntax, but is used to both define a callback function for a behaviour,
/// as well as provide an expected type specification for that function.
#[derive(Debug, Clone, Spanned)]
pub struct Callback {
    #[span]
    pub span: SourceSpan,
    pub optional: bool,
    pub module: Option<Ident>,
    pub function: Ident,
    pub sigs: Vec<TypeSig>,
}
impl PartialEq for Callback {
    fn eq(&self, other: &Self) -> bool {
        self.optional == other.optional
            && self.module == other.module
            && self.function == other.function
            && self.sigs == other.sigs
    }
}

/// Contains type information for a single clause of a function type specification
#[derive(Debug, Clone, PartialEq, Spanned)]
pub struct TypeSig {
    #[span]
    pub span: SourceSpan,
    pub params: Vec<Type>,
    pub ret: Box<Type>,
    pub guards: Option<Vec<TypeGuard>>,
}

/// Contains a single subtype constraint to be applied to a type specification
#[derive(Debug, Clone, Spanned)]
pub struct TypeGuard {
    #[span]
    pub span: SourceSpan,
    pub var: Name,
    pub ty: Type,
}
impl PartialEq for TypeGuard {
    fn eq(&self, other: &TypeGuard) -> bool {
        self.var == other.var && self.ty == other.ty
    }
}

/// Represents a user-defined custom attribute.
///
/// ## Example
///
/// ```text
/// -my_attribute([foo, bar]).
/// ```
#[derive(Debug, Clone, Spanned)]
pub struct UserAttribute {
    #[span]
    pub span: SourceSpan,
    pub name: Ident,
    pub value: Expr,
}
impl PartialEq for UserAttribute {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.value == other.value
    }
}

/// Represents a deprecated function or module
#[derive(Debug, Clone, Spanned)]
pub enum Deprecation {
    Module {
        #[span]
        span: SourceSpan,
        flag: DeprecatedFlag,
    },
    Function {
        #[span]
        span: SourceSpan,
        function: Span<syntax_core::FunctionName>,
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
            Self::Module { .. } => (),
            Self::Function { ref function, .. } => function.hash(state),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Represents the set of allowed attributes in the body of a module
#[derive(Debug, Clone)]
pub enum Attribute {
    Type(TypeDef),
    Spec(TypeSpec),
    Callback(Callback),
    Custom(UserAttribute),
    ExportType(SourceSpan, Vec<Span<syntax_core::FunctionName>>),
    Export(SourceSpan, Vec<Span<syntax_core::FunctionName>>),
    Import(SourceSpan, Ident, Vec<Span<syntax_core::FunctionName>>),
    Removed(SourceSpan, Vec<(Span<syntax_core::FunctionName>, Ident)>),
    Compile(SourceSpan, Expr),
    Vsn(SourceSpan, Expr),
    Author(SourceSpan, Expr),
    OnLoad(SourceSpan, Span<syntax_core::FunctionName>),
    Nifs(SourceSpan, Vec<Span<syntax_core::FunctionName>>),
    Behaviour(SourceSpan, Ident),
    Deprecation(Vec<Deprecation>),
}
impl Spanned for Attribute {
    fn span(&self) -> SourceSpan {
        match self {
            Self::Type(attr) => attr.span(),
            Self::Spec(attr) => attr.span(),
            Self::Callback(attr) => attr.span(),
            Self::Custom(attr) => attr.span(),
            Self::ExportType(span, _)
            | Self::Export(span, _)
            | Self::Import(span, _, _)
            | Self::Removed(span, _)
            | Self::Compile(span, _)
            | Self::Vsn(span, _)
            | Self::Author(span, _)
            | Self::OnLoad(span, _)
            | Self::Nifs(span, _)
            | Self::Behaviour(span, _) => *span,
            Self::Deprecation(deprecations) => {
                if let Some(d) = deprecations.first() {
                    d.span()
                } else {
                    SourceSpan::UNKNOWN
                }
            }
        }
    }
}
impl PartialEq for Attribute {
    fn eq(&self, other: &Self) -> bool {
        let left = std::mem::discriminant(self);
        let right = std::mem::discriminant(other);
        if left != right {
            return false;
        }

        match (self, other) {
            (&Attribute::Type(ref x), &Attribute::Type(ref y)) => x == y,
            (&Attribute::Spec(ref x), &Attribute::Spec(ref y)) => x == y,
            (&Attribute::Callback(ref x), &Attribute::Callback(ref y)) => x == y,
            (&Attribute::Custom(ref x), &Attribute::Custom(ref y)) => x == y,
            (&Attribute::ExportType(_, ref x), &Attribute::ExportType(_, ref y)) => x == y,
            (&Attribute::Export(_, ref x), &Attribute::Export(_, ref y)) => x == y,
            (&Attribute::Import(_, ref x1, ref x2), &Attribute::Import(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            (&Attribute::Removed(_, ref x), &Attribute::Removed(_, ref y)) => x == y,
            (&Attribute::Compile(_, ref x), &Attribute::Compile(_, ref y)) => x == y,
            (&Attribute::Vsn(_, ref x), &Attribute::Vsn(_, ref y)) => x == y,
            (&Attribute::Author(_, ref x), &Attribute::Author(_, ref y)) => x == y,
            (&Attribute::OnLoad(_, ref x), &Attribute::OnLoad(_, ref y)) => x == y,
            (&Attribute::Nifs(_, ref x), &Attribute::Nifs(_, ref y)) => x == y,
            (&Attribute::Behaviour(_, ref x), &Attribute::Behaviour(_, ref y)) => x == y,
            _ => false,
        }
    }
}
