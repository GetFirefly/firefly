use firefly_diagnostics::{SourceSpan, Span, Spanned};
use firefly_syntax_base::{Deprecation, FunctionName};

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
impl Callback {
    pub fn is_impl(&self, name: &FunctionName) -> bool {
        if self.module.map(|id| id.name) != name.module || self.function.name != name.function {
            return false;
        }
        for sig in self.sigs.iter() {
            if name.arity as usize == sig.params.len() {
                return true;
            }
        }
        false
    }
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

/// Represents the set of allowed attributes in the body of a module
#[derive(Debug, Clone)]
pub enum Attribute {
    Type(TypeDef),
    Spec(TypeSpec),
    Callback(Callback),
    Custom(UserAttribute),
    ExportType(SourceSpan, Vec<Span<FunctionName>>),
    Export(SourceSpan, Vec<Span<FunctionName>>),
    Import(SourceSpan, Ident, Vec<Span<FunctionName>>),
    Removed(SourceSpan, Vec<(Span<FunctionName>, Ident)>),
    Compile(SourceSpan, Expr),
    Vsn(SourceSpan, Expr),
    Author(SourceSpan, Expr),
    OnLoad(SourceSpan, Span<FunctionName>),
    Nifs(SourceSpan, Vec<Span<FunctionName>>),
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
            Self::Deprecation(deprecations) => deprecations.first().map(|d| d.span()).unwrap(),
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
