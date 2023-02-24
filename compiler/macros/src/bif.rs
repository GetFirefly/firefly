use proc_macro::TokenStream;
use proc_macro2::Span;

use quote::{quote, ToTokens};
use syn::ext::IdentExt;
use syn::parse::{Error, Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    braced, parenthesized, token, Expr, Ident, LitInt, LitStr, Path, PathSegment, Result, Token,
};

use inflector::Inflector;

#[derive(Debug)]
pub struct MFA {
    module: Ident,
    _colon: Token![:],
    function: FunctionName,
    _slash: Token![/],
    arity: LitInt,
}
impl Parse for MFA {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        if !lookahead.peek(Ident) {
            return Err(lookahead.error());
        }
        let module = input.parse()?;
        let lookahead = input.lookahead1();
        if !lookahead.peek(Token![:]) {
            return Err(lookahead.error());
        }
        let _colon = input.parse()?;
        let function = input.parse()?;
        let lookahead = input.lookahead1();
        if !lookahead.peek(Token![/]) {
            return Err(lookahead.error());
        }
        let _slash: Token![/] = input.parse()?;
        let arity = input.parse()?;
        Ok(Self {
            module,
            _colon,
            function,
            _slash,
            arity,
        })
    }
}
impl ToTokens for MFA {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.module.to_tokens(tokens);
        self._colon.to_tokens(tokens);
        self.function.to_tokens(tokens);
        self._slash.to_tokens(tokens);
        self.arity.to_tokens(tokens);
    }
}

syn::custom_punctuation!(EqExact, =:=);
syn::custom_punctuation!(NeqExact, =/=);
syn::custom_punctuation!(Lte, =<);
syn::custom_punctuation!(Concat, ++);
syn::custom_punctuation!(Subtract, --);

#[derive(Debug)]
pub enum FunctionName {
    Ident(Ident),
    Add(Token![+]),
    Sub(Token![-]),
    Mul(Token![*]),
    Fdiv(Token![/]),
    EqExact(EqExact),
    Eq(Token![==]),
    NeqExact(NeqExact),
    Neq(Token![/=]),
    Gt(Token![>]),
    Gte(Token![>=]),
    Lt(Token![<]),
    Lte(Lte),
    Bang(Token![!]),
    Concat(Concat),
    Subtract(Subtract),
}
impl FunctionName {
    pub fn to_string(&self) -> String {
        match self {
            Self::Ident(ident) => ident.to_string(),
            Self::Add(_) => "+".to_string(),
            Self::Sub(_) => "-".to_string(),
            Self::Mul(_) => "*".to_string(),
            Self::Fdiv(_) => "/".to_string(),
            Self::EqExact(_) => "=:=".to_string(),
            Self::Eq(_) => "==".to_string(),
            Self::NeqExact(_) => "=/=".to_string(),
            Self::Neq(_) => "/=".to_string(),
            Self::Gt(_) => ">".to_string(),
            Self::Gte(_) => ">=".to_string(),
            Self::Lt(_) => "<".to_string(),
            Self::Lte(_) => "=<".to_string(),
            Self::Concat(_) => "++".to_string(),
            Self::Subtract(_) => "--".to_string(),
            Self::Bang(_) => "!".to_string(),
        }
    }

    pub fn span(&self) -> proc_macro2::Span {
        use syn::spanned::Spanned;
        match self {
            Self::Ident(ident) => ident.span(),
            Self::Add(tok) => tok.span(),
            Self::Sub(tok) => tok.span(),
            Self::Mul(tok) => tok.span(),
            Self::Fdiv(tok) => tok.span(),
            Self::EqExact(tok) => tok.span(),
            Self::Eq(tok) => tok.span(),
            Self::NeqExact(tok) => tok.span(),
            Self::Neq(tok) => tok.span(),
            Self::Gt(tok) => tok.span(),
            Self::Gte(tok) => tok.span(),
            Self::Lt(tok) => tok.span(),
            Self::Lte(tok) => tok.span(),
            Self::Concat(tok) => tok.span(),
            Self::Subtract(tok) => tok.span(),
            Self::Bang(tok) => tok.span(),
        }
    }
}
impl Parse for FunctionName {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead1 = input.lookahead1();
        let parsed = if lookahead1.peek(Concat) {
            Self::Concat(input.parse()?)
        } else if lookahead1.peek(Token![+]) {
            Self::Add(input.parse()?)
        } else if lookahead1.peek(Subtract) {
            Self::Subtract(input.parse()?)
        } else if lookahead1.peek(Token![-]) {
            Self::Sub(input.parse()?)
        } else if lookahead1.peek(Token![==]) {
            Self::Eq(input.parse()?)
        } else if lookahead1.peek(EqExact) {
            Self::EqExact(input.parse()?)
        } else if lookahead1.peek(NeqExact) {
            Self::NeqExact(input.parse()?)
        } else if lookahead1.peek(Lte) {
            Self::Lte(input.parse()?)
        } else if lookahead1.peek(Token![>=]) {
            Self::Gte(input.parse()?)
        } else if lookahead1.peek(Token![>]) {
            Self::Gt(input.parse()?)
        } else if lookahead1.peek(Token![/=]) {
            Self::Neq(input.parse()?)
        } else if lookahead1.peek(Token![/]) {
            Self::Fdiv(input.parse()?)
        } else if lookahead1.peek(Token![*]) {
            Self::Mul(input.parse()?)
        } else if lookahead1.peek(Token![<]) {
            Self::Lt(input.parse()?)
        } else if lookahead1.peek(Token![!]) {
            Self::Bang(input.parse()?)
        } else if lookahead1.peek(Ident::peek_any) {
            Self::Ident(input.call(Ident::parse_any)?)
        } else {
            return Err(lookahead1.error());
        };
        Ok(parsed)
    }
}
impl ToTokens for FunctionName {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Self::Ident(ident) => ident.to_tokens(tokens),
            Self::Add(tok) => tok.to_tokens(tokens),
            Self::Sub(tok) => tok.to_tokens(tokens),
            Self::Mul(tok) => tok.to_tokens(tokens),
            Self::Fdiv(tok) => tok.to_tokens(tokens),
            Self::EqExact(tok) => tok.to_tokens(tokens),
            Self::Eq(tok) => tok.to_tokens(tokens),
            Self::NeqExact(tok) => tok.to_tokens(tokens),
            Self::Neq(tok) => tok.to_tokens(tokens),
            Self::Gt(tok) => tok.to_tokens(tokens),
            Self::Gte(tok) => tok.to_tokens(tokens),
            Self::Lt(tok) => tok.to_tokens(tokens),
            Self::Lte(tok) => tok.to_tokens(tokens),
            Self::Concat(tok) => tok.to_tokens(tokens),
            Self::Subtract(tok) => tok.to_tokens(tokens),
            Self::Bang(tok) => tok.to_tokens(tokens),
        }
    }
}

pub struct BifSpec {
    span: Span,
    vis: Option<Visibility>,
    cc: Option<CallConv>,
    mfa: MFA,
    params: Vec<TypeSpec>,
    results: Vec<TypeSpec>,
}
impl Parse for BifSpec {
    fn parse(input: ParseStream) -> Result<Self> {
        let span = input.span();
        let vis = if input.fork().parse::<Visibility>().is_ok() {
            Some(input.parse::<Visibility>()?)
        } else {
            None
        };
        let cc = if input.fork().parse::<CallConv>().is_ok() {
            Some(input.parse::<CallConv>()?)
        } else {
            None
        };
        let mfa = input.parse()?;
        let content;
        let _paren: token::Paren = parenthesized!(content in input);
        let params: Punctuated<TypeSpec, Token![,]> = content.parse_terminated(TypeSpec::parse)?;

        let mut results = vec![];
        if input.peek(token::RArrow) {
            let _arrow: token::RArrow = input.parse()?;
            results = Punctuated::<TypeSpec, Token![,]>::parse_separated_nonempty(input)?
                .into_iter()
                .collect();
        }

        Ok(Self {
            span,
            vis,
            cc,
            mfa,
            params: params.into_iter().collect(),
            results,
        })
    }
}

pub enum TypeSpec {
    Unknown(Span),
    Never(Span),
    Named(Ident),
    Ptr(Span, Box<TypeSpec>),
    Array(Span, Box<TypeSpec>, usize),
    Struct(Span, Vec<TypeSpec>),
    Generic(Ident, Box<TypeSpec>),
    Function(Span, Vec<TypeSpec>, Vec<TypeSpec>),
}
impl TypeSpec {
    fn span(&self) -> Span {
        match self {
            Self::Unknown(span) => span.clone(),
            Self::Never(span) => span.clone(),
            Self::Named(id) => id.span(),
            Self::Ptr(span, _) => span.clone(),
            Self::Array(span, _, _) => span.clone(),
            Self::Struct(span, _) => span.clone(),
            Self::Generic(id, _) => id.span(),
            Self::Function(span, _, _) => span.clone(),
        }
    }
}
impl Parse for TypeSpec {
    fn parse(input: ParseStream) -> Result<Self> {
        let span = input.span();
        let lookahead = input.lookahead1();
        if lookahead.peek(token::Star) {
            // Parse pointer type
            let _: token::Star = input.parse()?;
            let lookahead = input.lookahead1();
            if lookahead.peek(token::Const) {
                let _: token::Const = input.parse()?;
            } else if lookahead.peek(token::Mut) {
                let _: token::Mut = input.parse()?;
            } else {
                return Err(lookahead.error());
            }
            let pointee: Self = input.parse()?;
            Ok(Self::Ptr(span, Box::new(pointee)))
        } else if lookahead.peek(token::Bang) {
            let _: token::Bang = input.parse()?;
            Ok(Self::Never(span))
        } else if lookahead.peek(token::Question) {
            let _: token::Question = input.parse()?;
            Ok(Self::Unknown(span))
        } else if lookahead.peek(token::Brace) {
            // Parse struct type
            let content;
            let _: token::Brace = braced!(content in input);
            let fields: Punctuated<TypeSpec, Token![,]> = content.parse_terminated(Self::parse)?;
            Ok(Self::Struct(span, fields.into_iter().collect()))
        } else if lookahead.peek(token::Bracket) {
            // Parse array type
            let elem: Self = input.parse()?;
            let _: Token![;] = input.parse()?;
            let arity: syn::LitInt = input.parse()?;
            let arity = arity.base10_parse()?;
            Ok(Self::Array(span, Box::new(elem), arity))
        } else if lookahead.peek(token::Fn) {
            // Parse function type
            let _: token::Fn = input.parse()?;
            let params_content;
            let _: token::Paren = parenthesized!(params_content in input);
            let params: Punctuated<TypeSpec, Token![,]> =
                params_content.parse_terminated(Self::parse)?;
            let _: token::RArrow = input.parse()?;
            let results_content;
            let _: token::Paren = parenthesized!(results_content in input);
            let results: Punctuated<TypeSpec, Token![,]> =
                results_content.parse_terminated(Self::parse)?;
            let params = params.into_iter().collect();
            let results = results.into_iter().collect();
            Ok(Self::Function(span, params, results))
        } else if lookahead.peek(Ident::peek_any) {
            // Parse named/generic
            let name: Ident = input.parse()?;
            let lookahead = input.lookahead1();
            if lookahead.peek(token::Lt) {
                let inner: Self = input.parse()?;
                Ok(Self::Generic(name, Box::new(inner)))
            } else {
                Ok(Self::Named(name))
            }
        } else {
            Err(lookahead.error())
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
enum Visibility {
    /// Public functions are callable by anyone
    Public,
    /// Private functions are limited to the defining module
    #[default]
    Private,
    /// Guards are always public
    Guard,
}
impl Parse for Visibility {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(Token![pub]) {
            let _pub: Token![pub] = input.parse()?;
            Ok(Visibility::Public)
        } else if lookahead.peek(Ident) {
            let id: Ident = input.parse()?;
            if id == "guard" {
                Ok(Visibility::Guard)
            } else {
                Err(Error::new(
                    id.span(),
                    format!("expected `guard`, but got `{}`", &id),
                ))
            }
        } else {
            Err(lookahead.error())
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
enum CallConv {
    #[default]
    Erlang,
    ErlangAsync,
    C,
}
impl Parse for CallConv {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(Token![async]) {
            let _async: Token![async] = input.parse()?;
            Ok(CallConv::ErlangAsync)
        } else if lookahead.peek(Token![extern]) && input.peek2(LitStr) {
            let _extern: Token![extern] = input.parse()?;
            let s: LitStr = input.parse()?;
            let cc = s.value();
            match cc.as_str() {
                "C" => Ok(CallConv::C),
                "Erlang" => Ok(CallConv::Erlang),
                "ErlangAsync" => Ok(CallConv::ErlangAsync),
                other => Err(Error::new(
                    s.span(),
                    format!("expected valid calling convention, got \"{}\"", other),
                )),
            }
        } else {
            Err(lookahead.error())
        }
    }
}

pub fn define_bif(mut spec: BifSpec) -> Result<TokenStream> {
    let visibility = match spec.vis.unwrap_or_default() {
        Visibility::Guard => {
            quote!(
                crate::Visibility::PUBLIC
                    | crate::Visibility::GUARD
                    | crate::Visibility::IMPORTED
                    | crate::Visibility::EXTERNAL
            )
        }
        Visibility::Public => {
            quote!(
                crate::Visibility::PUBLIC
                    | crate::Visibility::IMPORTED
                    | crate::Visibility::EXTERNAL
            )
        }
        Visibility::Private => quote!(
            crate::Visibility::PRIVATE | crate::Visibility::IMPORTED | crate::Visibility::EXTERNAL
        ),
    };
    let cc = spec.cc.unwrap_or_default();
    let callconv = match cc {
        CallConv::Erlang => quote!(crate::CallConv::Erlang),
        CallConv::ErlangAsync => quote!(crate::CallConv::ErlangAsync),
        CallConv::C => quote!(crate::CallConv::C),
    };
    let module = if spec.mfa.module == "erlang" {
        quote!(firefly_intern::symbols::Erlang)
    } else {
        let m = spec.mfa.module;
        quote!(firefly_intern::Symbol::intern(stringify!(#m)))
    };
    let name = spec.mfa.function.to_string();
    let name_lit = LitStr::new(name.as_str(), spec.mfa.function.span());
    let params = spec
        .params
        .drain(..)
        .map(type_to_type_expr)
        .try_collect::<Vec<_>>()?;
    let params = to_vec_expr(spec.span.clone(), params);
    let results = spec
        .results
        .drain(..)
        .map(type_to_type_expr)
        .try_collect::<Vec<_>>()?;
    let results = to_vec_expr(spec.span.clone(), results);
    let quoted = quote! {
        crate::Signature {
            visibility: #visibility,
            cc: #callconv,
            module: #module,
            name: firefly_intern::Symbol::intern(#name_lit),
            ty: crate::FunctionType::new(#params, #results),
        }
    };
    Ok(TokenStream::from(quoted))
}

fn type_to_type_expr(ty: TypeSpec) -> Result<Expr> {
    match ty {
        TypeSpec::Unknown(span) => Ok(type_variant(span, "Unknown", None)),
        TypeSpec::Never(span) => Ok(type_variant(span, "Never", None)),
        TypeSpec::Named(id) => {
            let span = id.span();
            let name = format!("{}", &id);
            if let Some(variant) = get_primitive_type(span, name.as_str()) {
                return Ok(variant);
            }
            if let Some(variant) = get_term_type(span, name.as_str()) {
                Ok(variant)
            } else {
                Ok(type_variant(span, name.to_class_case(), None))
            }
        }
        TypeSpec::Ptr(span, box pointee) => {
            let pointee = type_to_primitive_type_expr(pointee)?;
            let boxed = enum_variant(
                span.clone(),
                to_path(span.clone(), &["std", "boxed", "Box", "new"]),
                Some(vec![pointee]),
            );
            Ok(primitive_type(span, "Ptr", Some(vec![boxed])))
        }
        TypeSpec::Array(span, box elem, len) => {
            let elem = type_to_primitive_type_expr(elem)?;
            let len = to_lit_int(span.clone(), len);
            let boxed = enum_variant(
                span.clone(),
                to_path(span.clone(), &["std", "boxed", "Box"]),
                Some(vec![elem]),
            );
            Ok(primitive_type(span, "Array", Some(vec![boxed, len])))
        }
        TypeSpec::Struct(span, mut fields) => {
            let fields = fields
                .drain(..)
                .map(type_to_primitive_type_expr)
                .try_collect()?;
            Ok(primitive_type(
                span,
                "Struct",
                Some(vec![to_vec_expr(span.clone(), fields)]),
            ))
        }
        TypeSpec::Generic(id, box inner) => {
            let inner = type_to_type_expr(inner)?;
            let name = format!("{}", &id);
            let name = name.to_class_case();
            Ok(type_variant(id.span(), name, Some(vec![inner])))
        }
        TypeSpec::Function(span, mut params, mut results) => {
            let params = params.drain(..).map(type_to_type_expr).try_collect()?;
            let results = results.drain(..).map(type_to_type_expr).try_collect()?;
            let fun = enum_variant(
                span.clone(),
                to_path(span.clone(), &["crate", "FunctionType", "new"]),
                Some(vec![
                    to_vec_expr(span.clone(), params),
                    to_vec_expr(span.clone(), results),
                ]),
            );
            Ok(type_variant(span, "Function", Some(vec![fun])))
        }
    }
}

fn type_to_primitive_type_expr(ty: TypeSpec) -> Result<Expr> {
    match ty {
        TypeSpec::Named(id) => {
            let span = id.span();
            let name = format!("{}", &id);
            get_primitive_type(span, name.as_str())
                .ok_or_else(|| Error::new(span, "expected a primitive type"))
        }
        ty @ (TypeSpec::Ptr(_, _) | TypeSpec::Array(_, _, _) | TypeSpec::Struct(_, _)) => {
            type_to_type_expr(ty)
        }
        other => Err(Error::new(other.span(), "expected a primitive type")),
    }
}

fn get_primitive_type(span: Span, name: &str) -> Option<Expr> {
    match name {
        "i1" | "i8" | "i16" | "i32" | "i64" | "isize" | "f64" => {
            let name = name.to_class_case();
            let path = to_path(span.clone(), &["crate", "PrimitiveType", name.as_str()]);
            Some(enum_variant(span, path, None))
        }
        _ => None,
    }
}

fn get_term_type(span: Span, name: &str) -> Option<Expr> {
    match name {
        "any" | "term" | "timeout" => Some(term_type(span, "Any", None)),
        "atom" | "module" | "node" => Some(term_type(span, "Atom", None)),
        "binary" => Some(term_type(span, "Binary", None)),
        "bitstring" => Some(term_type(span, "Bitstring", None)),
        "bool" | "boolean" => Some(term_type(span, "Bool", None)),
        "number" => Some(term_type(span, "Number", None)),
        "float" => Some(term_type(span, "Float", None)),
        "integer" | "neg_integer" | "non_neg_integer" | "pos_integer" | "arity" | "byte"
        | "char" => Some(term_type(span, "Integer", None)),
        "nil" => Some(term_type(span, "Nil", None)),
        "tuple" => {
            let inner = enum_variant(span.clone(), to_path(span.clone(), &["None"]), None);
            Some(term_type(span, "Tuple", Some(vec![inner])))
        }
        "nonempty_list" | "nonempty_string" => Some(term_type(span, "Cons", None)),
        "list" | "string" | "iovec" => {
            let inner = enum_variant(span.clone(), to_path(span.clone(), &["None"]), None);
            Some(term_type(span, "List", Some(vec![inner])))
        }
        "maybe_improper_list"
        | "nonempty_improper_list"
        | "nonempty_maybe_improper_list"
        | "iolist" => Some(term_type(span, "MaybeImproperList", None)),
        "map" => Some(term_type(span, "Map", None)),
        "function" => Some(term_type(
            span,
            "Fun",
            Some(vec![enum_variant(
                span.clone(),
                to_path(span.clone(), &["None"]),
                None,
            )]),
        )),
        "mfa" => {
            let atom1 = term_type_variant(span.clone(), "Atom", None);
            let atom2 = atom1.clone();
            let integer = term_type_variant(span.clone(), "Integer", None);
            Some(term_type(
                span,
                "Tuple",
                Some(vec![to_opt_vec_expr(
                    span.clone(),
                    vec![atom1, atom2, integer],
                )]),
            ))
        }
        "time" | "timestamp" => {
            let int1 = term_type_variant(span.clone(), "Integer", None);
            let int2 = int1.clone();
            let int3 = int1.clone();
            Some(term_type(
                span,
                "Tuple",
                Some(vec![to_opt_vec_expr(span.clone(), vec![int1, int2, int3])]),
            ))
        }
        "pid" => Some(term_type(span, "Pid", None)),
        "port" => Some(term_type(span, "Port", None)),
        "reference" => Some(term_type(span, "Reference", None)),
        "no_return" => Some(type_variant(span, "Never", None)),
        "exception" => Some(type_variant(span, "Exception", None)),
        "trace" => Some(type_variant(span, "ExceptionTrace", None)),
        "none" => Some(type_variant(span, "Unit", None)),
        "spawn_monitor" => {
            let pid = term_type_variant(span.clone(), "Pid", None);
            let reference = term_type_variant(span.clone(), "Reference", None);
            Some(term_type(
                span,
                "Tuple",
                Some(vec![to_opt_vec_expr(span.clone(), vec![pid, reference])]),
            ))
        }
        "binary_split" => {
            let bin1 = term_type_variant(span.clone(), "Pid", None);
            let bin2 = bin1.clone();
            Some(term_type(
                span,
                "Tuple",
                Some(vec![to_opt_vec_expr(span.clone(), vec![bin1, bin2])]),
            ))
        }
        _ => None,
    }
}

fn enum_variant(span: Span, path: Path, params: Option<Vec<Expr>>) -> Expr {
    match params {
        None => Expr::Path(syn::ExprPath {
            attrs: vec![],
            qself: None,
            path,
        }),
        Some(mut exprs) => Expr::Call(syn::ExprCall {
            attrs: vec![],
            func: Box::new(Expr::Path(syn::ExprPath {
                attrs: vec![],
                qself: None,
                path,
            })),
            paren_token: token::Paren { span: span.clone() },
            args: exprs.drain(..).collect(),
        }),
    }
}

#[inline]
fn type_variant<S: AsRef<str>>(span: Span, tag: S, params: Option<Vec<Expr>>) -> Expr {
    let path = to_path(span.clone(), &["crate", "Type", tag.as_ref()]);
    enum_variant(span, path, params)
}

fn term_type<S: AsRef<str>>(span: Span, tag: S, params: Option<Vec<Expr>>) -> Expr {
    let inner = term_type_variant(span.clone(), tag, params);
    type_variant(span, "Term", Some(vec![inner]))
}

fn term_type_variant<S: AsRef<str>>(span: Span, tag: S, params: Option<Vec<Expr>>) -> Expr {
    let path = to_path(span.clone(), &["crate", "TermType", tag.as_ref()]);
    enum_variant(span.clone(), path, params)
}

fn primitive_type<S: AsRef<str>>(span: Span, tag: S, params: Option<Vec<Expr>>) -> Expr {
    let inner = primitive_type_variant(span.clone(), tag, params);
    type_variant(span, "Primitive", Some(vec![inner]))
}

fn primitive_type_variant<S: AsRef<str>>(span: Span, tag: S, params: Option<Vec<Expr>>) -> Expr {
    let path = to_path(span.clone(), &["crate", "PrimitiveType", tag.as_ref()]);
    enum_variant(span.clone(), path, params)
}

fn to_opt_vec_expr(span: Span, elements: Vec<Expr>) -> Expr {
    if elements.is_empty() {
        enum_variant(span, to_path(span.clone(), &["None"]), None)
    } else {
        enum_variant(
            span,
            to_path(span.clone(), &["Some"]),
            Some(vec![to_vec_expr(span.clone(), elements)]),
        )
    }
}

fn to_vec_expr(span: Span, mut elements: Vec<Expr>) -> Expr {
    let elements = elements.drain(..).collect::<Punctuated<Expr, Token![,]>>();
    let mac = syn::Macro {
        path: to_path(span.clone(), &["vec"]),
        bang_token: token::Bang {
            spans: [span.clone()],
        },
        delimiter: syn::MacroDelimiter::Brace(token::Brace { span: span.clone() }),
        tokens: elements.into_token_stream(),
    };
    Expr::Macro(syn::ExprMacro { attrs: vec![], mac })
}

#[inline]
fn to_path(span: Span, segments: &[&str]) -> Path {
    Path {
        leading_colon: None,
        segments: segments
            .iter()
            .copied()
            .map(|s| to_path_segment(span.clone(), s))
            .collect(),
    }
}

#[inline]
fn to_path_segment(span: Span, name: &str) -> PathSegment {
    PathSegment {
        ident: Ident::new(name, span),
        arguments: syn::PathArguments::None,
    }
}

#[inline]
fn to_lit_int(span: Span, value: usize) -> Expr {
    let repr = format!("{}", value);
    Expr::Lit(syn::ExprLit {
        attrs: vec![],
        lit: syn::Lit::Int(syn::LitInt::new(repr.as_str(), span)),
    })
}
