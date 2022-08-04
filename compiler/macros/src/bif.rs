use proc_macro::TokenStream;
use proc_macro2::Span;

use quote::{quote, quote_spanned, ToTokens};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::{Pair, Punctuated};
use syn::{
    parenthesized, parse_quote, token, Expr, Ident, LitInt, LitStr, Path, PathSegment, Result,
    Token,
};

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
        }
    }
}

pub struct BifSpec {
    vis: Option<token::Pub>,
    pub mfa: MFA,
    _paren: token::Paren,
    pub params: Punctuated<Ident, Token![,]>,
    _arrow: Token![->],
    pub result_ty: Ident,
}
impl Parse for BifSpec {
    fn parse(input: ParseStream) -> Result<Self> {
        let vis = input.parse()?;
        let mfa = input.parse()?;
        let content;
        let _paren = parenthesized!(content in input);
        let params = content.parse_terminated(Ident::parse_any)?;
        let _arrow = input.parse()?;
        let result_ty = input.parse()?;

        Ok(Self {
            vis,
            mfa,
            _paren,
            params,
            _arrow,
            result_ty,
        })
    }
}

pub fn define_bif(spec: BifSpec) -> TokenStream {
    define_bif_internal(spec, false)
}

pub fn define_guard_bif(spec: BifSpec) -> TokenStream {
    define_bif_internal(spec, true)
}

fn define_bif_internal(spec: BifSpec, is_guard: bool) -> TokenStream {
    let visibility = match spec.vis {
        Some(_) if is_guard => {
            quote!(
                crate::Visibility::PUBLIC
                    | crate::Visibility::GUARD
                    | crate::Visibility::IMPORTED
                    | crate::Visibility::EXTERNAL
            )
        }
        Some(_) => {
            quote!(
                crate::Visibility::PUBLIC
                    | crate::Visibility::IMPORTED
                    | crate::Visibility::EXTERNAL
            )
        }
        _ if is_guard => quote!(
            crate::Visibility::PRIVATE
                | crate::Visibility::GUARD
                | crate::Visibility::IMPORTED
                | crate::Visibility::EXTERNAL
        ),
        _ => quote!(
            crate::Visibility::PRIVATE | crate::Visibility::IMPORTED | crate::Visibility::EXTERNAL
        ),
    };
    let module = if spec.mfa.module == "erlang" {
        quote!(liblumen_intern::symbols::Erlang)
    } else {
        let m = spec.mfa.module;
        quote!(liblumen_intern::Symbol::intern(stringify!(#m)))
    };
    let params = spec
        .params
        .into_pairs()
        .map(erlang_types_to_core_type_enum)
        .collect::<Punctuated<Expr, Token![,]>>();
    let name = spec.mfa.function.to_string();
    let name_lit = LitStr::new(name.as_str(), spec.mfa.function.span());
    let result_ty = erlang_type_to_core_type_enum(spec.result_ty);
    let quoted = quote! {
        crate::Signature {
            visibility: #visibility,
            cc: crate::CallConv::Erlang,
            module: #module,
            name: liblumen_intern::Symbol::intern(#name_lit),
            params: vec![#params],
            results: vec![crate::Type::Primitive(crate::PrimitiveType::I1), #result_ty],
        }
    };
    TokenStream::from(quoted)
}

/// Convert some common Erlang type names to their equivalent representation as a liblumen_syntax_core::Type variant
fn erlang_types_to_core_type_enum(pair: Pair<Ident, Token![,]>) -> Pair<Expr, Token![,]> {
    let (ident, punct) = pair.into_tuple();
    let expr = erlang_type_to_core_type_enum(ident.clone());
    Pair::new(expr, punct)
}

fn erlang_type_to_core_type_enum(ident: Ident) -> Expr {
    let span = ident.span();
    let name = ident.to_string();
    match name.as_str() {
        "any" | "term" | "timeout" => core_enum_variant("Any", span),
        "atom" | "module" | "node" => core_enum_variant("Atom", span),
        "binary" => core_enum_variant("Binary", span),
        "bitstring" => core_enum_variant("Bitstring", span),
        "bool" | "boolean" => core_enum_variant("Bool", span),
        "float" => core_enum_variant("Float", span),
        "number" => core_enum_variant("Number", span),
        "integer" | "neg_integer" | "non_neg_integer" | "pos_integer" | "arity" | "byte"
        | "char" => core_enum_variant("Integer", span),
        "function" => core_enum_fun_variant(span),
        "nil" => core_enum_variant("Nil", span),
        "tuple" => core_enum_tuple_variant(None, span),
        "list" | "nonempty_list" | "string" | "nonempty_string" | "iovec" => {
            core_enum_list_variant(span)
        }
        "maybe_improper_list"
        | "nonempty_improper_list"
        | "nonempty_maybe_improper_list"
        | "iolist" => core_enum_variant("MaybeImproperList", span),
        "map" => core_enum_variant("Map", span),
        "mfa" => core_enum_tuple_variant(Some(&["Atom", "Atom", "Integer"]), span),
        "time" | "timestamp" => {
            core_enum_tuple_variant(Some(&["Integer", "Integer", "Integer"]), span)
        }
        "pid" => core_enum_variant("Pid", span),
        "port" => core_enum_variant("Port", span),
        "reference" => core_enum_variant("Reference", span),
        "no_return" => core_special_variant("NoReturn", span),
        "none" => core_special_variant("Invalid", span),
        "spawn_monitor" => core_enum_tuple_variant(Some(&["Pid", "Reference"]), span),
        "binary_split" => core_enum_tuple_variant(Some(&["Binary", "Binary"]), span),
        // Anything else is either inherently polymorphic, or unrecognized, so we just say Term
        _ => core_enum_variant("Any", span),
    }
}

/// Convert the given string to a path representing the instantiation of a liblumen_syntax_core::Type variant
fn core_enum_variant(name: &str, span: Span) -> Expr {
    use syn::PathArguments;

    let mut path = Path {
        leading_colon: None,
        segments: Punctuated::new(),
    };
    path.segments.push(PathSegment {
        ident: Ident::new("crate", Span::call_site()),
        arguments: PathArguments::None,
    });
    path.segments.push(PathSegment {
        ident: Ident::new("TermType", Span::call_site()),
        arguments: PathArguments::None,
    });
    path.segments.push(PathSegment {
        ident: Ident::new(name, span),
        arguments: PathArguments::None,
    });

    let quoted = quote_spanned! { span =>
      crate::Type::Term(#path)
    };

    parse_quote! { #quoted }
}

fn core_term_variant(name: &str, span: Span) -> Expr {
    use syn::PathArguments;

    let mut path = Path {
        leading_colon: None,
        segments: Punctuated::new(),
    };
    path.segments.push(PathSegment {
        ident: Ident::new("crate", Span::call_site()),
        arguments: PathArguments::None,
    });
    path.segments.push(PathSegment {
        ident: Ident::new("TermType", Span::call_site()),
        arguments: PathArguments::None,
    });
    path.segments.push(PathSegment {
        ident: Ident::new(name, span),
        arguments: PathArguments::None,
    });

    parse_quote! { #path }
}

fn core_special_variant(name: &str, span: Span) -> Expr {
    use syn::PathArguments;

    let mut path = Path {
        leading_colon: None,
        segments: Punctuated::new(),
    };
    path.segments.push(PathSegment {
        ident: Ident::new("crate", Span::call_site()),
        arguments: PathArguments::None,
    });
    path.segments.push(PathSegment {
        ident: Ident::new("Type", Span::call_site()),
        arguments: PathArguments::None,
    });
    path.segments.push(PathSegment {
        ident: Ident::new(name, span),
        arguments: PathArguments::None,
    });

    parse_quote! { #path }
}

fn core_enum_fun_variant(span: Span) -> Expr {
    let quoted = quote_spanned! { span => crate::Type::Term(crate::TermType::Fun(None)) };
    parse_quote!(#quoted)
}

fn core_enum_list_variant(span: Span) -> Expr {
    let quoted = quote_spanned! { span => crate::Type::Term(crate::TermType::List(None)) };
    parse_quote!(#quoted)
}

fn core_enum_tuple_variant(elements: Option<&[&str]>, span: Span) -> Expr {
    let quoted = if elements.is_none() {
        quote_spanned! { span =>
            crate::Type::Term(crate::TermType::Tuple(None))
        }
    } else {
        let elements_punctuated = elements
            .unwrap()
            .iter()
            .map(|e| core_term_variant(e, span))
            .collect::<Punctuated<Expr, Token![,]>>();
        quote_spanned! { span =>
            crate::Type::Term(crate::TermType::Tuple(Some(vec![#elements_punctuated])))
        }
    };
    parse_quote!(#quoted)
}
