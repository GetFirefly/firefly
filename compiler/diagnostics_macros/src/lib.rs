extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;

use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Arm, Attribute, Member};
use syn::{Data, DeriveInput, Error, Ident, Index, Token, UnOp};
use syn::{Expr, ExprField, ExprMatch, ExprMethodCall, ExprPath, ExprUnary};
use syn::{FieldPat, Pat, PatIdent, PatRest, PatStruct, PatTuple, PatTupleStruct, PatWild};
use syn::{Path, PathArguments, PathSegment};

#[proc_macro_derive(Spanned, attributes(span))]
pub fn derive_spanned(input: TokenStream) -> TokenStream {
    // Parse into syntax tree
    let derive = parse_macro_input!(input as DeriveInput);
    // Structure name
    let name = derive.ident;
    let result = match derive.data {
        Data::Struct(data) => derive_spanned_struct(name, data, derive.generics),
        Data::Enum(data) => derive_spanned_enum(name, data, derive.generics),
        Data::Union(_) => Err(Error::new(
            name.span(),
            "deriving Spanned on unions is not currently supported",
        )),
    };
    match result {
        Ok(ts) => ts,
        Err(err) => err.into_compile_error().into(),
    }
}

fn derive_spanned_struct(
    name: Ident,
    data: syn::DataStruct,
    generics: syn::Generics,
) -> Result<TokenStream, Error> {
    let access = extract_span_fields(name.span(), None, &data.fields)?;
    let span_expr = match access {
        SpanAccess::Field(member) => make_member_access(make_self_expr(member.span()), member),
        SpanAccess::Delegated(member) => {
            let base = make_member_access(make_self_expr(member.span()), member);
            make_delegated_member_access(base)
        }
        _ => unreachable!(),
    };

    let (impl_gen, ty_gen, where_clause) = generics.split_for_impl();
    let quoted = quote! {
        impl #impl_gen ::liblumen_diagnostics::Spanned for #name #ty_gen #where_clause {
            #[inline]
            fn span(&self) -> ::liblumen_diagnostics::SourceSpan {
                #span_expr
            }
        }
    };

    Ok(TokenStream::from(quoted))
}

fn derive_spanned_enum(
    name: Ident,
    data: syn::DataEnum,
    generics: syn::Generics,
) -> Result<TokenStream, Error> {
    let mut variants = Vec::with_capacity(data.variants.len());
    for variant in data.variants.iter() {
        let span = variant.span();
        let access = extract_span_fields(span, Some(variant.ident.clone()), &variant.fields)?;
        variants.push(access);
    }

    // Generate match patterns for each variant, where the body of the match arm returns the span
    let arms = variants
        .drain(..)
        .map(|access| match access {
            SpanAccess::Variant(variant, Member::Named(id)) => {
                let span = variant.span();
                let pat = Pat::Struct(PatStruct {
                    attrs: vec![],
                    path: make_path(&[Ident::new("Self", span), variant]),
                    brace_token: syn::token::Brace { span },
                    fields: Punctuated::from_iter(core::iter::once(FieldPat {
                        attrs: vec![],
                        member: Member::Named(id.clone()),
                        colon_token: None,
                        pat: Box::new(Pat::Ident(PatIdent {
                            attrs: vec![],
                            by_ref: None,
                            mutability: None,
                            ident: id.clone(),
                            subpat: None,
                        })),
                    })),
                    dot2_token: Some(Token![..](span)),
                });
                Arm {
                    attrs: vec![],
                    pat,
                    guard: None,
                    fat_arrow_token: Token![=>](span),
                    body: Box::new(make_deref_expr(make_var_expr(id))),
                    comma: Some(Token![,](span)),
                }
            }
            SpanAccess::Variant(variant, Member::Unnamed(idx)) => {
                let span = variant.span();
                let mut elems = Punctuated::new();
                let index = idx.index;
                for i in 0..=index {
                    if i == index {
                        elems.push(Pat::Ident(PatIdent {
                            attrs: vec![],
                            by_ref: None,
                            mutability: None,
                            ident: Ident::new("span", span),
                            subpat: None,
                        }))
                    } else {
                        elems.push(Pat::Wild(PatWild {
                            attrs: vec![],
                            underscore_token: Token![_](span),
                        }));
                    }
                }
                elems.push(Pat::Rest(PatRest {
                    attrs: vec![],
                    dot2_token: Token![..](span),
                }));
                let pat = Pat::TupleStruct(PatTupleStruct {
                    attrs: vec![],
                    path: make_path(&[Ident::new("Self", span), variant]),
                    pat: PatTuple {
                        attrs: vec![],
                        paren_token: syn::token::Paren { span },
                        elems,
                    },
                });
                Arm {
                    attrs: vec![],
                    pat,
                    guard: None,
                    fat_arrow_token: Token![=>](span),
                    body: Box::new(make_deref_expr(make_var_expr(Ident::new("span", span)))),
                    comma: Some(Token![,](span)),
                }
            }
            SpanAccess::DelegatedVariant(variant, Member::Named(id)) => {
                let span = variant.span();
                let pat = Pat::Struct(PatStruct {
                    attrs: vec![],
                    path: make_path(&[Ident::new("Self", span), variant]),
                    brace_token: syn::token::Brace { span },
                    fields: Punctuated::from_iter(core::iter::once(FieldPat {
                        attrs: vec![],
                        member: Member::Named(id.clone()),
                        colon_token: None,
                        pat: Box::new(Pat::Ident(PatIdent {
                            attrs: vec![],
                            by_ref: None,
                            mutability: None,
                            ident: id.clone(),
                            subpat: None,
                        })),
                    })),
                    dot2_token: Some(Token![..](span)),
                });
                let body = make_delegated_member_access(make_var_expr(id));
                Arm {
                    attrs: vec![],
                    pat,
                    guard: None,
                    fat_arrow_token: Token![=>](span),
                    body: Box::new(body),
                    comma: Some(Token![,](span)),
                }
            }
            SpanAccess::DelegatedVariant(variant, Member::Unnamed(idx)) => {
                let span = variant.span();
                let mut elems = Punctuated::new();
                let index = idx.index;
                for i in 0..=index {
                    if i == index {
                        elems.push(Pat::Ident(PatIdent {
                            attrs: vec![],
                            by_ref: None,
                            mutability: None,
                            ident: Ident::new("span", span),
                            subpat: None,
                        }))
                    } else {
                        elems.push(Pat::Wild(PatWild {
                            attrs: vec![],
                            underscore_token: Token![_](span),
                        }));
                    }
                }
                elems.push(Pat::Rest(PatRest {
                    attrs: vec![],
                    dot2_token: Token![..](span),
                }));
                let pat = Pat::TupleStruct(PatTupleStruct {
                    attrs: vec![],
                    path: make_path(&[Ident::new("Self", span), variant]),
                    pat: PatTuple {
                        attrs: vec![],
                        paren_token: syn::token::Paren { span },
                        elems,
                    },
                });
                let body = make_delegated_member_access(make_var_expr(Ident::new("span", span)));
                Arm {
                    attrs: vec![],
                    pat,
                    guard: None,
                    fat_arrow_token: Token![=>](span),
                    body: Box::new(body),
                    comma: Some(Token![,](span)),
                }
            }
            _ => unreachable!(),
        })
        .collect();

    let span = name.span();
    let span_expr = Expr::Match(ExprMatch {
        attrs: vec![],
        match_token: Token![match](span),
        expr: Box::new(make_self_expr(span)),
        brace_token: syn::token::Brace { span },
        arms,
    });

    let (impl_gen, ty_gen, where_clause) = generics.split_for_impl();
    let quoted = quote! {
        impl #impl_gen ::liblumen_diagnostics::Spanned for #name #ty_gen #where_clause {
            fn span(&self) -> ::liblumen_diagnostics::SourceSpan {
                #span_expr
            }
        }
    };

    Ok(TokenStream::from(quoted))
}

enum SpanAccess {
    Field(Member),
    Variant(Ident, Member),
    Delegated(Member),
    DelegatedVariant(Ident, Member),
}
impl Spanned for SpanAccess {
    fn span(&self) -> Span {
        match self {
            Self::Field(m)
            | Self::Variant(_, m)
            | Self::Delegated(m)
            | Self::DelegatedVariant(_, m) => m.span(),
        }
    }
}

fn make_path(idents: &[Ident]) -> Path {
    let mut segments = Punctuated::new();
    for ident in idents.iter().cloned() {
        segments.push(PathSegment {
            ident,
            arguments: PathArguments::None,
        });
    }
    Path {
        leading_colon: None,
        segments,
    }
}

fn make_self_expr(span: Span) -> Expr {
    make_var_expr(Ident::new("self", span))
}

fn make_var_expr(ident: Ident) -> Expr {
    Expr::Path(ExprPath {
        attrs: vec![],
        qself: None,
        path: make_path(&[ident]),
    })
}

fn make_deref_expr(expr: Expr) -> Expr {
    let span = expr.span();
    Expr::Unary(ExprUnary {
        attrs: vec![],
        op: UnOp::Deref(Token![*](span)),
        expr: Box::new(expr),
    })
}

fn make_member_access(base: Expr, member: Member) -> Expr {
    let span = member.span();
    Expr::Field(ExprField {
        attrs: vec![],
        base: Box::new(base),
        dot_token: Token![.](span),
        member,
    })
}

fn make_delegated_member_access(receiver: Expr) -> Expr {
    let span = receiver.span();
    Expr::MethodCall(ExprMethodCall {
        attrs: vec![],
        receiver: Box::new(receiver),
        dot_token: Token![.](span),
        method: Ident::new("span", span),
        turbofish: None,
        paren_token: syn::token::Paren { span },
        args: Punctuated::new(),
    })
}

fn has_span_attr(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .find(|attr| attr.path.is_ident("span"))
        .is_some()
}

fn extract_span_fields(
    span: Span,
    variant: Option<Ident>,
    fields: &syn::Fields,
) -> Result<SpanAccess, Error> {
    match fields {
        syn::Fields::Named(fields) => {
            let mut spanned = fields
                .named
                .iter()
                .filter_map(|f| {
                    if has_span_attr(f.attrs.as_slice()) {
                        let delegated = !is_source_span(&f.ty);
                        Some((delegated, Member::Named(f.ident.clone().unwrap())))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if spanned.is_empty() {
                Err(Error::new(
                    span,
                    "Spanned requires at least one field to be marked with the #[span] attribute",
                ))
            } else if spanned.len() > 1 {
                Err(Error::new(
                    span,
                    "Spanned requires one field tagged with #[span], but multiple were found",
                ))
            } else {
                let (delegated, member) = spanned.pop().unwrap();
                match variant {
                    None if delegated => Ok(SpanAccess::Delegated(member)),
                    None => Ok(SpanAccess::Field(member)),
                    Some(variant) if delegated => Ok(SpanAccess::DelegatedVariant(variant, member)),
                    Some(variant) => Ok(SpanAccess::Variant(variant, member)),
                }
            }
        }
        syn::Fields::Unnamed(fields) => {
            let mut spanned = fields
                .unnamed
                .iter()
                .enumerate()
                .filter_map(|(i, f)| {
                    if has_span_attr(f.attrs.as_slice()) {
                        let delegated = !is_source_span(&f.ty);
                        Some((
                            delegated,
                            Member::Unnamed(Index {
                                index: i as u32,
                                span: f.span(),
                            }),
                        ))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if spanned.is_empty() {
                // If there are multiple fields, then we can't make a selection, so raise an error
                if fields.unnamed.len() > 1 {
                    return Err(Error::new(span, "Spanned requires at least one field to be marked with the #[span] attribute"));
                }
                // Otherwise, we can infer the only field to contain a Spanned impl
                match variant {
                    None => Ok(SpanAccess::Delegated(Member::Unnamed(Index {
                        index: 0,
                        span,
                    }))),
                    Some(variant) => Ok(SpanAccess::DelegatedVariant(
                        variant,
                        Member::Unnamed(Index { index: 0, span }),
                    )),
                }
            } else if spanned.len() > 1 {
                Err(Error::new(
                    span,
                    "Spanned requires one field tagged with #[span], but multiple were found",
                ))
            } else {
                let (delegated, member) = spanned.pop().unwrap();
                match variant {
                    None if delegated => Ok(SpanAccess::Delegated(member)),
                    None => Ok(SpanAccess::Field(member)),
                    Some(variant) if delegated => Ok(SpanAccess::DelegatedVariant(variant, member)),
                    Some(variant) => Ok(SpanAccess::Variant(variant, member)),
                }
            }
        }
        syn::Fields::Unit if variant.is_some() => Err(Error::new(
            span,
            "Spanned cannot be derived on enums with unit variants",
        )),
        syn::Fields::Unit => Err(Error::new(
            span,
            "Spanned requires a struct with at least one SourceSpan field",
        )),
    }
}

fn is_source_span(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(tpath) => {
            if tpath.path.is_ident("SourceSpan") {
                return true;
            }
            match tpath.path.segments.len() {
                1 if tpath.path.leading_colon.is_none() => {
                    return false;
                }
                1 => {
                    let first = tpath.path.segments.first().unwrap();
                    first.ident == "SourceSpan" && first.arguments == PathArguments::None
                }
                2 => {
                    let mut iter = tpath.path.segments.iter();
                    let first = iter.next().unwrap();
                    if first.ident != "liblumen_diagnostics"
                        || first.arguments != PathArguments::None
                    {
                        return false;
                    }
                    let second = iter.next().unwrap();
                    second.ident == "SourceSpan" && second.arguments == PathArguments::None
                }
                _ => false,
            }
        }
        _ => false,
    }
}
