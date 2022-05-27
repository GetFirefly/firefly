#![feature(box_patterns)]
#![deny(warnings)]

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Span;

use quote::{quote, quote_spanned};

use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::token::Colon2;
use syn::{parse, parse_macro_input};
use syn::{
    Ident, ItemFn, PathArguments, PathSegment, ReturnType, TraitBoundModifier, Type, TypeParamBound,
};

enum EntryOutput {
    Default,
    ExitStatus,
    Never,
    ImplTermination,
}

/// Attribute to declare the entry point of the program
///
/// Entry point must:
///
/// - Returns `impl ::std::process::Termination + 'static`
/// - Accept no arguments
/// - Is not generic
/// - Has no custom ABI specified
/// - Is not const
///
/// The entry function will be invoked by a shim introduced which handles translating the
/// entry return type to a normalized status code, and is exported as `lumen_entry`.
///
/// As a result, only one function can be decorated with `#[entry]` in the dependency
/// tree for a build, or linking will fail.
///
/// The core runtime (`liblumen_crt`) will invoke the entry point after initialization
#[proc_macro_attribute]
pub fn entry(args: TokenStream, input: TokenStream) -> TokenStream {
    let f = parse_macro_input!(input as ItemFn);
    let span = f.span();

    let mut entry_output = EntryOutput::Default;

    // check the function signature
    let valid_signature = f.sig.constness.is_none()
        && f.sig.abi.is_none()
        && f.sig.inputs.is_empty()
        && f.sig.generics.params.is_empty()
        && f.sig.generics.where_clause.is_none()
        && f.sig.variadic.is_none()
        && match f.sig.output {
            ReturnType::Default => true,
            ReturnType::Type(_, box Type::ImplTrait(ref ty)) => {
                if ty.bounds.len() != 2 {
                    false
                } else {
                    let has_termination = ty.bounds.iter().any(|b| {
                        if let TypeParamBound::Trait(ref bound) = b {
                            if let TraitBoundModifier::Maybe(_) = bound.modifier {
                                return false;
                            }
                            if bound.lifetimes.is_some() {
                                return false;
                            }
                            is_trait(&bound.path.segments, "std::process::Termination")
                        } else {
                            false
                        }
                    });
                    let has_static_lifetime = ty.bounds.iter().any(|b| {
                        if let TypeParamBound::Lifetime(ref lt) = b {
                            lt.ident == "static"
                        } else {
                            false
                        }
                    });
                    if has_termination && has_static_lifetime {
                        entry_output = EntryOutput::ImplTermination;
                        true
                    } else {
                        false
                    }
                }
            }
            ReturnType::Type(_, box Type::Never(_)) => {
                entry_output = EntryOutput::Never;
                true
            }
            ReturnType::Type(_, box Type::Path(ref path)) => {
                if path.path.is_ident("i32") {
                    entry_output = EntryOutput::ExitStatus;
                    true
                } else {
                    false
                }
            }
            _ => false,
        };

    if !valid_signature {
        return parse::Error::new(
            f.span(),
            "`#[entry]` function must have signature `fn() | fn -> [ ! | impl ::std::process::Termination + 'static]`",
        )
        .to_compile_error()
        .into();
    }

    if !args.is_empty() {
        return parse::Error::new(Span::call_site(), "This attribute accepts no arguments")
            .to_compile_error()
            .into();
    }

    let entry_ident = Ident::new(
        &format!("__lumen_crt_entry_{}", f.sig.ident),
        Span::call_site(),
    );
    let ident = &f.sig.ident;

    let entry = match entry_output {
        EntryOutput::ExitStatus => quote_spanned! { span =>
            pub unsafe extern "C" fn #entry_ident() -> i32 {
                #ident()
            }
        },
        EntryOutput::ImplTermination => quote_spanned! { span =>
            pub unsafe extern "C" fn #entry_ident() -> i32 {
                use ::std::process::Termination;
                #ident().report()
            }
        },
        EntryOutput::Never => quote_spanned! { span =>
            pub unsafe extern "C" fn #entry_ident() -> i32 {
                #ident();
            }
        },
        EntryOutput::Default => {
            quote_spanned! { span =>
                pub unsafe extern "C" fn #entry_ident() -> i32 {
                    #ident();
                    0
                }
            }
        }
    };

    quote!(
        #[doc(hidden)]
        #[export_name = "lumen_entry"]
        #entry

        #f
    )
    .into()
}

fn is_trait(path: &Punctuated<PathSegment, Colon2>, trait_name: &str) -> bool {
    if path.iter().any(|s| s.arguments != PathArguments::None) {
        return false;
    }
    let type_name_parts = path.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>();
    let type_name = type_name_parts.as_slice().join("::");
    type_name == trait_name
}
