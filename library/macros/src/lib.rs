#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
extern crate proc_macro;

use core::mem;
use proc_macro::TokenStream;

use proc_macro2::Span;
use quote::quote_spanned;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::token::{Bracket, Comma};
use syn::{parse_macro_input, parse_quote};
use syn::{Expr, ExprArray, ItemConst};
use syn::{ExprLit, Lit, LitInt};

#[proc_macro]
pub fn define_closure_impls(input: TokenStream) -> TokenStream {
    let mut config = parse_macro_input!(input as Meta);

    let config = match config {
        Meta::NameValue(kv) => read_config_items(core::iter::once(&kv)),
        Meta::List(items) => read_config_items(items.iter().map(|nm| match nm {
            NestedMeta::Meta(Meta::NameValue(kv)) => Ok(kv),
            nm => Err(Error::new(nm.span(), "expected key/value meta")),
        })),
        Meta::Path(p) => Err(Error::new(p.span(), "expected meta list or key/value")),
    };

    if let Err(err) = config {
        return err.into_compile_error().into();
    }

    let (max_arity, max_env_size) = config.unwrap();

    let mut impls = Vec::with_capacity(max_arity * max_env_size);

    for arity in 0..max_arity {
        for env_size in 0..max_env_size {
            impls.push(define_closure_impl(span, arity, env_size));
        }
    }

    TokenStream::from(impls.drain(..).reduce(|mut acc, ts| {
        acc.extend(ts);
        acc
    }))
}

fn define_closure_impl(span: Span, arity: u8, env_size: u8) -> proc_macro2::TokenStream {
    quote_spanned! {span =>
        impl FnOnce<Args0> for &Closure {
            type Output = ErlangResult;

            #[inline]
            extern "rust-call" fn call_once(self, _args: Args0) -> Self::Output {
                assert_eq!(self.arity, 0, "mismatched arity");
                let fun = unsafe { core::mem::transmute::<_, Fun0>(self.fun) };
                match self.env.len() {
                    0 => fun(),
                    1 => fun(self.env[0]),
                    2 => fun(self.env[0], self.env[1]),
                    3 => fun(self.env[0], self.env[1], self.env[2]),
                    4 => fun(self.env[0], self.env[1], self.env[2], self.env[3]),
                    5 => fun(self.env[0], self.env[1], self.env[2], self.env[3], self.env[4]),
                    6 => fun(self.env[0], self.env[1], self.env[2], self.env[3], self.env[4], self.env[5]),
                    _ => unimplemented!()
                }
            }
        }
        impl FnMut<Args0> for &Closure {
            #[inline]
            extern "rust-call" fn call_mut(&mut self, _args: Args0) -> Self::Output {
                assert_eq!(self.arity, 0, "mismatched arity");
                let fun = unsafe { core::mem::transmute::<_, Fun0>(self.fun) };
                match self.env.len() {
                    0 => fun(),
                    1 => fun(self.env[0]),
                    2 => fun(self.env[0], self.env[1]),
                    3 => fun(self.env[0], self.env[1], self.env[2]),
                    4 => fun(self.env[0], self.env[1], self.env[2], self.env[3]),
                    5 => fun(self.env[0], self.env[1], self.env[2], self.env[3], self.env[4]),
                    6 => fun(self.env[0], self.env[1], self.env[2], self.env[3], self.env[4], self.env[5]),
                    _ => unimplemented!()
                }
            }
        }
        impl Fn<Args0> for &Closure {
            #[inline]
            extern "rust-call" fn call(&self, _args: Args0) -> Self::Output {
                assert_eq!(self.arity, 0, "mismatched arity");
                let fun = unsafe { core::mem::transmute::<_, Fun0>(self.fun) };
                match self.env.len() {
                    0 => fun(),
                    1 => fun(self.env[0]),
                    2 => fun(self.env[0], self.env[1]),
                    3 => fun(self.env[0], self.env[1], self.env[2]),
                    4 => fun(self.env[0], self.env[1], self.env[2], self.env[3]),
                    5 => fun(self.env[0], self.env[1], self.env[2], self.env[3], self.env[4]),
                    6 => fun(self.env[0], self.env[1], self.env[2], self.env[3], self.env[4], self.env[5]),
                    _ => unimplemented!()
                }
            }
        }
    };

    TokenStream::from(quoted)
}

fn read_config_items<I: Iterator<Item = Result<&MetaNameValue, Error>>>(
    span: Span,
    mut items: I,
) -> Result<(u8, u8), Error> {
    let mut max_arity: Option<u8> = None;
    let mut max_env_size: Option<u8> = None;

    while let Some(kv) = items.next() {
        let kv = kv?;
        if kv.path.is_ident("max_arity") {
            match kv.lit {
                Lit::Int(i) => {
                    max_arity.replace(i.base10_parse().unwrap());
                }
                lit => {
                    return Err(Error::new(
                        lit.span(),
                        "invalid value for max_arity, expected integer",
                    ))
                }
            }
        } else if kv.path.is_ident("max_env_size") {
            match kv.lit {
                Lit::Int(i) => {
                    max_env_size.replace(i.base10_parse().unwrap());
                }
                lit => {
                    return Err(Error::new(
                        lit.span(),
                        "invalid value for max_env_size, expected integer",
                    ))
                }
            }
        } else {
            return Err(Error::new(
                kv.span(),
                format!("invalid config key: {}", &kv.path.get_ident().unwrap()),
            ));
        }
    }
    if max_arity.is_none() {
        Err(Error::new(
            span,
            "max_arity config key is required, but is missing!",
        ))
    } else if max_env_size == 0 {
        Err(Error::new(
            span,
            "max_env_size config key is required, but is missing!",
        ))
    } else {
        Ok((max_arity.unwrap(), max_env_size.unwrap()))
    }
}
