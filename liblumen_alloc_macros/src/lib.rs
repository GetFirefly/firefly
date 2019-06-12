#![recursion_limit = "128"]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
extern crate proc_macro;

mod size_classes;

use core::mem;
use proc_macro::TokenStream;

use proc_macro2::Span;
use quote::quote_spanned;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::token::{Bracket, Comma};
use syn::{parse_macro_input, parse_quote};
use syn::{Expr, ExprArray, ItemConst};
use syn::{ExprLit, IntSuffix, Lit, LitInt};

#[proc_macro_derive(SizeClassIndex)]
pub fn derive_size_class_index(input: TokenStream) -> TokenStream {
    self::size_classes::derive(input)
}

#[proc_macro]
pub fn generate_heap_sizes(input: TokenStream) -> TokenStream {
    let mut heap_sizes_const = parse_macro_input!(input as ItemConst);

    let max_heap_sizes = if cfg!(target_pointer_width = "64") {
        152
    } else {
        57
    };
    let mut heap_sizes: Vec<usize> = Vec::with_capacity(max_heap_sizes);
    // The seed for the fibonacci sequence, the first heap size will be 232 words
    let mut a = 90;
    let mut b = 142;
    for i in 0..19 {
        let c = b + a;
        heap_sizes.insert(i, c);
        mem::swap(&mut a, &mut b);
        b = c;
    }
    // Grow heap by 20% from this point on (at ~1M words)
    for i in 19..max_heap_sizes {
        let last_heap_size = heap_sizes[i - 1];
        heap_sizes.insert(i, last_heap_size + (last_heap_size / 5));
    }

    // Construct constant
    let mut heap_sizes_elems: Punctuated<Expr, Comma> = Punctuated::new();
    for heap_size in heap_sizes.iter() {
        heap_sizes_elems.push_value(Expr::Lit(ExprLit {
            attrs: Vec::new(),
            lit: Lit::Int(LitInt::new(
                *heap_size as u64,
                IntSuffix::None,
                Span::call_site(),
            )),
        }));
        heap_sizes_elems.push_punct(parse_quote!(,));
    }

    heap_sizes_const.ty = Box::new(parse_quote!([usize; #max_heap_sizes]));
    heap_sizes_const.expr = Box::new(Expr::Array(ExprArray {
        attrs: Vec::new(),
        bracket_token: Bracket {
            span: heap_sizes_const.expr.span(),
        },
        elems: heap_sizes_elems,
    }));

    let span = heap_sizes_const.span();
    let quoted = quote_spanned! {span=>
        #heap_sizes_const
    };

    TokenStream::from(quoted)
}
