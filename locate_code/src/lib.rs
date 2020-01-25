#![deny(warnings)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Error, ItemFn};

#[proc_macro_attribute]
pub fn locate_code(attributes: TokenStream, code_token_stream: TokenStream) -> TokenStream {
    if attributes.is_empty() {
        let code_fn = parse_macro_input!(code_token_stream as ItemFn);
        let span = code_fn.span();
        let path = span.source_file().path();
        let file = path.to_string_lossy();
        let line_column = span.start();
        let line = line_column.line as u32;
        let column = line_column.column as u32;

        let all_tokens = quote! {
            pub const LOCATION: liblumen_alloc::location::Location = liblumen_alloc::location::Location {
                file: #file,
                line: #line,
                column: #column
            };

            pub const LOCATED_CODE: liblumen_alloc::erts::process::code::LocatedCode = liblumen_alloc::erts::process::code::LocatedCode {
                location: LOCATION,
                code
            };

            #code_fn
        };

        all_tokens.into()
    } else {
        Error::new(Span::call_site(), "located_code takes no arguments")
            .to_compile_error()
            .into()
    }
}
