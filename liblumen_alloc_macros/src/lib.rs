#![recursion_limit = "128"]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
extern crate proc_macro;

mod size_classes;

use proc_macro::TokenStream;

#[proc_macro_derive(SizeClassIndex)]
pub fn derive_size_class_index(input: TokenStream) -> TokenStream {
    self::size_classes::derive(input)
}
