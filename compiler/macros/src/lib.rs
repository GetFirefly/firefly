#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(iterator_try_collect)]
extern crate proc_macro;

mod bif;
mod option_group;
mod utils;

use proc_macro::TokenStream;

use syn::parse_macro_input;
use syn::AttributeArgs;

use self::option_group::{OptionGroupConfig, OptionGroupStruct};

/// Defines an option group which can be added to a `clap::App` as an argument
#[proc_macro_attribute]
pub fn option_group(attr: TokenStream, item: TokenStream) -> TokenStream {
    let option_group_config_args = parse_macro_input!(attr as AttributeArgs);
    let option_group_config = match OptionGroupConfig::from_args(option_group_config_args) {
        Ok(config) => config,
        Err(err) => return err.to_compile_error().into(),
    };
    let option_group = parse_macro_input!(item as OptionGroupStruct);
    self::option_group::generate_option_group(option_group_config, option_group)
}

#[proc_macro]
pub fn bif(input: TokenStream) -> TokenStream {
    use self::bif::BifSpec;

    let spec = parse_macro_input!(input as BifSpec);
    match self::bif::define_bif(spec) {
        Ok(tokens) => tokens,
        Err(error) => error.into_compile_error().into(),
    }
}
