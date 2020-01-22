#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
#![feature(box_syntax)]
#![feature(box_patterns)]
extern crate proc_macro;

mod ffi;
mod option_group;
mod utils;

use proc_macro::TokenStream;

use syn::parse_macro_input;
use syn::{AttributeArgs, ItemStruct};

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

#[proc_macro_attribute]
pub fn foreign_struct(attr: TokenStream, item: TokenStream) -> TokenStream {
    let strukt = parse_macro_input!(item as ItemStruct);
    let foreign_struct_args = parse_macro_input!(attr as AttributeArgs);
    self::ffi::define_foreign_struct(foreign_struct_args, strukt)
}
