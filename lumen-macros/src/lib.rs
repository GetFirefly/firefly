#![recursion_limit="128"]
///! This crate contains all procedural macros used in other Lumen crates
///!
///! Currently that consists of the `#[tag_type]` and `#[tagged_with(T)]` attributes
extern crate proc_macro;
extern crate proc_macro2;

use quote::quote;
use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{parse_macro_input, parse_quote, Ident, DeriveInput};

///! The `#[tagged_with(T)]` attribute is added to structs or enums which need to have
///! their pointers tagged with a value in the enum of type `T`.
#[proc_macro_attribute]
pub fn tagged_with(args: TokenStream, input: TokenStream) -> TokenStream {
    let tag_type_ident = parse_macro_input!(args as Ident);
    let enum_ast = parse_macro_input!(input as DeriveInput);
    let tagged_type = &enum_ast.ident;
    let tag_type = Ident::new(&format!("Tagged{}", tag_type_ident), Span::call_site());
    TokenStream::from(quote!{
        #enum_ast

        impl #tag_type<#tagged_type> for #tagged_type {}
    })
}

///! The `#[tag_type]` attribute defines a type which will be used to tag pointers to data structures.
///!
///! The type decorated by this attribute must follow these rules:
///!
///! * Must be an enum
///! * All variants of the num must only be names, they can have no fields (no tuple structs or struct variants)
///!
///! The decorated enum will have the following added:
///!
///! * `#[repr(usize)]`
///! * `#[derive]` for PartialEq, Eq, Clone, Copy, Debug
///!
///! Under the covers this expands to a new trait, which will be derived automatically for types decorated
///! with the `#[tagged_with(T)]` attribute
#[proc_macro_attribute]
pub fn tag_type(_args: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as DeriveInput);
    let expanded = impl_tag_type(&mut ast);
    expanded.into()
}

fn impl_tag_type(ast: &mut syn::DeriveInput) -> impl Into<TokenStream> {
    use syn::*;

    let tag_type = &ast.ident;

    // Extend tag type
    let repr_attr: Attribute = parse_quote!(#[repr(usize)]);
    ast.attrs.push(repr_attr);
    let derive_attr: Attribute = parse_quote!(#[derive(PartialEq, Eq, Clone, Copy, Debug)]);
    ast.attrs.push(derive_attr);

    // Define trait
    let tag_type_trait_type = Ident::new(&format!("Tagged{}", tag_type), Span::call_site());
    let variants: Vec<syn::Type> = if let Data::Enum(DataEnum { variants, .. }) = &ast.data {
        variants.iter().map(|ref v| {
            syn::parse_str(&format!("{}::{}", tag_type, v.ident)).unwrap()
        }).collect()
    } else {
        panic!("#[derive(TagType)] must be used on enums only")
    };

    quote!{
        #ast

        pub trait #tag_type_trait_type<T: Sized> {
            #[inline]
            fn has_tag(ptr: *const T, tag: #tag_type) -> bool {
                (ptr as usize) & (tag as usize) == (tag as usize)
            }

            fn tag_of(ptr: *const T) -> #tag_type {
                #(
                    let v = #variants;
                    if Self::has_tag(ptr, v) {
                        return v;
                    }
                )*
                unsafe { std::mem::transmute::<usize, #tag_type>(0) }
            }

            fn tag(ptr: *const T, tag: #tag_type) -> *const T {
                let tag = tag as usize;
                let uptr = ptr as usize;
                (uptr & !tag | (tag * true as usize)) as *const T
            }

            fn untag(ptr: *const T, tag: #tag_type) -> *const T {
                let tag = tag as usize;
                let uptr = ptr as usize;
                (uptr & !tag | (tag * false as usize)) as *const T
            }
        }
    }
}
