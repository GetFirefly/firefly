use proc_macro::TokenStream;

use quote::{format_ident, quote};

use syn::spanned::Spanned;
use syn::Error;
use syn::{AttributeArgs, ItemStruct};

pub fn define_foreign_struct(_args: AttributeArgs, strukt: ItemStruct) -> TokenStream {
    let name = strukt.ident.clone();
    let ref_name = format_ident!("{}{}", name.clone(), "Ref");
    let vis = strukt.vis.clone();
    let has_fields = !strukt.fields.is_empty();
    let has_generics = !strukt.generics.params.is_empty();
    let fields = strukt.fields.iter();
    let attrs = strukt.attrs.iter();

    if has_generics {
        return Error::new(
            strukt.span(),
            "generic foreign structs are currently unsupported",
        )
        .to_compile_error()
        .into();
    }

    let quoted = {
        let struct_quoted = if has_fields {
            quote! {
                #[repr(C)]
                #(#attrs)*
                #vis struct #name {
                    #(
                        #fields,
                    )*
                }
            }
        } else {
            quote! {
                #[repr(C)]
                #(#attrs)*
                #vis struct #name {
                    _private: liblumen_util::ffi::Opaque,
                }
            }
        };

        let impl_quoted = quote! {
            #[repr(transparent)]
            #[derive(Copy, Clone, Hash)]
            #vis struct #ref_name(*mut #name);
            impl #ref_name {
                pub unsafe fn new(ptr: *mut #name) -> Self {
                    Self(ptr)
                }

                pub fn is_null(&self) -> bool { self.0.is_null() }
            }
            impl Default for #ref_name {
                fn default() -> Self {
                    Self(::core::ptr::null_mut())
                }
            }
            impl std::convert::AsRef<#name> for #ref_name {
                fn as_ref(&self) -> &#name {
                    unsafe { &*self.0 }
                }
            }
            impl std::convert::From<*const #name> for #ref_name {
                fn from(ptr: *const #name) -> Self {
                    Self(ptr as *mut _)
                }
            }
            impl std::convert::From<*mut #name> for #ref_name {
                fn from(ptr: *mut #name) -> Self {
                    Self(ptr)
                }
            }
            impl std::convert::Into<*const #name> for #ref_name {
                fn into(self) -> *const #name {
                    self.0 as *const _
                }
            }
            impl std::convert::Into<*mut #name> for #ref_name {
                fn into(self) -> *mut #name {
                    self.0
                }
            }
            impl std::fmt::Debug for #ref_name {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "{:p}", self.0)
                }
            }
            impl std::fmt::Pointer for #ref_name {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "{:p}", self.0)
                }
            }
            impl Eq for #ref_name {}
            impl PartialEq for #ref_name {
                fn eq(&self, other: &Self) -> bool {
                    ::core::ptr::eq(self.0, other.0)
                }
            }
        };

        quote! {
            #struct_quoted
            #impl_quoted
        }
    };

    TokenStream::from(quoted)
}
