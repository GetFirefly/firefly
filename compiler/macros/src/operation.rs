use proc_macro::TokenStream;

use quote::{format_ident, quote};

use syn::spanned::Spanned;
use syn::Error;
use syn::{AttributeArgs, Expr, ExprLit, ItemStruct, Lit, LitStr, NestedMeta};

fn args_to_operation_name(meta: &NestedMeta) -> Result<&LitStr, Error> {
    if let NestedMeta::Lit(Lit::Str(ref s)) = meta {
        Ok(s)
    } else {
        Err(Error::new(
            meta.span(),
            "expected literal string (the operation name) here",
        ))
    }
}

pub fn define_operation(args: AttributeArgs, strukt: ItemStruct) -> TokenStream {
    let name = strukt.ident.clone();
    let vis = strukt.vis.clone();
    let attrs = strukt.attrs.iter();

    if args.len() != 1 {
        return Error::new(
            strukt.span(),
            "#[operation] attribute requires an operation name",
        )
        .to_compile_error()
        .into();
    }

    let maybe_op_name = args_to_operation_name(&args[0]);
    if let Err(err) = maybe_op_name {
        return err.to_compile_error().into();
    }
    let op_name = maybe_op_name.unwrap();
    let op_name_str = op_name.value();
    let dyn_cast_extern_link_name = Expr::Lit(ExprLit {
        attrs: vec![],
        lit: Lit::Str(LitStr::new(
            &format!("mlirOperationDynCast{}", &op_name_str),
            strukt.span(),
        )),
    });
    let dyn_cast_extern_name = format_ident!("mlir_operation_dyn_cast_{}", &op_name_str);

    let quoted = quote! {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        #(#attrs)*
        #vis struct #name(::liblumen_mlir::ir::Operation);
        impl #name {
            const NAME: 'static str = #op_name;

            #[allow(unused)]
            #[inline(always)]
            pub fn is_null(&self) -> bool {
                self.0.is_null()
            }

            #[allow(unused)]
            fn build(loc: ::liblumen_mlir::ir::Location) -> ::liblumen_mlir::ir::OperationState {
                ::liblumen_mlir::ir::OperationState::get(Self::NAME, loc);
            }
        }
        impl ::std::convert::Into<::liblumen_mlir::ir::Operation> for #name {
            #[inline(always)]
            fn into(self) -> ::liblumen_mlir::ir::Operation {
                self.0
            }
        }
        impl ::std::convert::TryFrom<::liblumen_mlir::ir::Operation> for #name {
            type Err = ::anyhow::Error;

            #[inline(always)]
            fn try_from(op: ::liblumen_mlir::ir::Operation) -> Result<Self, Self::Err> {
                op.dyn_cast::<#name>()
            }
        }
        impl ::std::ops::Deref for #name {
            type Target = ::liblumen_mlir::ir::Operation;

            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl Eq for #name {}
        impl PartialEq for #name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl ::std::fmt::Debug for #name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                write!(f, "{}", &self.0);
            }
        }
        impl ::std::fmt::Display for #name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                write!(f, "{}", &self.0);
            }
        }
        impl ::std::fmt::Pointer for #name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                write!(f, "{:p}", self.0)
            }
        }
        impl ::std::convert::TryFrom<::liblumen_mlir::ir::Operation> for #name {
            type Error = ::liblumen_mlir::ir::InvalidOperationCastError;

            #[inline(always)]
            fn try_from(op: Operation) -> Result<Self, Self::Err> {
                use ::liblumen_mlir::ir::InvalidOperationCastError;

                let result = unsafe { #dyn_cast_extern_name(op) };
                if result.is_null() {
                    Err(InvalidOperationCastError)
                } else {
                    Ok(Self(result))
                }
            }
        }

        extern "C" {
            #[link_name = #dyn_cast_extern_link_name]
            fn #dyn_cast_extern_name(op: Operation) -> #name;
        }
    };

    TokenStream::from(quoted)
}
