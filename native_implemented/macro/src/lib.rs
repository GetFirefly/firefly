#![deny(warnings)]
#![feature(box_patterns)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_def_site)]
extern crate proc_macro;

use proc_macro2::Ident;

use proc_macro::TokenStream;

use quote::{quote, ToTokens};

use syn::parse::{Parse, ParseBuffer};
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, AngleBracketedGenericArguments, Error, FnArg, GenericArgument, ItemFn,
    LitInt, Pat, PatIdent, PatType, Path, PathArguments, PathSegment, Token, Type, TypePath,
    TypeReference,
};

#[proc_macro_attribute]
pub fn label(_: TokenStream, result_token_stream: TokenStream) -> TokenStream {
    let result_item_fn = parse_macro_input!(result_token_stream as ItemFn);

    match Signatures::label(&result_item_fn) {
        Ok(signatures) => {
            let frame = frame_for_label();
            let const_native = signatures.const_native();
            let native_fn = signatures.native_fn();

            let all_tokens = quote! {
                #frame
                #const_native

                #native_fn
                #result_item_fn
            };

            all_tokens.into()
        }
        Err(error) => error.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn function(
    module_function_arity_token_stream: TokenStream,
    result_token_stream: TokenStream,
) -> TokenStream {
    let module_function_arity =
        parse_macro_input!(module_function_arity_token_stream as ModuleFunctionArity);
    let result_item_fn = parse_macro_input!(result_token_stream as ItemFn);

    match Signatures::entry_point(&result_item_fn, module_function_arity.arity) {
        Ok(signatures) => {
            let const_arity = signatures.const_arity();
            let const_native = signatures.const_native();
            let frame = frame_for_entry_point();
            let frame_for_native = frame_for_native();
            let function = module_function_arity.function();
            let function_symbol = function_symbol();
            let module_function_arity_fn = module_function_arity_fn();
            let export_name = module_function_arity.export_name();
            let native_fn = signatures.native_fn();

            let all_tokens = quote! {
                #const_arity
                #const_native

                #frame
                #frame_for_native
                #function
                #function_symbol
                #module_function_arity_fn
                #[export_name = #export_name]
                #native_fn
                #result_item_fn
            };

            all_tokens.into()
        }
        Err(error) => error.to_compile_error().into(),
    }
}

fn fn_arg_to_ident(fn_arg: &FnArg) -> Ident {
    match fn_arg {
        FnArg::Typed(PatType {
            pat: box Pat::Ident(PatIdent { ident, .. }),
            ..
        }) => ident.clone(),
        _ => unimplemented!(
            "result function is not expected to have argument like {:?}",
            fn_arg
        ),
    }
}

fn frame_for_entry_point() -> proc_macro2::TokenStream {
    quote! {
        pub fn frame() -> liblumen_alloc::erts::process::Frame {
            frame_for_native(NATIVE)
        }
    }
}

fn frame_for_label() -> proc_macro2::TokenStream {
    quote! {
        pub fn frame() -> liblumen_alloc::erts::process::Frame {
            super::frame_for_native(NATIVE)
        }
    }
}

fn frame_for_native() -> proc_macro2::TokenStream {
    quote! {
        pub fn frame_for_native(native: liblumen_alloc::erts::process::Native) -> liblumen_alloc::erts::process::Frame {
            liblumen_alloc::erts::process::Frame::new(module_function_arity(), native)
        }
    }
}

fn function_symbol() -> proc_macro2::TokenStream {
    quote! {
        pub fn function_symbol() -> liblumen_core::symbols::FunctionSymbol {
            liblumen_core::symbols::FunctionSymbol {
                module: super::module_id(),
                function: function().id(),
                arity: ARITY,
                ptr: native as *const std::ffi::c_void
            }
        }
    }
}

fn module_function_arity_fn() -> proc_macro2::TokenStream {
    quote! {
        pub fn module_function_arity() -> liblumen_alloc::erts::ModuleFunctionArity {
            liblumen_alloc::erts::ModuleFunctionArity {
                module: super::module(),
                function: function(),
                arity: ARITY,
            }
        }
    }
}

#[derive(Debug)]
struct ModuleFunctionArity {
    module: String,
    function: String,
    arity: u8,
}

impl ModuleFunctionArity {
    fn export_name(&self) -> String {
        format!("{}:{}/{}", self.module, self.function, self.arity)
    }

    fn function(&self) -> proc_macro2::TokenStream {
        let function = &self.function;

        quote! {
            pub fn function() -> liblumen_alloc::erts::term::prelude::Atom {
                liblumen_alloc::erts::term::prelude::Atom::from_str(#function)
            }
        }
    }

    fn parse_arity(input: &ParseBuffer) -> syn::parse::Result<u8> {
        let arity_lit_int = input.parse::<LitInt>()?;

        arity_lit_int.base10_parse()
    }

    fn parse_function(input: &ParseBuffer) -> syn::parse::Result<String> {
        let function = if let Ok(ident) = input.parse::<syn::Ident>() {
            ident.to_string()
        } else if let Ok(_) = input.parse::<Token![loop]>() {
            "loop".to_string()
        } else if let Ok(_) = input.parse::<Token![self]>() {
            "self".to_string()
        } else if let Ok(_) = input.parse::<Token![*]>() {
            "*".to_string()
        } else if let Ok(_) = input.parse::<Token![+]>() {
            if let Ok(_) = input.parse::<Token![+]>() {
                "++".to_string()
            } else {
                "+".to_string()
            }
        } else if let Ok(_) = input.parse::<Token![-]>() {
            if let Ok(_) = input.parse::<Token![-]>() {
                "--".to_string()
            } else {
                "-".to_string()
            }
        } else if let Ok(_) = input.parse::<Token![/]>() {
            if let Ok(_) = input.parse::<Token![=]>() {
                "/=".to_string()
            } else {
                "/".to_string()
            }
        } else if let Ok(_) = input.parse::<Token![<]>() {
            "<".to_string()
        } else if let Ok(_) = input.parse::<Token![=]>() {
            if let Ok(_) = input.parse::<Token![/]>() {
                if let Ok(_) = input.parse::<Token![=]>() {
                    "=/=".to_string()
                } else {
                    unimplemented!("parse function name from {:?}", input);
                }
            } else if let Ok(_) = input.parse::<Token![:]>() {
                if let Ok(_) = input.parse::<Token![=]>() {
                    "=:=".to_string()
                } else {
                    unimplemented!("parse function name from {:?}", input);
                }
            } else if let Ok(_) = input.parse::<Token![<]>() {
                "=<".to_string()
            } else if let Ok(_) = input.parse::<Token![=]>() {
                "==".to_string()
            } else {
                unimplemented!("parse function name from {:?}", input);
            }
        } else if let Ok(_) = input.parse::<Token![>]>() {
            if let Ok(_) = input.parse::<Token![=]>() {
                ">=".to_string()
            } else {
                ">".to_string()
            }
        // anonymous functions
        } else if let Ok(index) = input.parse::<LitInt>() {
            if let Ok(_) = input.parse::<Token![-]>() {
                if let Ok(old_unique) = input.parse::<LitInt>() {
                    if let Ok(_) = input.parse::<Token![-]>() {
                        if let Ok(unique) = input.parse::<LitInt>() {
                            let span = unique.span();
                            let start = span.start();
                            let end = span.end();

                            if start.line == end.line {
                                let len = end.column - start.column;

                                if len == 32 {
                                    format!("{}-{}-{}", index, old_unique, unique)
                                } else {
                                    return Err(Error::new(span, format!("UNIQUE should be a 32-digit hexadecimal integer, but is {} digits long", len)));
                                }
                            } else {
                                return Err(Error::new(span, "UNIQUE should be on one line"));
                            }
                        } else {
                            return Err(input.error(
                                "Missing UNIQUE in anonymous function (INDEX-OLD_UNIQUE-UNIQUE",
                            ));
                        }
                    } else {
                        return Err(input.error("Missing `-` after OLD_UNIQUE in anonymous function (INDEX-OLD_UNIQUE-UNIQUE)"));
                    }
                } else {
                    return Err(input.error(
                        "Missing OLD_UNIQUE in anonymous function (INDEX-OLD_UNIQUE-UNIQUE",
                    ));
                }
            } else {
                return Err(input.error(
                    "Missing `-` after INDEX in anonymous function (INDEX-OLD_UNIQUE-UNIQUE",
                ));
            }
        } else {
            unimplemented!("parse function name from {:?}", input);
        };

        Ok(function)
    }

    fn parse_module(input: &ParseBuffer) -> syn::parse::Result<String> {
        let mut module = if let Ok(ident) = input.parse::<syn::Ident>() {
            ident.to_string()
        } else {
            unimplemented!("parse module name from {:?}", input);
        };

        loop {
            // End of module name.
            // This separates module name and function name
            if input.peek(Token![:]) {
                break;
            } else {
                if let Ok(_) = input.parse::<Token![.]>() {
                    module.push('.');

                    if let Ok(relative) = input.parse::<syn::Ident>() {
                        module.push_str(&relative.to_string());
                    } else {
                        return Err(input.error(
                            "Relative module names must follow Elixir namespace separator (`.`)",
                        ));
                    }
                } else {
                    return Err(input.error("Elixir namespace operator (`.`) must follow qualifier.  Use `:` separate module from function name."));
                }
            }
        }

        Ok(module)
    }
}

impl Parse for ModuleFunctionArity {
    fn parse(input: &ParseBuffer) -> syn::parse::Result<Self> {
        if input.is_empty() {
            Err(input.error("MODULE:FUNCTION/ARITY required"))
        } else {
            let module = Self::parse_module(input)?;

            input.parse::<Token![:]>()?;

            let function = Self::parse_function(input)?;

            input.parse::<Token![/]>()?;

            let arity = Self::parse_arity(input)?;

            Ok(ModuleFunctionArity {
                module,
                function,
                arity,
            })
        }
    }
}

struct Native {
    fn_arg_vec: Vec<FnArg>,
}

impl Native {
    pub fn arity(&self) -> u8 {
        self.fn_arg_vec.len() as u8
    }
}

enum Process {
    Arc,
    Ref,
    None,
}

struct Result {
    process: Process,
    return_type: ReturnType,
}

enum ReturnType {
    Result,
    Term,
}

struct Signatures {
    native: Native,
    result: Result,
}

impl Signatures {
    pub fn entry_point(result_item_fn: &ItemFn, arity: u8) -> std::result::Result<Self, Error> {
        if result_item_fn.sig.ident != "result" {
            return Err(Error::new(
                result_item_fn.sig.ident.span(),
                format!(
                    "`{}` should be called `result` when using native_implemented::function macro",
                    result_item_fn.sig.ident
                ),
            ));
        }

        let result_fn_arg_vec: Vec<FnArg> = result_item_fn
            .sig
            .inputs
            .iter()
            .map(|input| match input {
                FnArg::Typed(PatType {
                    pat: box Pat::Ident(PatIdent { .. }),
                    ..
                }) => input.clone(),
                _ => unimplemented!(
                    "result function is not expected to have argument like {:?}",
                    input
                ),
            })
            .collect();

        let result_arity = result_fn_arg_vec.len();

        let (process, native_fn_arg_vec) = if result_arity == ((arity + 1) as usize) {
            let process = match result_item_fn.sig.inputs.first().unwrap() {
                FnArg::Typed(PatType { ty, .. }) => match **ty {
                    Type::Reference(TypeReference {
                        elem:
                            box Type::Path(TypePath {
                                path: Path { ref segments, .. },
                                ..
                            }),
                        ..
                    }) => {
                        let PathSegment { ident, .. } = segments.last().unwrap();

                        match ident.to_string().as_ref() {
                            "Process" => Process::Ref,
                            s => unimplemented!(
                                "Extracting result function process from reference ident like {:?}",
                                s
                            ),
                        }
                    }
                    Type::Path(TypePath {
                        path: Path { ref segments, .. },
                        ..
                    }) => {
                        let PathSegment { ident, arguments } = segments.last().unwrap();

                        match ident.to_string().as_ref() {
                            "Arc" => match arguments {
                                PathArguments::AngleBracketed(AngleBracketedGenericArguments { args: punctuated_generic_arguments, .. }) => {
                                    match punctuated_generic_arguments.len() {
                                        1 => {
                                            match punctuated_generic_arguments.first().unwrap() {
                                                GenericArgument::Type(Type::Path(TypePath { path: Path { segments, .. }, .. })) => {
                                                    let PathSegment { ident, .. } = segments.last().unwrap();

                                                    match ident.to_string().as_ref() {
                                                        "Process" => Process::Arc,
                                                        s => unimplemented!(
                                                            "Extracting result function process from reference ident like {:?}",
                                                            s
                                                        ),
                                                    }
                                                }
                                                generic_argument => unimplemented!("Extracting result function process from argument to Arc like {:?}", generic_argument)
                                            }
                                        }
                                        n => unimplemented!("Extracting result function process from {:?} arguments to Arc like {:?}", n, punctuated_generic_arguments)
                                    }
                                }
                                _ => unimplemented!("Extracting result function process from arguments to Arc like {:?}", arguments),
                            }
                            s => unimplemented!(
                                "Extracting result function process from path ident like {:?}",
                                s
                            ),
                        }
                    }
                    _ => {
                        unimplemented!("Extracting result function process from type like {:?}", ty)
                    }
                },
                input => unimplemented!(
                    "Extracting result function process from argument like {:?}",
                    input
                ),
            };

            (process, result_fn_arg_vec[1..].to_vec())
        } else if result_arity == (arity as usize) {
            (Process::None, result_fn_arg_vec)
        } else {
            unreachable!(
                "The Erlang arity of a function should not include the Process argument.  For this result function, an arity of {} is expected if Process is not used or {} if the Process is the first argument",
                arity,
                arity + 1
            );
        };

        let return_type = match result_item_fn.sig.output {
            syn::ReturnType::Type(
                _,
                box Type::Path(TypePath {
                                   path: Path { ref segments, .. },
                                   ..
                               }),
            ) => {
                let PathSegment { ident, .. } = segments.last().unwrap();

                match ident.to_string().as_ref() {
                    "Result" => ReturnType::Result,
                    "Term" => ReturnType::Term,
                    _ => return Err(Error::new(ident.span(), "result function return type is neither Result nor Term"))
                }
            }
            ref output => return Err(Error::new(
                output.span(),
                "result functions must return either liblumen_alloc::erts::exception::Result or liblumen_alloc::erts::term::Term"
            )),
        };

        Ok(Self {
            result: Result {
                process,
                return_type,
            },
            native: Native {
                fn_arg_vec: native_fn_arg_vec,
            },
        })
    }

    pub fn label(result_item_fn: &ItemFn) -> std::result::Result<Self, Error> {
        if result_item_fn.sig.ident != "result" {
            return Err(Error::new(
                result_item_fn.sig.ident.span(),
                format!(
                    "`{}` should be called `result` when using native_implemented_function macro",
                    result_item_fn.sig.ident
                ),
            ));
        }

        let result_fn_arg_vec: Vec<FnArg> = result_item_fn
            .sig
            .inputs
            .iter()
            .map(|input| match input {
                FnArg::Typed(PatType {
                    pat: box Pat::Ident(PatIdent { .. }),
                    ..
                }) => input.clone(),
                _ => unimplemented!(
                    "result function is not expected to have argument like {:?}",
                    input
                ),
            })
            .collect();

        let (process, native_fn_arg_vec) = match result_item_fn.sig.inputs.first().unwrap() {
            FnArg::Typed(PatType { ty, .. }) => match **ty {
                Type::Reference(TypeReference {
                    elem:
                        box Type::Path(TypePath {
                            path: Path { ref segments, .. },
                            ..
                        }),
                    ..
                }) => {
                    let PathSegment { ident, .. } = segments.last().unwrap();

                    match ident.to_string().as_ref() {
                        "Process" => (Process::Ref, result_fn_arg_vec[1..].to_vec()),
                        s => unimplemented!(
                            "Extracting result function process from reference ident like {:?}",
                            s
                        ),
                    }
                }
                Type::Path(TypePath {
                    path: Path { ref segments, .. },
                    ..
                }) => {
                    let PathSegment { ident, arguments } = segments.last().unwrap();

                    match ident.to_string().as_ref() {
                            "Arc" => match arguments {
                                PathArguments::AngleBracketed(AngleBracketedGenericArguments { args: punctuated_generic_arguments, .. }) => {
                                    match punctuated_generic_arguments.len() {
                                        1 => {
                                            match punctuated_generic_arguments.first().unwrap() {
                                                GenericArgument::Type(Type::Path(TypePath { path: Path { segments, .. }, .. })) => {
                                                    let PathSegment { ident, .. } = segments.last().unwrap();

                                                    match ident.to_string().as_ref() {
                                                        "Process" => (Process::Arc, result_fn_arg_vec[1..].to_vec()),
                                                        s => unimplemented!(
                                                            "Extracting result function process from reference ident like {:?}",
                                                            s
                                                        ),
                                                    }
                                                }
                                                generic_argument => unimplemented!("Extracting result function process from argument to Arc like {:?}", generic_argument)
                                            }
                                        }
                                        n => unimplemented!("Extracting result function process from {:?} arguments to Arc like {:?}", n, punctuated_generic_arguments)
                                    }
                                }
                                _ => unimplemented!("Extracting result function process from arguments to Arc like {:?}", arguments),
                            }
                            "Term" => (Process::None, result_fn_arg_vec),
                            s => unimplemented!(
                                "Extracting result function process from path ident like {:?}",
                                s
                            ),
                        }
                }
                _ => unimplemented!("Extracting result function process from type like {:?}", ty),
            },
            input => unimplemented!(
                "Extracting result function process from argument like {:?}",
                input
            ),
        };

        let return_type = match result_item_fn.sig.output {
            syn::ReturnType::Type(
                _,
                box Type::Path(TypePath {
                                   path: Path { ref segments, .. },
                                   ..
                               }),
            ) => {
                let PathSegment { ident, .. } = segments.last().unwrap();

                match ident.to_string().as_ref() {
                    "Result" => ReturnType::Result,
                    "Term" => ReturnType::Term,
                    _ => return Err(Error::new(ident.span(), "result function return type is neither Result nor Term"))
                }
            }
            ref output => return Err(Error::new(
                output.span(),
                "result functions must return either liblumen_alloc::erts::exception::Result or liblumen_alloc::erts::term::Term"
            )),
        };

        Ok(Self {
            result: Result {
                process,
                return_type,
            },
            native: Native {
                fn_arg_vec: native_fn_arg_vec,
            },
        })
    }

    pub fn arity(&self) -> u8 {
        self.native.arity()
    }

    fn const_arity(&self) -> proc_macro2::TokenStream {
        let arity = &self.arity();

        quote! {
           pub const ARITY: liblumen_alloc::Arity = #arity;
        }
    }

    fn const_native(&self) -> proc_macro2::TokenStream {
        let native_variant = self.native_variant();

        quote! {
             pub const NATIVE: liblumen_alloc::erts::process::Native = #native_variant;
        }
    }

    pub fn native_fn(&self) -> proc_macro2::TokenStream {
        let mut result_argument_ident: Vec<Box<dyn ToTokens>> = match self.result.process {
            Process::Arc => vec![Box::new(quote! { arc_process.clone() })],
            Process::Ref => vec![Box::new(quote! { &arc_process })],
            Process::None => vec![],
        };

        result_argument_ident.extend(
            self.native
                .fn_arg_vec
                .iter()
                .map(fn_arg_to_ident)
                .map(|ident| -> Box<dyn ToTokens> { Box::new(ident) }),
        );

        let native_fn_arg = &self.native.fn_arg_vec;

        let result_call = match self.result.return_type {
            ReturnType::Result => {
                quote! {
                     arc_process.return_status(result(#(#result_argument_ident),*))
                }
            }
            ReturnType::Term => {
                quote! {
                    result(#(#result_argument_ident),*)
                }
            }
        };

        quote! {
            pub extern "C" fn native(#(#native_fn_arg),*) -> Term {
                let arc_process = crate::runtime::process::current_process();
                arc_process.reduce();

                #result_call
            }
        }
    }

    fn native_variant(&self) -> proc_macro2::TokenStream {
        match self.arity() {
            0 => quote! {
                liblumen_alloc::erts::process::Native::Zero(native)
            },
            1 => quote! {
                liblumen_alloc::erts::process::Native::One(native)
            },
            2 => quote! {
                liblumen_alloc::erts::process::Native::Two(native)
            },
            3 => quote! {
                liblumen_alloc::erts::process::Native::Three(native)
            },
            4 => quote! {
                liblumen_alloc::erts::process::Native::Four(native)
            },
            5 => quote! {
                liblumen_alloc::erts::process::Native::Five(native)
            },
            arity => unimplemented!("Don't know how to convert arity ({}) to Native", arity),
        }
    }
}
