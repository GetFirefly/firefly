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
pub fn native_implemented_function(
    function_arity_token_stream: TokenStream,
    native_token_stream: TokenStream,
) -> TokenStream {
    let function_arity = parse_macro_input!(function_arity_token_stream as FunctionArity);
    let native_item_fn = parse_macro_input!(native_token_stream as ItemFn);

    match Signatures::new(&native_item_fn, function_arity.arity) {
        Ok(signatures) => {
            let place_frame_with_arguments = signatures.place_frame_with_arguments();
            let code = signatures.code();
            let arity = function_arity.arity();
            let export = signatures.export();
            let frame = frame();
            let function = function_arity.function();
            let module_function_arity = signatures.module_function_arity();

            let all_tokens = quote! {
                #place_frame_with_arguments

                // Private

                #code
                #arity
                #export
                #frame
                #function
                #module_function_arity
                #native_item_fn
            };

            all_tokens.into()
        }
        Err(error) => error.to_compile_error().into(),
    }
}

fn frame() -> proc_macro2::TokenStream {
    quote! {
        fn frame() -> liblumen_alloc::erts::process::code::stack::frame::Frame {
            liblumen_alloc::erts::process::code::stack::frame::Frame::new(module_function_arity(), code)
        }
    }
}

struct Code {
    argument_ident_vec: Vec<Ident>,
}

struct Native {
    process: Process,
    return_type: ReturnType,
}

enum Process {
    Arc,
    Ref,
    None,
}

enum ReturnType {
    Result,
    Term,
}

struct Signatures {
    native: Native,
    code: Code,
}

impl Signatures {
    pub fn new(native_item_fn: &ItemFn, arity: u8) -> Result<Self, Error> {
        let native_argument_ident_vec: Vec<Ident> = native_item_fn
            .sig
            .inputs
            .iter()
            .map(|input| match input {
                FnArg::Typed(PatType {
                    pat: box Pat::Ident(PatIdent { ident, .. }),
                    ..
                }) => ident.clone(),
                _ => unimplemented!(
                    "native function is not expected to have argument like {:?}",
                    input
                ),
            })
            .collect();

        let native_arity = native_argument_ident_vec.len();

        let (process, code_argument_ident_vec) = if native_arity == ((arity + 1) as usize) {
            let process = match native_item_fn.sig.inputs.first().unwrap() {
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
                                "Extracting native function process from reference ident like {:?}",
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
                                                            "Extracting native function process from reference ident like {:?}",
                                                            s
                                                        ),
                                                    }
                                                }
                                                generic_argument => unimplemented!("Extracting native function process from argument to Arc like {:?}", generic_argument)
                                            }
                                        }
                                        n => unimplemented!("Extracting native function process from {:?} arguments to Arc like {:?}", n, punctuated_generic_arguments)
                                    }
                                }
                                _ => unimplemented!("Extracting native function process from arguments to Arc like {:?}", arguments),
                            }
                            s => unimplemented!(
                                "Extracting native function process from path ident like {:?}",
                                s
                            ),
                        }
                    }
                    _ => {
                        unimplemented!("Extracting native function process from type like {:?}", ty)
                    }
                },
                input => unimplemented!(
                    "Extracting native function process from argument like {:?}",
                    input
                ),
            };

            (process, native_argument_ident_vec[1..].to_vec())
        } else if native_arity == (arity as usize) {
            (Process::None, native_argument_ident_vec)
        } else {
            unreachable!(
                "The Erlang arity of a function should not include the Process argument.  For this native function, an arity of {} is expected if Process is not used or {} if the Process is the first argument",
                arity,
                arity + 1
            );
        };

        let return_type = match native_item_fn.sig.output {
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
                    _ => return Err(Error::new(ident.span(), "native function return type is neither Result nor Term"))
                }
            }
            ref output => return Err(Error::new(
                output.span(),
                "native functions must return either liblumen_alloc::erts::exception::Result or liblumen_alloc::erts::term::Term"
            )),
        };

        Ok(Self {
            native: Native {
                process,
                return_type,
            },
            code: Code {
                argument_ident_vec: code_argument_ident_vec,
            },
        })
    }

    pub fn code(&self) -> proc_macro2::TokenStream {
        let native_argument_ident: Vec<Box<dyn ToTokens>> = match self.native.process {
            Process::Arc => {
                let mut native_argument_vec: Vec<Box<dyn ToTokens>> =
                    Vec::with_capacity(self.code.argument_ident_vec.len() + 1);

                native_argument_vec.push(Box::new(quote! { arc_process.clone() }));

                for ident in self.code.argument_ident_vec.iter() {
                    native_argument_vec.push(Box::new(ident.clone()))
                }

                native_argument_vec
            }
            Process::Ref => {
                let mut native_argument_vec: Vec<Box<dyn ToTokens>> =
                    Vec::with_capacity(self.code.argument_ident_vec.len() + 1);

                native_argument_vec.push(Box::new(quote! { arc_process }));

                for ident in self.code.argument_ident_vec.iter() {
                    native_argument_vec.push(Box::new(ident.clone()))
                }

                native_argument_vec
            }
            Process::None => {
                let mut native_argument_vec: Vec<Box<dyn ToTokens>> =
                    Vec::with_capacity(self.code.argument_ident_vec.len());

                for ident in self.code.argument_ident_vec.iter() {
                    native_argument_vec.push(Box::new(ident.clone()))
                }

                native_argument_vec
            }
        };

        let stack_peeked_code_argument_ident = &self.code.argument_ident_vec;
        let arguments_len = self.code.argument_ident_vec.len();
        // stack_slot is 1-based
        let stack_peek_index = 1..=arguments_len;

        let native_call = match self.native.return_type {
            ReturnType::Result => {
                quote! {
                     match native(#(#native_argument_ident),*) {
                         Ok(return_value) => {
                             arc_process.return_from_call(#arguments_len, return_value).unwrap();

                             liblumen_alloc::erts::process::Process::call_code(arc_process)
                         }
                         Err(exception) => liblumen_alloc::erts::process::code::result_from_exception(arc_process, #arguments_len, exception),
                     }
                }
            }
            ReturnType::Term => {
                quote! {
                    let return_value = native(#(#native_argument_ident),*);
                    arc_process.return_from_call(#arguments_len, return_value).unwrap();

                    liblumen_alloc::erts::process::Process::call_code(arc_process)
                }
            }
        };

        quote! {
            pub fn code(arc_process: &std::sync::Arc<liblumen_alloc::erts::process::Process>) -> liblumen_alloc::erts::process::code::Result {
                arc_process.reduce();

                #(let #stack_peeked_code_argument_ident = arc_process.stack_peek(#stack_peek_index).unwrap();)*

                #native_call
            }
        }
    }

    pub fn export(&self) -> proc_macro2::TokenStream {
        quote! {
            pub fn export() {
                crate::runtime::code::export::insert(super::module(), function(), ARITY, code);
            }
        }
    }

    pub fn module_function_arity(&self) -> proc_macro2::TokenStream {
        quote! {
            pub fn module_function_arity() -> std::sync::Arc<liblumen_alloc::erts::ModuleFunctionArity> {
                std::sync::Arc::new(liblumen_alloc::erts::ModuleFunctionArity {
                    module: super::module(),
                    function: function(),
                    arity: ARITY,
                })
            }
        }
    }

    pub fn place_frame_with_arguments(&self) -> proc_macro2::TokenStream {
        let argument_ident = &self.code.argument_ident_vec;
        let pushed_argument_ident = self.code.argument_ident_vec.iter().rev();

        quote! {
            pub fn place_frame_with_arguments(
                     process: &liblumen_alloc::erts::process::Process,
                     placement: liblumen_alloc::erts::process::code::stack::frame::Placement,
                     #(#argument_ident: Term),*
                   ) -> liblumen_alloc::erts::exception::AllocResult<()> {
                #(process.stack_push(#pushed_argument_ident)?;)*

                process.place_frame(frame(), placement);

                Ok(())
            }
        }
    }
}

#[derive(Debug)]
struct FunctionArity {
    function: String,
    arity: u8,
}

impl FunctionArity {
    fn arity(&self) -> proc_macro2::TokenStream {
        let arity = &self.arity;

        quote! {
           const ARITY: liblumen_alloc::Arity = #arity;
        }
    }

    fn function(&self) -> proc_macro2::TokenStream {
        let function = &self.function;

        quote! {
            pub fn function() -> liblumen_alloc::erts::term::prelude::Atom {
                liblumen_alloc::erts::term::prelude::Atom::try_from_str(#function).unwrap()
            }
        }
    }
}

impl Parse for FunctionArity {
    fn parse(input: &ParseBuffer) -> syn::parse::Result<Self> {
        if input.is_empty() {
            Err(input.error("function = \"NAME\" required"))
        } else {
            let function: String = if let Ok(ident) = input.parse::<syn::Ident>() {
                ident.to_string()
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
            } else {
                unimplemented!("parse function name from {:?}", input);
            };

            input.parse::<Token![/]>()?;

            let arity_lit_int = input.parse::<LitInt>()?;
            let arity = arity_lit_int.base10_parse()?;

            Ok(FunctionArity { function, arity })
        }
    }
}
