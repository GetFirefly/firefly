///! This module contains the signatures of all NIFs in SSA IR
///!
///! The purpose of this is to centralize the registration of native functions that the
///! compiler is aware of and can reason about. If the code generator needs to implement
///! an op using a native function, it should be registered here with the appropriate type
///! signature.
use std::collections::BTreeMap;

use firefly_intern::{symbols, Symbol};
use lazy_static::lazy_static;

use crate::{CallConv, Signature, Visibility};
use crate::{FunctionType, PrimitiveType, TermType, Type};

lazy_static! {
    static ref NIF_SIGNATURES: Vec<Signature> = {
        vec![
            // pub __firefly_make_tuple(isize) -> tuple
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Empty, symbols::NifMakeTuple, FunctionType::new(vec![Type::Process, Type::Primitive(PrimitiveType::Isize)], vec![Type::Term(TermType::Tuple(None))])),
            // pub __firefly_tuple_size(term) -> i1, u32
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Empty, symbols::NifTupleSize, FunctionType::new(vec![Type::Term(TermType::Any)], vec![Type::Primitive(PrimitiveType::I1), Type::Primitive(PrimitiveType::I32)])),
            // pub __firefly_map_empty() -> map
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Empty, symbols::NifMapEmpty, FunctionType::new(vec![Type::Process], vec![Type::Term(TermType::Map)])),
            // pub __firefly_map_put(map, term, term) -> map
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Empty, symbols::NifMapPut, FunctionType::new(vec![Type::Process, Type::Term(TermType::Map), Type::Term(TermType::Any), Type::Term(TermType::Any)], vec![Type::Term(TermType::Map)])),
            // pub __firefly_map_put_mut(map, term, term) -> map
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Empty, symbols::NifMapPutMut, FunctionType::new(vec![Type::Term(TermType::Map), Type::Term(TermType::Any), Type::Term(TermType::Any)], vec![Type::Term(TermType::Map)])),
            // pub __firefly_map_update(map, term, term) -> i1, map
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::Erlang, symbols::Empty, symbols::NifMapUpdate, FunctionType::new(vec![Type::Process, Type::Term(TermType::Map), Type::Term(TermType::Any), Type::Term(TermType::Any)], vec![Type::Primitive(PrimitiveType::I1), Type::Term(TermType::Map)])),
            // pub __firefly_map_update_mut(map, term, term) -> i1, map
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::Erlang, symbols::Empty, symbols::NifMapUpdateMut, FunctionType::new(vec![Type::Term(TermType::Map), Type::Term(TermType::Any), Type::Term(TermType::Any)], vec![Type::Primitive(PrimitiveType::I1), Type::Term(TermType::Map)])),
            // pub __firefly_map_fetch(map, term) -> i1, term
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::Erlang, symbols::Empty, symbols::NifMapFetch, FunctionType::new(vec![Type::Term(TermType::Map), Type::Term(TermType::Any)], vec![Type::Primitive(PrimitiveType::I1), Type::Term(TermType::Any)])),
            // pub __firefly_build_stacktrace(exception_trace) -> term
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Empty, symbols::NifBuildStacktrace, FunctionType::new(vec![Type::Process, Type::ExceptionTrace], vec![Type::Term(TermType::Any)])),
            // pub __firefly_bs_init() -> i1, term
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::Erlang, symbols::Empty, symbols::NifBsInit, FunctionType::new(vec![], vec![Type::Primitive(PrimitiveType::I1), Type::Term(TermType::Any)])),
            // pub __firefly_bs_finish(binary_builder) -> i1, term
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::Erlang, symbols::Empty, symbols::NifBsFinish, FunctionType::new(vec![Type::Process, Type::BinaryBuilder], vec![Type::Primitive(PrimitiveType::I1), Type::Term(TermType::Any)])),
        ]
    };
}

lazy_static! {
    static ref NIF_MAP: BTreeMap<Symbol, &'static Signature> = {
        let mut nifs = BTreeMap::new();
        for sig in NIF_SIGNATURES.iter() {
            let op = sig.name;
            nifs.insert(op, sig);
        }
        nifs
    };
}

/// Get the Signature matching the provided name
pub fn get(op: &Symbol) -> Option<&'static Signature> {
    NIF_MAP.get(op).copied()
}

/// Get the Signature matching the provided name, or panic if unregistered
pub fn fetch(op: &Symbol) -> &'static Signature {
    match NIF_MAP.get(op).copied() {
        Some(sig) => sig,
        None => panic!("unregistered nif {}", op),
    }
}

/// Get a slice containing the Signature of all built-in functions
pub fn all() -> &'static [Signature] {
    NIF_SIGNATURES.as_slice()
}
