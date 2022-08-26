///! This module contains the signatures of all BIFs in SSA IR
///!
///! The purpose of this is to provide more precise type information to Erlang programs
///! by using the BIFs as a source of that information. This can be used to optimize
///! pattern matching, but can also be used to drive type inference bottom-up.
///!
///! NOTE: The types are not perfectly accurate, as currently we don't try to fully represent
///! type specs, however we do make an attempt to be as precise as possible with the accuracy
///! we do have. In the future we'll hopefully be able to make the type information not only
///! richer, but context-sensitive for those situations in which the polymorphism of a given BIF
///! is dependent on constant inputs which we can reason about relatively easily
use std::collections::BTreeMap;

use lazy_static::lazy_static;
use firefly_compiler_macros::{bif, guard_bif};
use firefly_intern::symbols;

use crate::{CallConv, FunctionName, Signature, Visibility};
use crate::{FunctionType, PrimitiveType, TermType, Type};

lazy_static! {
    static ref BIF_SIGNATURES: Vec<Signature> = {
        vec![
            guard_bif!(pub erlang:++/2(list, list) -> list),
            guard_bif!(pub erlang:--/2(list, list) -> list),
            guard_bif!(pub erlang:=:=/2(term, term) -> bool),
            guard_bif!(pub erlang:==/2(term, term) -> bool),
            guard_bif!(pub erlang:=/=/2(term, term) -> bool),
            guard_bif!(pub erlang:/=/2(term, term) -> bool),
            guard_bif!(pub erlang:>=/2(term, term) -> bool),
            guard_bif!(pub erlang:=</2(term, term) -> bool),
            guard_bif!(pub erlang:</2(term, term) -> bool),
            guard_bif!(pub erlang:>/2(term, term) -> bool),
            guard_bif!(pub erlang:and/2(bool, bool) -> bool),
            guard_bif!(pub erlang:andalso/2(bool, bool) -> bool),
            guard_bif!(pub erlang:or/2(bool, bool) -> bool),
            guard_bif!(pub erlang:orelse/2(bool, bool) -> bool),
            guard_bif!(pub erlang:xor/2(bool, bool) -> bool),
            guard_bif!(pub erlang:not/1(bool) -> bool),
            guard_bif!(pub erlang:+/2(number, number) -> number),
            guard_bif!(pub erlang:-/2(number, number) -> number),
            guard_bif!(pub erlang:-/1(number) -> number),
            guard_bif!(pub erlang:*/2(number, number) -> number),
            // NOTE: This bif is the implementation of erlang://2, but with a less problematic name
            guard_bif!(pub erlang:fdiv/2(number, number) -> float),
            guard_bif!(pub erlang:div/2(integer, integer) -> integer),
            guard_bif!(pub erlang:rem/2(integer, integer) -> integer),
            guard_bif!(pub erlang:band/2(integer, integer) -> integer),
            guard_bif!(pub erlang:bor/2(integer, integer) -> integer),
            guard_bif!(pub erlang:bxor/2(integer, integer) -> integer),
            guard_bif!(pub erlang:bsl/2(integer, integer) -> integer),
            guard_bif!(pub erlang:bsr/2(integer, integer) -> integer),
            guard_bif!(pub erlang:bnot/1(integer) -> integer),
            guard_bif!(pub erlang:abs/1(number) -> number),
            bif!(pub erlang:alias/0() -> reference),
            bif!(pub erlang:alias/0(list) -> reference),
            bif!(pub erlang:apply/2(function, list) -> any),
            bif!(pub erlang:apply/3(module, function, list) -> any),
            bif!(pub erlang:atom_to_binary/1(atom) -> binary),
            bif!(pub erlang:atom_to_binary/2(atom, atom) -> binary),
            bif!(pub erlang:atom_to_list/1(atom) -> string),
            guard_bif!(pub erlang:binary_part/2(binary, tuple) -> binary),
            guard_bif!(pub erlang:binary_part/3(binary, non_neg_integer, integer) -> binary),
            bif!(pub erlang:binary_to_atom/1(binary) -> atom),
            bif!(pub erlang:binary_to_atom/2(binary, atom) -> atom),
            bif!(pub erlang:binary_to_existing_atom/1(binary) -> atom),
            bif!(pub erlang:binary_to_existing_atom/2(binary, atom) -> atom),
            bif!(pub erlang:binary_to_float/1(binary) -> float),
            bif!(pub erlang:binary_to_integer/1(binary) -> integer),
            bif!(pub erlang:binary_to_integer/2(binary, pos_integer) -> integer),
            bif!(pub erlang:binary_to_list/1(binary) -> list),
            bif!(pub erlang:binary_to_list/3(binary, pos_integer, pos_integer) -> list),
            bif!(pub erlang:binary_to_term/1(binary) -> term),
            bif!(pub erlang:binary_to_term/2(binary, list) -> term),
            guard_bif!(pub erlang:bit_size/1(bitstring) -> non_neg_integer),
            bif!(pub erlang:bitstring_to_list/1(bitstring) -> list),
            guard_bif!(pub erlang:byte_size/1(bitstring) -> non_neg_integer),
            guard_bif!(pub erlang:ceil/1(number) -> integer),
            bif!(pub erlang:date/0() -> tuple),
            bif!(pub erlang:demonitor/1(reference) -> boolean),
            bif!(pub erlang:demonitor/2(reference, list) -> boolean),
            bif!(pub erlang:disconnect_node/1(atom) -> atom),
            guard_bif!(pub erlang:element/2(non_neg_integer, tuple) -> term),
            bif!(pub erlang:erase/0() -> list),
            bif!(pub erlang:erase/1() -> term),
            bif!(pub erlang:error/1(term) -> term),
            bif!(pub erlang:error/2(term, term) -> term),
            bif!(pub erlang:error/3(term, term, list) -> term),
            bif!(pub erlang:exit/1(term) -> term),
            bif!(pub erlang:exit/2(term, term) -> term),
            guard_bif!(pub erlang:float/1(number) -> float),
            bif!(pub erlang:float_to_binary/1(float) -> binary),
            bif!(pub erlang:float_to_binary/2(float, list) -> binary),
            bif!(pub erlang:float_to_list/1(float) -> list),
            bif!(pub erlang:float_to_list/2(float, list) -> list),
            guard_bif!(pub erlang:floor/1(number) -> integer),
            bif!(pub erlang:garbage_collect/0() -> boolean),
            bif!(pub erlang:garbage_collect/1(pid) -> boolean),
            bif!(pub erlang:garbage_collect/2(pid, list) -> term),
            bif!(pub erlang:get/0() -> list),
            bif!(pub erlang:get/1(term) -> term),
            bif!(pub erlang:get_keys/0() -> list),
            bif!(pub erlang:get_keys/1(term) -> list),
            bif!(pub erlang:group_leader/0() -> pid),
            bif!(pub erlang:group_leader/2(pid, pid) -> boolean),
            bif!(pub erlang:halt/0() -> no_return),
            bif!(pub erlang:halt/1(term) -> no_return),
            bif!(pub erlang:halt/2(term, list) -> no_return),
            guard_bif!(pub erlang:hd/1(list) -> term),
            bif!(pub erlang:integer_to_binary/1(integer) -> binary),
            bif!(pub erlang:integer_to_binary/2(integer, pos_integer) -> binary),
            bif!(pub erlang:integer_to_list/1(integer) -> string),
            bif!(pub erlang:integer_to_list/2(integer, pos_integer) -> string),
            bif!(pub erlang:iolist_size/1(term) -> non_neg_integer),
            bif!(pub erlang:iolist_to_binary/1(term) -> binary),
            bif!(pub erlang:is_alive/0() -> boolean),
            guard_bif!(pub erlang:is_atom/1(any) -> boolean),
            guard_bif!(pub erlang:is_binary/1(any) -> boolean),
            guard_bif!(pub erlang:is_bitstring/1(any) -> boolean),
            guard_bif!(pub erlang:is_boolean/1(any) -> boolean),
            guard_bif!(pub erlang:is_float/1(any) -> boolean),
            guard_bif!(pub erlang:is_function/1(any) -> boolean),
            guard_bif!(pub erlang:is_function/2(any, arity) -> boolean),
            guard_bif!(pub erlang:is_integer/1(any) -> boolean),
            guard_bif!(pub erlang:is_list/1(any) -> boolean),
            guard_bif!(pub erlang:is_map/1(any) -> boolean),
            guard_bif!(pub erlang:is_map_key/2(any, map) -> boolean),
            guard_bif!(pub erlang:is_number/1(any) -> boolean),
            guard_bif!(pub erlang:is_pid/1(any) -> boolean),
            guard_bif!(pub erlang:is_port/1(any) -> boolean),
            bif!(pub erlang:is_process_alive/1(pid) -> boolean),
            guard_bif!(pub erlang:is_record/2(any, atom) -> boolean),
            guard_bif!(pub erlang:is_record/3(any, atom, non_neg_integer) -> boolean),
            guard_bif!(pub erlang:is_reference/1(any) -> boolean),
            guard_bif!(pub erlang:is_tuple/1(any) -> boolean),
            guard_bif!(pub erlang:length/1(list) -> non_neg_integer),
            bif!(pub erlang:link/1(term) -> boolean),
            bif!(pub erlang:list_to_atom/1(string) -> atom),
            bif!(pub erlang:list_to_binary/1(iolist) -> binary),
            bif!(pub erlang:list_to_bitstring/1(maybe_improper_list) -> bitstring),
            bif!(pub erlang:list_to_existing_atom/1(string) -> atom),
            bif!(pub erlang:list_to_float/1(string) -> float),
            bif!(pub erlang:list_to_integer/1(string) -> integer),
            bif!(pub erlang:list_to_integer/2(string, pos_integer) -> integer),
            bif!(pub erlang:list_to_pid/1(string) -> pid),
            bif!(pub erlang:list_to_port/1(string) -> port),
            bif!(pub erlang:list_to_ref/1(string) -> reference),
            bif!(pub erlang:list_to_tuple/1(list) -> tuple),
            bif!(pub erlang:load_nif/2(string, term) -> term),
            bif!(pub erlang:make_ref/0() -> reference),
            guard_bif!(pub erlang:map_get/2(any, map) -> any),
            guard_bif!(pub erlang:map_size/1(map) -> non_neg_integer),
            guard_bif!(pub erlang:match_fail/2(atom, term) -> term),
            bif!(pub erlang:max/2(term, term) -> term),
            bif!(pub erlang:min/2(term, term) -> term),
            bif!(pub erlang:monitor/2(atom, term) -> reference),
            bif!(pub erlang:monitor/3(atom, term, list) -> reference),
            bif!(pub erlang:monitor_node/2(node, boolean) -> boolean),
            bif!(pub erlang:monitor_node/3(node, boolean, list) -> boolean),
            guard_bif!(pub erlang:node/0() -> node),
            guard_bif!(pub erlang:node/1(term) -> node),
            bif!(pub erlang:nodes/0() -> list),
            bif!(pub erlang:nodes/1(term) -> list),
            bif!(pub erlang:now/0() -> timestamp),
            bif!(pub erlang:open_port/2(term, list) -> port),
            bif!(pub erlang:pid_to_list/1(pid) -> string),
            bif!(pub erlang:port_close/1(term) -> boolean),
            bif!(pub erlang:port_command/2(term, term) -> boolean),
            bif!(pub erlang:port_command/3(term, term, list) -> boolean),
            bif!(pub erlang:port_connect/2(term, pid) -> boolean),
            bif!(pub erlang:port_control/3(term, integer, term) -> term),
            bif!(pub erlang:port_to_list/1(port) -> string),
            bif!(pub erlang:process_flag/2(term, term) -> term),
            bif!(pub erlang:process_flag/3(pid, atom, non_neg_integer) -> non_neg_integer),
            bif!(pub erlang:process_info/1(pid) -> term),
            bif!(pub erlang:process_info/2(pid, term) -> term),
            bif!(pub erlang:processes/0() -> list),
            bif!(pub erlang:put/2(term, term) -> term),
            bif!(pub erlang:raise/2(any, trace) -> term),
            bif!(pub erlang:raise/3(atom, any, list) -> term),
            bif!(pub erlang:ref_to_list/1(reference) -> string),
            bif!(pub erlang:register/2(atom, term) -> boolean),
            bif!(pub erlang:registered/0() -> list),
            guard_bif!(pub erlang:round/1(number) -> integer),
            bif!(pub erlang:setelement/3(pos_integer, tuple, term) -> tuple),
            guard_bif!(pub erlang:self/0() -> pid),
            guard_bif!(pub erlang:size/1(term) -> non_neg_integer),
            bif!(pub erlang:spawn/1(function) -> pid),
            bif!(pub erlang:spawn/2(node, function) -> pid),
            bif!(pub erlang:spawn/3(module, atom, list) -> pid),
            bif!(pub erlang:spawn/4(node, module, atom, list) -> pid),
            bif!(pub erlang:spawn_link/1(function) -> pid),
            bif!(pub erlang:spawn_link/2(node, function) -> pid),
            bif!(pub erlang:spawn_link/3(module, atom, list) -> pid),
            bif!(pub erlang:spawn_link/4(node, module, atom, list) -> pid),
            bif!(pub erlang:spawn_monitor/1(function) -> spawn_monitor),
            bif!(pub erlang:spawn_monitor/2(node, function) -> spawn_monitor),
            bif!(pub erlang:spawn_monitor/3(module, atom, list) -> spawn_monitor),
            bif!(pub erlang:spawn_monitor/4(node, module, atom, list) -> spawn_monitor),
            bif!(pub erlang:spawn_opt/1(function) -> term),
            bif!(pub erlang:spawn_opt/2(node, function) -> term),
            bif!(pub erlang:spawn_opt/3(module, atom, list) -> term),
            bif!(pub erlang:spawn_opt/4(node, module, atom, list) -> term),
            bif!(pub erlang:spawn_opt/5(node, module, atom, list, list) -> term),
            bif!(pub erlang:spawn_request/1(function) -> reference),
            bif!(pub erlang:spawn_request/2(term, term) -> reference),
            bif!(pub erlang:spawn_request/3(term, term, term) -> reference),
            bif!(pub erlang:spawn_request/4(term, term, term, list) -> reference),
            bif!(pub erlang:spawn_request/5(node, module, atom, list, list) -> reference),
            bif!(pub erlang:spawn_request_abandon/1(reference) -> boolean),
            bif!(pub erlang:split_binary/2(binary, non_neg_integer) -> binary_split),
            bif!(pub erlang:statistics/1(atom) -> term),
            bif!(pub erlang:term_to_binary/1(term) -> binary),
            bif!(pub erlang:term_to_binary/2(term, list) -> binary),
            bif!(pub erlang:term_to_iovec/1(term) -> list),
            bif!(pub erlang:term_to_iovec/2(term, list) -> list),
            bif!(pub erlang:throw/1(any) -> term),
            bif!(pub erlang:time/0() -> time),
            guard_bif!(pub erlang:tl/1(nonempty_maybe_improper_list) -> term),
            guard_bif!(pub erlang:trunc/1(number) -> integer),
            guard_bif!(pub erlang:tuple_size/1(tuple) -> non_neg_integer),
            bif!(pub erlang:tuple_to_list/1(tuple) -> list),
            bif!(pub erlang:unlink/1(term) -> boolean),
            bif!(pub erlang:unregister/1(atom) -> boolean),
            bif!(pub erlang:whereis/1(atom) -> term),
            // pub erlang:make_fun/3(atom, atom, int) -> i1, term
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Erlang, symbols::MakeFun, FunctionType::new(vec![Type::Term(TermType::Atom), Type::Term(TermType::Atom), Type::Term(TermType::Integer)], vec![Type::Primitive(PrimitiveType::I1), Type::Term(TermType::Any)])),
            // pub erlang:build_stacktrace/1(exception_trace) -> term
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Erlang, symbols::BuildStacktrace, FunctionType::new(vec![Type::ExceptionTrace], vec![Type::Term(TermType::Any)])),
            // pub erlang:nif_start/0
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Erlang, symbols::NifStart, FunctionType::default()),
            // pub erlang:remove_message/0()
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Erlang, symbols::RemoveMessage, FunctionType::default()),
            // pub erlang:recv_next/0()
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Erlang, symbols::RecvNext, FunctionType::default()),
            // pub erlang:recv_peek_message/0() -> <peek_succeeded, message>
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Erlang, symbols::RecvPeekMessage, FunctionType::new(vec![], vec![Type::Term(TermType::Bool), Type::Term(TermType::Any)])),
            // pub erlang:recv_wait_timeout/1(timeout) -> <is_err, timeout_expired | *exception>
            Signature::new(Visibility::PUBLIC | Visibility::EXTERNAL, CallConv::C, symbols::Erlang, symbols::RecvWaitTimeout, FunctionType::new(vec![Type::Term(TermType::Any)], vec![Type::Primitive(PrimitiveType::I1), Type::Term(TermType::Any)])),
        ]
    };
}

lazy_static! {
    static ref BIF_MAP: BTreeMap<FunctionName, &'static Signature> = {
        let mut bifs = BTreeMap::new();
        for sig in BIF_SIGNATURES.iter() {
            let mfa = sig.mfa();
            bifs.insert(mfa, sig);
        }
        bifs
    };
}

/// Get the Signature matching the provided module/function/arity
pub fn get(mfa: &FunctionName) -> Option<&'static Signature> {
    BIF_MAP.get(mfa).copied()
}

/// Get the Signature matching the provided module/function/arity, or panic if unregistered
pub fn fetch(mfa: &FunctionName) -> &'static Signature {
    match BIF_MAP.get(mfa).copied() {
        Some(sig) => sig,
        None => panic!("unregistered builtin {}", mfa),
    }
}

/// Get a slice containing the Signature of all built-in functions
pub fn all() -> &'static [Signature] {
    BIF_SIGNATURES.as_slice()
}
