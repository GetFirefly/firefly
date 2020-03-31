#![deny(warnings)]
// `Alloc`
#![feature(allocator_api)]
#![feature(type_ascription)]

mod heap;
mod module;
mod start;

use lumen_rt_full::sys;

use lumen_interpreter::VM;

use libeir_ir::Module;

use libeir_passes::PassManager;

use libeir_syntax_erl::ast::Module as ErlAstModule;
use libeir_syntax_erl::lower_module;
use libeir_syntax_erl::{Parse, ParseConfig, Parser};

use libeir_util_parse::{ArcCodemap, Errors};

use wasm_bindgen::prelude::*;

use crate::start::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn start() {
    set_panic_hook();

    // Dereference lazy_static to force initialization
    &*VM;

    liblumen_web::start();

    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_liblumen_web_window());
    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_liblumen_web_document());
    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_liblumen_web_element());
    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_liblumen_web_node());

    sys::io::puts("initialized");
}

fn parse<T>(input: &str, config: ParseConfig) -> (T, ArcCodemap)
where
    T: Parse<T>,
{
    let parser = Parser::new(config);
    let mut errors = Errors::new();
    let codemap: ArcCodemap = Default::default();

    if let Ok(ast) = parser.parse_string::<&str, T>(&mut errors, &codemap, input) {
        (ast, codemap)
    } else {
        errors.print(&codemap);
        panic!("parse failed");
    }
}

pub fn lower(input: &str, config: ParseConfig) -> Result<Module, ()> {
    let (parsed, codemap): (ErlAstModule, _) = parse(input, config);
    let mut errors = Errors::new();
    let res = lower_module(&mut errors, &codemap, &parsed);
    errors.print(&codemap);

    res
}

#[wasm_bindgen]
pub fn compile_erlang_module(text: &str) {
    let config = ParseConfig::default();
    let mut eir_mod = lower(text, config).unwrap();

    for function_definition in eir_mod.function_iter() {
        function_definition.function().graph_validate_global();
    }

    let mut pass_manager = PassManager::default();
    pass_manager.run(&mut eir_mod);

    for function_definition in eir_mod.function_iter() {
        function_definition.function().graph_validate_global();
    }

    sys::io::puts(&format!("Compiled and registered {}", eir_mod.name()));
    VM.modules.write().unwrap().register_erlang_module(eir_mod);
}
