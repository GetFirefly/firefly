#![deny(warnings)]
// `Alloc`
#![feature(allocator_api)]
#![feature(type_ascription)]

mod heap;
mod module;
mod start;

use lumen_runtime::system;

use liblumen_eir_interpreter::VM;

use libeir_diagnostics::{ColorChoice, Emitter, StandardStreamEmitter};

use libeir_ir::Module;

use libeir_passes::PassManager;

use libeir_syntax_erl::ast::Module as ErlAstModule;
use libeir_syntax_erl::lower_module;
use libeir_syntax_erl::{Parse, ParseConfig, Parser};

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

    lumen_web::start();

    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_lumen_web_window());
    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_lumen_web_document());
    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_lumen_web_element());
    VM.modules
        .write()
        .unwrap()
        .register_native_module(module::make_lumen_web_node());

    system::io::puts("initialized");
}

fn parse<T>(input: &str, config: ParseConfig) -> (T, Parser)
where
    T: Parse<T>,
{
    let parser = Parser::new(config);
    let errs = match parser.parse_string::<&str, T>(input) {
        Ok(ast) => return (ast, parser),
        Err(errs) => errs,
    };
    let emitter =
        StandardStreamEmitter::new(ColorChoice::Auto).set_codemap(parser.config.codemap.clone());
    for err in errs.iter() {
        emitter.diagnostic(&err.to_diagnostic()).unwrap();
    }
    panic!("parse failed");
}

pub fn lower(input: &str, config: ParseConfig) -> Result<Module, ()> {
    let (parsed, parser): (ErlAstModule, _) = parse(input, config);
    let (res, messages) = lower_module(&parsed);

    let emitter =
        StandardStreamEmitter::new(ColorChoice::Auto).set_codemap(parser.config.codemap.clone());
    for err in messages.iter() {
        emitter.diagnostic(&err.to_diagnostic()).unwrap();
    }

    res
}

#[wasm_bindgen]
pub fn compile_erlang_module(text: &str) {
    let config = ParseConfig::default();
    let mut eir_mod = lower(text, config).unwrap();

    for fun in eir_mod.functions.values() {
        fun.graph_validate_global();
    }

    let mut pass_manager = PassManager::default();
    pass_manager.run(&mut eir_mod);

    for fun in eir_mod.functions.values() {
        fun.graph_validate_global();
    }

    system::io::puts(&format!("Compiled and registered {}", eir_mod.name));
    VM.modules.write().unwrap().register_erlang_module(eir_mod);
}
