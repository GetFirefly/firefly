//#![deny(warnings)]
// `Alloc`
#![feature(allocator_api)]
#![feature(type_ascription)]

mod heap;
mod module;
mod start;

//use liblumen_alloc::erts::process::code::stack::frame::Placement;

//use lumen_runtime::process::spawn::options::Options;
use lumen_runtime::scheduler::Scheduler;
use lumen_runtime::system;

use lumen_web::wait;

use liblumen_eir_interpreter::VM;

use libeir_diagnostics::{ColorChoice, Emitter, StandardStreamEmitter};

use libeir_ir::Module;

use libeir_passes::PassManager;

use libeir_syntax_erl::ast::Module as ErlAstModule;
use libeir_syntax_erl::lower_module;
use libeir_syntax_erl::{Parse, ParseConfig, Parser};

use liblumen_alloc::erts::term::Atom;

use wasm_bindgen::prelude::*;

//use crate::elixir::chain::{console_1, dom_1};
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

//#[wasm_bindgen]
//pub fn log_to_console(count: usize) -> js_sys::Promise {
//    run(count, Output::Console)
//}
//
//#[wasm_bindgen]
//pub fn log_to_dom(count: usize) -> js_sys::Promise {
//    run(count, Output::Dom)
//}

enum Output {
    Console,
    Dom,
}

fn run(count: usize, output: Output) -> js_sys::Promise {
    let arc_scheduler = Scheduler::current();
    // Don't register, so that tests can run concurrently
    let parent_arc_process = arc_scheduler.spawn_init(0).unwrap();

    panic!()

    //let mut options: Options = Default::default();
    //options.min_heap_size = Some(79 + count * 5);

    //wait::with_return_0::spawn(
    //    &parent_arc_process,
    //    options,
    //|child_process| {
    //        let count_term = child_process.integer(count)?;

    //        match output {
    //            Output::Console => {

    //                // if this fails use a bigger sized heap
    //                console_1::place_frame_with_arguments(child_process, Placement::Push,
    // count_term)            }
    //            Output::Dom => {
    //                // if this fails use a bigger sized heap
    //                dom_1::place_frame_with_arguments(child_process, Placement::Push, count_term)
    //            }
    //        }
    //    })
    //// if this fails use a bigger sized heap
    //.unwrap()
}
