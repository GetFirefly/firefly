use super::call_erlang;
use super::VM;

use libeir_diagnostics::{ColorChoice, Emitter, StandardStreamEmitter};

use libeir_ir::Module;

use libeir_passes::PassManager;

use libeir_syntax_erl::ast::Module as ErlAstModule;
use libeir_syntax_erl::lower_module;
use libeir_syntax_erl::{Parse, ParseConfig, Parser};

use liblumen_alloc::erts::term::Atom;

use lumen_runtime::scheduler::Scheduler;

fn parse<T>(input: &str, config: ParseConfig) -> (T, Parser)
where
    T: Parse<T>,
{
    let parser = Parser::new(config);
    let errs = match parser.parse_string::<&str, T>(input) {
        Ok(ast) => return (ast, parser),
        Err(errs) => errs,
    };
    let emitter = StandardStreamEmitter::new(ColorChoice::Auto)
        .set_codemap(parser.config.codemap.clone());
    for err in errs.iter() {
        emitter.diagnostic(&err.to_diagnostic()).unwrap();
    }
    panic!("parse failed");
}

pub fn lower(input: &str, config: ParseConfig) -> Result<Module, ()> {
    let (parsed, parser): (ErlAstModule, _) = parse(input, config);
    let (res, messages) = lower_module(&parsed);

    let emitter = StandardStreamEmitter::new(ColorChoice::Auto)
        .set_codemap(parser.config.codemap.clone());
    for err in messages.iter() {
        emitter.diagnostic(&err.to_diagnostic()).unwrap();
    }

    res
}

#[test]
#[ignore]
fn nonexistent_function_call() {
    &*VM;

    let arc_scheduler = Scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    let module = Atom::try_from_str("foo").unwrap();
    let function = Atom::try_from_str("bar").unwrap();

    call_erlang(init_arc_process, module, function, &[])
        .err()
        .unwrap();
}

#[test]
fn simple_function() {
    &*VM;

    let arc_scheduler = Scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    let module = Atom::try_from_str("simple_function_test").unwrap();
    let function = Atom::try_from_str("run").unwrap();

    let config = ParseConfig::default();
    let mut eir_mod = lower(
        "
-module(simple_function_test).

run() -> yay.
",
        config,
    )
        .unwrap();

    for fun in eir_mod.functions.values() {
        fun.graph_validate_global();
    }

    let mut pass_manager = PassManager::default();
    pass_manager.run(&mut eir_mod);

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    call_erlang(init_arc_process, module, function, &[])
        .ok()
        .unwrap();
}

#[test]
fn fib() {
    &*VM;

    let arc_scheduler = Scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    let module = Atom::try_from_str("fib").unwrap();
    let function = Atom::try_from_str("fib").unwrap();

    let config = ParseConfig::default();
    let mut eir_mod = lower(
        "
-module(fib).

fib(X) when X < 2 -> 1;
fib(X) -> fib(X - 1) + fib(X-2).
",
        config,
    )
        .unwrap();

    for fun in eir_mod.functions.values() {
        fun.graph_validate_global();
    }

    let mut pass_manager = PassManager::default();
    pass_manager.run(&mut eir_mod);

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let int = init_arc_process.integer(5).unwrap();
    call_erlang(init_arc_process, module, function, &[int])
        .ok()
        .unwrap();
}

//#[test]
//fn fib_gc() {
//    &*VM;
//
//    let arc_scheduler = Scheduler::current();
//    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();
//
//    let module = Atom::try_from_str("fib").unwrap();
//    let function = Atom::try_from_str("fib").unwrap();
//
//    let config = ParseConfig::default();
//    let mut eir_mod = lower(
//        "
//-module(fib).
//
//fib(X) when X < 2 -> 1;
//fib(X) -> fib(X - 1) + fib(X-2).
//",
//        config,
//    )
//        .unwrap();
//
//    for fun in eir_mod.functions.values() {
//        fun.graph_validate_global();
//    }
//
//    let mut pass_manager = PassManager::default();
//    pass_manager.run(&mut eir_mod);
//
//    VM.modules.write().unwrap().register_erlang_module(eir_mod);
//
//    let int = init_arc_process.integer(20).unwrap();
//    call_erlang(init_arc_process, module, function, &[int])
//        .ok()
//        .unwrap();
//}
