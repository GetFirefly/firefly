use std::path::Path;

use clap::{App, Arg};

use libeir_diagnostics::{ColorChoice, Emitter, StandardStreamEmitter};

use libeir_ir::{FunctionIdent, Module};

use libeir_passes::PassManager;

use libeir_syntax_erl::ast::Module as ErlAstModule;
use libeir_syntax_erl::lower_module;
use libeir_syntax_erl::{Parse, ParseConfig, Parser};

use liblumen_eir_interpreter::call_result::call_run_erlang;
use liblumen_eir_interpreter::VM;

use liblumen_alloc::erts::term::prelude::Atom;

use lumen_runtime::scheduler::Scheduler;

fn parse_file<T, P>(path: P, config: ParseConfig) -> (T, Parser)
where
    T: Parse<T>,
    P: AsRef<Path>,
{
    let parser = Parser::new(config);
    let errs = match parser.parse_file::<_, T>(path) {
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

fn lower_file<P>(path: P, config: ParseConfig) -> Result<Module, ()>
where
    P: AsRef<Path>,
{
    let (parsed, parser): (ErlAstModule, _) = parse_file(path, config);
    let (res, messages) = lower_module(&parser.config.codemap.lock().unwrap(), &parsed);

    let emitter =
        StandardStreamEmitter::new(ColorChoice::Auto).set_codemap(parser.config.codemap.clone());
    for err in messages.iter() {
        emitter.diagnostic(&err.to_diagnostic()).unwrap();
    }

    res
}

fn main() {
    let matches = App::new("Lumen Eir Interpreter CLI")
        .version("alpha")
        .arg(
            Arg::from_usage("<LOAD_ERL_FILES> 'load files into the interpreter'")
                .multiple(true)
                .required(false),
        )
        .arg(
            Arg::from_usage("<FUN_IDENT> -i,--ident <IDENT> 'select single function'")
                .required(true),
        )
        .get_matches();

    let ident = FunctionIdent::parse(matches.value_of("FUN_IDENT").unwrap()).unwrap();

    &*VM;

    let arc_scheduler = Scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    let module = Atom::try_from_str(&ident.module.as_str()).unwrap();
    let function = Atom::try_from_str(&ident.name.as_str()).unwrap();
    assert!(ident.arity == 0);

    for file in matches.values_of("LOAD_ERL_FILES").unwrap() {
        let config = ParseConfig::default();
        let mut eir_mod = lower_file(file, config).unwrap();

        for fun_def in eir_mod.function_iter() {
            let fun = fun_def.function();
            fun.graph_validate_global();
        }

        let mut pass_manager = PassManager::default();
        pass_manager.run(&mut eir_mod);

        VM.modules.write().unwrap().register_erlang_module(eir_mod);
    }

    let res = call_run_erlang(init_arc_process, module, function, &[]);
    println!("Returned with {:?}", res.result);
}
