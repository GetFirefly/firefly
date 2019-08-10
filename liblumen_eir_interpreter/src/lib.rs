use std::sync::Arc;

use lumen_runtime::system;
use lumen_runtime::code::apply_fn;
use lumen_runtime::scheduler::Scheduler;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::erts::term::{ Term, Atom, atom_unchecked };
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::process::{heap, next_heap_size, Status};

mod exec;
mod module;
mod vm;
mod code;
mod native;

use self::vm::VMState;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref VM: VMState = VMState::new();
}

pub fn call_erlang(
    proc: Arc<ProcessControlBlock>,
    module: Atom,
    function: Atom,
    args: &[Term],
) -> std::result::Result<(), ()>
{
    let return_ok = {
        let mfa = ModuleFunctionArity {
            module: Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap(),
            function: Atom::try_from_str("return_ok").unwrap(),
            arity: 1,
        };
        proc.closure(
            proc.pid_term(),
            mfa.into(),
            crate::code::return_ok,
            vec![],
        ).unwrap()
    };
    let return_throw = {
        let mfa = ModuleFunctionArity {
            module: Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap(),
            function: Atom::try_from_str("return_throw").unwrap(),
            arity: 3,
        };
        proc.closure(
            proc.pid_term(),
            mfa.into(),
            crate::code::return_throw,
            vec![],
        ).unwrap()
    };

    let mut args_vec = vec![return_ok, return_throw];
    args_vec.extend(args.iter().cloned());

    let arguments = proc
        .list_from_slice(&args_vec)
    // if not enough memory here, resize `spawn_init` heap
        .unwrap();

    let heap_size = next_heap_size(50000);
    // if this fails the entire tab is out-of-memory
    let heap = heap(heap_size).unwrap();

    let run_arc_process = Scheduler::spawn(
        &proc,
        module,
        function,
        arguments,
        apply_fn(),
        heap,
        heap_size)
    // if this fails, don't use `default_heap` and instead use a bigger sized heap
        .unwrap();

    loop {
        let ran = Scheduler::current().run_through(&run_arc_process);

        match *run_arc_process.status.read() {
            Status::Exiting(ref exception) => match exception {
                exception::runtime::Exception {
                    class: exception::runtime::Class::Exit,
                    reason,
                    ..
                } => {
                    if *reason != atom_unchecked("normal") {
                        return Err(());
                        //panic!("ProcessControlBlock exited: {:?}", reason);
                    } else {
                        return Ok(());
                    }
                }
                _ => {
                    return Err(());
                    //panic!(
                    //    "ProcessControlBlock exception: {:?}\n{:?}",
                    //    exception,
                    //    run_arc_process.stacktrace()
                    //);
                }
            },
            Status::Waiting => {
                if ran {
                    system::io::puts(&format!(
                        "WAITING Run queues len = {:?}",
                        Scheduler::current().run_queues_len()
                    ));
                } else {
                    panic!(
                        "{:?} did not run.  Deadlock likely in {:#?}",
                        run_arc_process,
                        Scheduler::current()
                    );
                }
            }
            Status::Runnable => {
                system::io::puts(&format!(
                    "RUNNABLE Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
            }
            Status::Running => {
                system::io::puts(&format!(
                    "RUNNING Run queues len = {:?}",
                    Scheduler::current().run_queues_len()
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::path::Path;

    use lumen_runtime::registry;
    use lumen_runtime::system;
    use lumen_runtime::scheduler::Scheduler;
    use lumen_runtime::code::apply_fn;
    use liblumen_alloc::erts::ModuleFunctionArity;
    use liblumen_alloc::erts::exception;
    use liblumen_alloc::erts::term::{ Atom, atom_unchecked };
    use liblumen_alloc::erts::process::{heap, next_heap_size, Status};

    use libeir_ir::{ Module, FunctionIdent };
    use libeir_syntax_erl::{ Parse, Parser, ParseConfig };
    use libeir_syntax_erl::ast::{ Module as ErlAstModule };
    use libeir_syntax_erl::lower_module;
    use libeir_intern::Ident;
    use libeir_diagnostics::{ Emitter, StandardStreamEmitter, ColorChoice };
    use libeir_passes::PassManager;

    use super::VM;
    use super::call_erlang;

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
        let emitter = StandardStreamEmitter::new(ColorChoice::Auto)
            .set_codemap(parser.config.codemap.clone());
        for err in errs.iter() {
            emitter.diagnostic(&err.to_diagnostic()).unwrap();
        }
        panic!("parse failed");
    }

    fn lower_file<P>(path: P, config: ParseConfig) -> Result<Module, ()>
    where
        P: AsRef<Path>
    {
        let (parsed, parser): (ErlAstModule, _) = parse_file(path, config);
        let (res, messages) = lower_module(&parsed);

        let emitter = StandardStreamEmitter::new(ColorChoice::Auto)
            .set_codemap(parser.config.codemap.clone());
        for err in messages.iter() {
            emitter.diagnostic(&err.to_diagnostic()).unwrap();
        }

        res
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
    fn nonexistent_function_call() {
        &*VM;

        let init_atom = Atom::try_from_str("init").unwrap();
        let init_arc_process = registry::atom_to_process(&init_atom).unwrap();

        let module = Atom::try_from_str("foo").unwrap();
        let function = Atom::try_from_str("bar").unwrap();

        call_erlang(init_arc_process, module, function, &[]).err().unwrap();
    }

    #[test]
    fn simple_function() {
        &*VM;

        let init_atom = Atom::try_from_str("init").unwrap();
        let init_arc_process = registry::atom_to_process(&init_atom).unwrap();

        let module = Atom::try_from_str("simple_function_test").unwrap();
        let function = Atom::try_from_str("run").unwrap();

        let config = ParseConfig::default();
        let mut eir_mod = lower("
-module(simple_function_test).

run() -> yay.
", config).unwrap();

        for fun in eir_mod.functions.values() {
            fun.graph_validate_global();
        }

        let mut pass_manager = PassManager::default();
        pass_manager.run(&mut eir_mod);

        VM.modules.write().unwrap().register_erlang_module(eir_mod);

        call_erlang(init_arc_process, module, function, &[]).ok().unwrap();
    }

    #[test]
    fn fib() {
        &*VM;

        let init_atom = Atom::try_from_str("init").unwrap();
        let init_arc_process = registry::atom_to_process(&init_atom).unwrap();

        let module = Atom::try_from_str("fib").unwrap();
        let function = Atom::try_from_str("fib").unwrap();

        let config = ParseConfig::default();
        let mut eir_mod = lower("
-module(fib).

fib(X) when X < 2 -> 1;
fib(X) -> fib(X - 1) + fib(X-2).
", config).unwrap();

        for fun in eir_mod.functions.values() {
            fun.graph_validate_global();
        }

        let mut pass_manager = PassManager::default();
        pass_manager.run(&mut eir_mod);

        VM.modules.write().unwrap().register_erlang_module(eir_mod);

        let int = init_arc_process.integer(5).unwrap();
        call_erlang(init_arc_process, module, function, &[int]).ok().unwrap();
    }

}
