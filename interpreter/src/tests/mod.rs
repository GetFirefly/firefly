use super::VM;

use libeir_ir::Module;

use libeir_passes::PassManager;

use libeir_syntax_erl::ast::Module as ErlAstModule;
use libeir_syntax_erl::lower_module;
use libeir_syntax_erl::{Parse, ParseConfig, Parser};

use libeir_util_parse::{ArcCodemap, Errors};

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::scheduler;

fn parse<T>(input: &str, config: ParseConfig) -> (T, ArcCodemap)
where
    T: Parse<T>,
{
    let parser = Parser::new(config);
    let mut errors = Errors::new();
    let codemap: ArcCodemap = Default::default();
    match parser.parse_string::<&str, T>(&mut errors, &codemap, input) {
        Ok(ast) => return (ast, codemap),
        Err(errs) => errs,
    };
    errors.print(&codemap);
    panic!("parse failed");
}

pub fn lower(input: &str, config: ParseConfig) -> Result<Module, ()> {
    let (parsed, codemap): (ErlAstModule, _) = parse(input, config);

    let mut errors = Errors::new();
    let res = lower_module(&mut errors, &codemap, &parsed);
    errors.print(&codemap);

    res
}

pub fn compile(input: &str) -> Module {
    let config = ParseConfig::default();
    let mut eir_mod = lower(input, config).unwrap();

    for fun_def in eir_mod.function_iter() {
        let fun = fun_def.function();
        fun.graph_validate_global();
    }

    let mut pass_manager = PassManager::default();
    pass_manager.run(&mut eir_mod);

    eir_mod
}

#[test]
fn simple_function() {
    &*VM;

    let arc_scheduler = scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    liblumen_otp::erlang::apply_3::export();

    let module = Atom::try_from_str("simple_function_test").unwrap();
    let function = Atom::try_from_str("run").unwrap();

    let eir_mod = compile(
        "
-module(simple_function_test).

run() -> yay.
",
    );

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let res = crate::call_result::call_run_erlang(init_arc_process, module, function, &[]);
    assert!(res.result == Ok(Atom::str_to_term("yay")));
}

#[test]
fn fib() {
    &*VM;

    let arc_scheduler = scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    liblumen_otp::erlang::apply_3::export();

    let module = Atom::try_from_str("fib").unwrap();
    let function = Atom::try_from_str("fib").unwrap();

    let eir_mod = compile(
        "
-module(fib).

fib(0) -> 0;
fib(1) -> 1;
fib(X) -> fib(X - 1) + fib(X - 2).
",
    );

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let int = init_arc_process.integer(5).unwrap();
    let res =
        crate::call_result::call_run_erlang(init_arc_process.clone(), module, function, &[int]);

    let int = init_arc_process.integer(5).unwrap();
    assert!(res.result == Ok(int));
}

#[test]
fn exception_test() {
    &*VM;

    let arc_scheduler = scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    liblumen_otp::erlang::apply_3::export();

    let module = Atom::try_from_str("exception_test").unwrap();
    let function = Atom::try_from_str("a").unwrap();

    let eir_mod = compile(
        "
-module(exception_test).

a() -> 1 + a.
",
    );

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let res = crate::call_result::call_run_erlang(init_arc_process.clone(), module, function, &[]);

    assert!(res.result.is_err());
    if let Err((typ, reason, _trace)) = res.result {
        assert!(typ == Atom::str_to_term("error"));
        assert!(reason == Atom::str_to_term("badarith"));
    }
}

#[test]
fn fib_gc() {
    &*VM;

    let arc_scheduler = scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    liblumen_otp::erlang::apply_3::export();

    let module = Atom::try_from_str("fib2").unwrap();
    let function = Atom::try_from_str("fib").unwrap();

    let eir_mod = compile(
        "
-module(fib2).

fib(0) -> 0;
fib(1) -> 1;
fib(X) -> fib(X - 1) + fib(X - 2).
",
    );

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let int = init_arc_process.integer(14).unwrap();
    let res =
        crate::call_result::call_run_erlang(init_arc_process.clone(), module, function, &[int]);

    let int = init_arc_process.integer(377).unwrap();
    assert!(res.result == Ok(int));
}

#[test]
fn ping_pong() {
    &*VM;

    let arc_scheduler = scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    liblumen_otp::erlang::apply_3::export();

    let module = Atom::try_from_str("ping_pong").unwrap();
    let function = Atom::try_from_str("run").unwrap();

    let eir_mod = compile(
        "
-module(ping_pong).

proc_a(A) ->
    receive
        {b, R} -> R ! c
    end.

proc_b(A, B) ->
    receive
        a ->
            B ! {b, self()},
            proc_b(A, B);
        c ->
            A ! d
    end.

run() ->
    P1 = spawn(ping_pong, proc_a, [self()]),
    P2 = spawn(ping_pong, proc_b, [self(), P1]),
    P2 ! a,
    receive
        Res -> Res
    end.
",
    );

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let res = crate::call_result::call_run_erlang(init_arc_process.clone(), module, function, &[]);

    assert!(res.result == Ok(Atom::str_to_term("d")));
}

#[test]
#[ignore]
fn ping_pong_count() {
    &*VM;

    let arc_scheduler = scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    liblumen_otp::erlang::apply_3::export();

    let module = Atom::try_from_str("ping_pong_count").unwrap();
    let function = Atom::try_from_str("run").unwrap();

    let eir_mod = compile(
        "
-module(ping_pong_count).

other_proc({add, A, B}, Ret) -> Ret ! {result, A + B}.

this_proc(0, Acc) ->
    Acc;
this_proc(N, Acc) ->
    spawn(ping_pong_count, other_proc, [{add, 1, Acc}, self()]),
    receive
        {result, Res} -> this_proc(N - 1, Res)
    end.

run(N) -> this_proc(N, 0).
",
    );

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let int = init_arc_process.integer(10).unwrap();
    let res =
        crate::call_result::call_run_erlang(init_arc_process.clone(), module, function, &[int]);

    println!("{:?}", res.result);
    //assert!(res.result == Ok(100));
}

#[test]
#[ignore]
fn ping_pong_count_large() {
    &*VM;

    let arc_scheduler = scheduler::current();
    let init_arc_process = arc_scheduler.spawn_init(0).unwrap();

    liblumen_otp::erlang::apply_3::export();

    let module = Atom::try_from_str("ping_pong_count_large").unwrap();
    let function = Atom::try_from_str("run").unwrap();

    let eir_mod = compile(
        "
-module(ping_pong_count_large).

other_proc({add, A, B}, Ret) -> Ret ! {result, A + B}.

this_proc(0, Acc) ->
    Acc;
this_proc(N, Acc) ->
    spawn(ping_pong_count_large, other_proc, [{add, 1, Acc}, self()]),
    receive
        {result, Res} -> this_proc(N - 1, Res)
    end.

run(N) -> this_proc(N, 0).
",
    );

    VM.modules.write().unwrap().register_erlang_module(eir_mod);

    let int = init_arc_process.integer(100).unwrap();
    let res =
        crate::call_result::call_run_erlang(init_arc_process.clone(), module, function, &[int]);

    println!("{:?}", res.result);
    //assert!(res.result == Ok(100));
}
