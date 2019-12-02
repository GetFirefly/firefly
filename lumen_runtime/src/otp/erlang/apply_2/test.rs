mod with_function;

use std::mem;
use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::borrow::clone_to_process::CloneToProcess;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::code::stack::Trace;
use liblumen_alloc::erts::term::prelude::Term;
use liblumen_alloc::{atom_from_str, badarg, ModuleFunctionArity};

use crate::future::{run_until_ready, Ready};
use crate::otp::erlang::apply_2::place_frame_with_arguments;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_function_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_function(arc_process.clone()),
                |function| {
                    let Ready {
                        arc_process: child_arc_proces,
                        result,
                        ..
                    } = run_until_ready(
                        Default::default(),
                        |child_process| {
                            let child_function = function.clone_to_process(child_process);
                            let child_arguments = Term::NIL;

                            place_frame_with_arguments(
                                child_process,
                                Placement::Push,
                                child_function,
                                child_arguments,
                            )
                            .map_err(|e| e.into())
                        },
                        5_000,
                    )
                    .unwrap();

                    prop_assert_eq!(result, Err(badarg!(trace()).into()));

                    mem::drop(child_arc_proces);

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn trace() -> Trace {
    Trace(vec![
        Arc::new(ModuleFunctionArity {
            module: atom_from_str!("erlang"),
            function: atom_from_str!("apply"),
            arity: 2,
        }),
        Arc::new(ModuleFunctionArity {
            module: atom_from_str!("Elixir.Lumen"),
            function: atom_from_str!("future"),
            arity: 0,
        }),
    ])
}
