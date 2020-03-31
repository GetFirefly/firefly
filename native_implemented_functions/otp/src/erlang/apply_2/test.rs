mod with_function;

use std::mem;
use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::strategy::Strategy;

use liblumen_alloc::borrow::clone_to_process::CloneToProcess;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::runtime::future::Ready;

use crate::erlang::apply_2::place_frame_with_arguments;
use crate::test::strategy;

#[test]
fn without_function_errors_badarg() {
    run!(
        |arc_process| strategy::term::is_not_function(arc_process.clone()),
        |function| {
            let Ready {
                arc_process: child_arc_proces,
                result,
                ..
            } = run_until_ready(function, Term::NIL);

            prop_assert_badarg!(result, format!("function ({}) is not a function", function));

            mem::drop(child_arc_proces);

            Ok(())
        },
    );
}

fn run_until_ready(function: Term, arguments: Term) -> Ready {
    crate::runtime::future::run_until_ready(
        Default::default(),
        |child_process| {
            let child_function = function.clone_to_process(child_process);
            let child_arguments = arguments.clone_to_process(child_process);

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
    .unwrap()
}
