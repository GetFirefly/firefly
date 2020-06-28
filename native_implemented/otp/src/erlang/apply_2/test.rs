mod with_function;

use std::mem;
use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};

use liblumen_alloc::borrow::clone_to_process::CloneToProcess;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::apply_2::{frame_with_arguments, result};
use crate::runtime;
use crate::runtime::future::Ready;
use crate::test::{self, *};

#[test]
fn without_function_errors_badarg() {
    run!(
        |arc_process| (
            Just(arc_process.clone()),
            strategy::term::is_not_function(arc_process.clone())
        ),
        |(arc_process, function)| {
            let result = result(&arc_process, function, Term::NIL);

            prop_assert_badarg!(result, format!("function ({}) is not a function", function));

            Ok(())
        },
    );
}

fn run_until_ready(function: Term, arguments: Term) -> Ready {
    runtime::future::run_until_ready(
        Default::default(),
        |child_process| {
            let child_function = function.clone_to_process(child_process);
            let child_arguments = arguments.clone_to_process(child_process);

            Ok(vec![frame_with_arguments(child_function, child_arguments)])
        },
        10,
    )
    .unwrap()
}
