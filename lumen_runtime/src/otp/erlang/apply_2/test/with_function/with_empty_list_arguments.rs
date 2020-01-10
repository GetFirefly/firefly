use super::*;

use std::sync::Arc;

use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Atom;

use crate::test::strategy::term::export_closure;

#[test]
fn without_arity_errors_badarity() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                module_function_arity::module(),
                module_function_arity::function(),
                (1_u8..=255_u8),
            )
                .prop_map(|(arc_process, module, function, arity)| {
                    (
                        arc_process.clone(),
                        arity,
                        export_closure(&arc_process.clone(), module, function, arity),
                    )
                })
        },
        |(arc_process, arity, function)| {
            let Ready {
                arc_process: child_arc_process,
                result,
            } = run_until_ready(function, Term::NIL);

            prop_assert_badarity!(
                result,
                &arc_process,
                function,
                Term::NIL,
                format!(
                    "arguments ([]) length (0) does not match arity ({}) of function ({})",
                    arity, function
                )
            );

            mem::drop(child_arc_process);

            Ok(())
        },
    );
}

#[test]
fn with_arity_returns_function_return() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                module_function_arity::module(),
                module_function_arity::function(),
            )
                .prop_map(|(arc_process, module, function)| {
                    let arity = 0;
                    let code: Code = |arc_process: &Arc<Process>| {
                        arc_process.return_from_call(Atom::str_to_term("return_from_fn"))?;

                        Process::call_code(arc_process)
                    };

                    arc_process
                        .export_closure(module, function, arity, Some(code))
                        .unwrap()
                })
        },
        |function| {
            let Ready {
                arc_process: child_arc_process,
                result,
            } = run_until_ready(function, Term::NIL);

            prop_assert_eq!(result, Ok(Atom::str_to_term("return_from_fn")));

            mem::drop(child_arc_process);

            Ok(())
        },
    );
}
