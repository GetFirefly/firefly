use super::*;

#[test]
fn without_arity_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process),
            )
                .prop_map(|(arc_process, first_argument, second_argument)| {
                    (
                        arc_process.clone(),
                        arc_process.list_from_slice(&[first_argument, second_argument]),
                    )
                })
        },
        |(arc_process, arguments)| {
            let module = Atom::from_str("module");
            let function = Atom::from_str("function");
            let function = strategy::term::export_closure(&arc_process, module, function, 1);

            let result = result(&arc_process, function, arguments);

            prop_assert_badarity!(
                result,
                &arc_process,
                function,
                arguments,
                format!(
                    "arguments ({}) length (2) does not match arity (1) of function ({})",
                    arguments, function
                )
            );

            Ok(())
        },
    );
}

// `with_arity_returns_function_return` in integration tests
