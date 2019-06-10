use super::*;

use proptest::strategy::Strategy;

use crate::process::ModuleFunctionArity;

#[test]
fn without_function_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::function(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be function", |v| !v.is_function()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_function_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::function(arc_process.clone()), |operand| {
                prop_assert_eq!(erlang::are_exactly_equal_2(operand, operand), true.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_same_value_function_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::function::arity_usize(),
                )
                    .prop_map(move |(module, function, arity_usize)| {
                        let code = |arc_process: &Arc<Process>| arc_process.wait();

                        let left_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity: arity_usize,
                        });
                        let left_term =
                            Term::function(left_module_function_arity, code, &arc_process);

                        let right_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity: arity_usize,
                        });
                        let right_term =
                            Term::function(right_module_function_arity, code, &arc_process);

                        (left_term, right_term)
                    }),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_function_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::function::arity_usize(),
                )
                    .prop_map(move |(module, function, arity_usize)| {
                        let left_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity: arity_usize,
                        });
                        let left_code = |arc_process: &Arc<Process>| arc_process.wait();
                        let left_term =
                            Term::function(left_module_function_arity, left_code, &arc_process);

                        let right_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity: arity_usize,
                        });
                        let right_code = |arc_process: &Arc<Process>| arc_process.wait();
                        let right_term =
                            Term::function(right_module_function_arity, right_code, &arc_process);

                        (left_term, right_term)
                    }),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}
