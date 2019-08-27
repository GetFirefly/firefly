use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_function_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_function(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be function", |v| !v.is_closure()),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_function_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_function(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(operand, operand),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_function_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::module_function_arity::module(),
                    strategy::module_function_arity::function(),
                    strategy::module_function_arity::arity(),
                )
                    .prop_map(move |(module, function, arity)| {
                        let code = |arc_process: &Arc<ProcessControlBlock>| {
                            arc_process.wait();

                            Ok(())
                        };
                        let creator = unsafe { arc_process.pid().as_term() };

                        let left_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity,
                        });
                        let left_term = arc_process
                            .closure(creator, left_module_function_arity, code, vec![])
                            .unwrap();

                        let right_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity,
                        });
                        let right_term = arc_process
                            .closure(creator, right_module_function_arity, code, vec![])
                            .unwrap();

                        (left_term, right_term)
                    }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_function_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::module_function_arity::module(),
                    strategy::module_function_arity::function(),
                    strategy::module_function_arity::arity(),
                )
                    .prop_map(move |(module, function, arity)| {
                        let creator = unsafe { arc_process.pid().as_term() };

                        let left_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity,
                        });
                        let left_code = |arc_process: &Arc<ProcessControlBlock>| {
                            arc_process.wait();

                            Ok(())
                        };
                        let left_term = arc_process
                            .closure(creator, left_module_function_arity, left_code, vec![])
                            .unwrap();

                        let right_module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity,
                        });
                        let right_code = |arc_process: &Arc<ProcessControlBlock>| {
                            arc_process.wait();

                            Ok(())
                        };
                        let right_term = arc_process
                            .closure(creator, right_module_function_arity, right_code, vec![])
                            .unwrap();

                        (left_term, right_term)
                    }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
