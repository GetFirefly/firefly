use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_function_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_function(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be function", |v| !v.is_function()),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), false.into());

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
            .run(
                &strategy::term::is_function(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(native(operand, operand), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_function_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::module_function_arity::module(),
                    strategy::module_function_arity::function(),
                    strategy::module_function_arity::arity(),
                )
                    .prop_map(move |(module, function, arity)| {
                        let code = |arc_process: &Arc<Process>| {
                            arc_process.wait();

                            Ok(())
                        };

                        let left_term = arc_process
                            .export_closure(module, function, arity, Some(code))
                            .unwrap();
                        let right_term = arc_process
                            .export_closure(module, function, arity, Some(code))
                            .unwrap();

                        (left_term, right_term)
                    }),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), true.into());

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
                    strategy::module_function_arity::module(),
                    strategy::module_function_arity::function(),
                    strategy::module_function_arity::arity(),
                )
                    .prop_map(move |(module, function, arity)| {
                        let left_code = |arc_process: &Arc<Process>| {
                            arc_process.wait();

                            Ok(())
                        };
                        let left_term = arc_process
                            .export_closure(module, function, arity, Some(left_code))
                            .unwrap();

                        let right_code = |arc_process: &Arc<Process>| {
                            arc_process.wait();

                            Ok(())
                        };
                        let right_term = arc_process
                            .export_closure(module, function, arity, Some(right_code))
                            .unwrap();

                        (left_term, right_term)
                    }),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}
