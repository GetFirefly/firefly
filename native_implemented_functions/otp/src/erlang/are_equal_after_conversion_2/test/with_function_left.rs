use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_function_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::is_function(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be function", |v| !v.is_boxed_function()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_function_right_returns_true() {
    run!(
        |arc_process| strategy::term::is_function(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_value_native_right_returns_true() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
                strategy::module_function_arity::arity(),
            )
                .prop_map(|(arc_process, module, function, arity)| {
                    extern "C" fn native() -> Term {
                        Term::NONE
                    };

                    let left_term = arc_process
                        .export_closure(module, function, arity, Some(native as _))
                        .unwrap();
                    let right_term = arc_process
                        .export_closure(module, function, arity, Some(native as _))
                        .unwrap();

                    (left_term, right_term)
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_native_right_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
                strategy::module_function_arity::arity(),
            )
                .prop_map(|(arc_process, module, function, arity)| {
                    extern "C" fn left_native() -> Term {
                        Term::NONE
                    };
                    let left_term = arc_process
                        .export_closure(module, function, arity, Some(left_native as _))
                        .unwrap();

                    extern "C" fn right_native() -> Term {
                        Term::NONE
                    };
                    let right_term = arc_process
                        .export_closure(module, function, arity, Some(right_native as _))
                        .unwrap();

                    (left_term, right_term)
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
