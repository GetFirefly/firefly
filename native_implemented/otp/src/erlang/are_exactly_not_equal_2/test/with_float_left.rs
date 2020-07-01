use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_float_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::float(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be a float", |v| !v.is_boxed_float()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_float_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |operand| {
                prop_assert_eq!(result(operand, operand), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_same_value_float_right_returns_false() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), any::<f64>()).prop_map(|(arc_process, f)| {
                (arc_process.float(f).unwrap(), arc_process.float(f).unwrap())
            })
        },
        |(left, right)| {
            prop_assert_eq!(result(left.into(), right.into()), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_float_right_returns_true() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), any::<f64>()).prop_map(|(arc_process, f)| {
                (
                    arc_process.float(f).unwrap(),
                    arc_process.float(f / 2.0 + 1.0).unwrap(),
                )
            })
        },
        |(left, right)| {
            prop_assert_eq!(result(left.into(), right.into()), true.into());

            Ok(())
        },
    );
}
