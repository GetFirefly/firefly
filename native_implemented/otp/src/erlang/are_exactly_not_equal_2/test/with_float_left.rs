use super::*;

#[test]
fn without_float_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::float(),
                strategy::term(arc_process.clone())
                    .prop_filter("Right must not be a float", |v| !v.is_float()),
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
            .run(&strategy::term::float(), |operand| {
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
            (Just(arc_process.clone()), any::<f64>())
                .prop_map(|(arc_process, f)| (f.into(), f.into()))
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
                (f.into(), (f / 2.0 + 1.0).into())
            })
        },
        |(left, right)| {
            prop_assert_eq!(result(left.into(), right.into()), true.into());

            Ok(())
        },
    );
}
