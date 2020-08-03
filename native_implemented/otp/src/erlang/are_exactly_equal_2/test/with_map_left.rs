use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_map_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::map(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Right cannot be a map", |right| !right.is_boxed_map()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_map_right_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| strategy::term::map(arc_process.clone())),
            |operand| {
                prop_assert_eq!(result(operand, operand), true.into());

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_same_value_map_right_returns_true() {
    run!(
        |arc_process| {
            let key_or_value = strategy::term(arc_process.clone());

            proptest::collection::hash_map(
                key_or_value.clone(),
                key_or_value,
                strategy::size_range(),
            )
            .prop_map(move |mut hash_map| {
                let entry_vec: Vec<(Term, Term)> = hash_map.drain().collect();

                (
                    arc_process.map_from_slice(&entry_vec).unwrap(),
                    arc_process.map_from_slice(&entry_vec).unwrap(),
                )
            })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_map_right_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (
                        strategy::term::map(arc_process.clone()),
                        strategy::term::map(arc_process.clone()),
                    )
                })
                .prop_filter("Maps must be different", |(left, right)| left != right),
            |(left, right)| {
                prop_assert_eq!(result(left, right), false.into());

                Ok(())
            },
        )
        .unwrap();
}
