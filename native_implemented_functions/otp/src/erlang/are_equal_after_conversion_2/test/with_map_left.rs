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
    run!(
        |arc_process| strategy::term::map(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), true.into());

            Ok(())
        },
    );
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
                let mut heap = arc_process.acquire_heap();
                let entry_vec: Vec<(Term, Term)> = hash_map.drain().collect();

                (
                    heap.map_from_slice(&entry_vec).unwrap(),
                    heap.map_from_slice(&entry_vec).unwrap(),
                )
            })
        },
        |(left, right)| {
            prop_assert_eq!(result(left.into(), right.into()), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_map_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::map(arc_process.clone()),
                strategy::term::map(arc_process.clone()),
            )
                .prop_filter("Maps must be different", |(left, right)| left != right)
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
