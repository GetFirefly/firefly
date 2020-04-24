use super::*;

use proptest::strategy::Just;

#[test]
fn without_function_arity_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::function::arity_u8(),
            )
                .prop_flat_map(|(arc_process, arity_u8)| {
                    (
                        strategy::term::is_function_with_arity(arc_process.clone(), arity_u8),
                        (Just(arc_process.clone()), 0..=255_u8, Just(arity_u8))
                            .prop_filter(
                                "Guard arity must be different than function arity",
                                |(_, guard_arity_u8, arity_u8)| guard_arity_u8 != arity_u8,
                            )
                            .prop_map(|(arc_process, u, _)| arc_process.integer(u).unwrap()),
                    )
                })
        },
        |(function, arity)| {
            prop_assert_eq!(result(function, arity), Ok(false.into()));

            Ok(())
        },
    );
}

#[test]
fn with_function_arity_returns_true() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::function::arity_u8(),
            )
                .prop_flat_map(|(arc_process, arity_usize)| {
                    (
                        strategy::term::is_function_with_arity(
                            arc_process.clone(),
                            arity_usize as u8,
                        ),
                        Just(arc_process.integer(arity_usize).unwrap()),
                    )
                })
        },
        |(function, arity)| {
            prop_assert_eq!(result(function, arity), Ok(true.into()));

            Ok(())
        },
    );
}
