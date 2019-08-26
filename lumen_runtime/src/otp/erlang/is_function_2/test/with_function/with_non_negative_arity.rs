use super::*;

use proptest::strategy::Just;

#[test]
fn without_function_arity_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::function::arity_usize().prop_flat_map(|arity_usize| {
                    (
                        strategy::term::is_function_with_arity(
                            arc_process.clone(),
                            arity_usize as u8,
                        ),
                        (Just(arc_process.clone()), 0..=255_usize, Just(arity_usize))
                            .prop_filter(
                                "Guard arity must be different than function arity",
                                |(_, guard_arity_usize, arity_usize)| {
                                    guard_arity_usize != arity_usize
                                },
                            )
                            .prop_map(|(arc_process, u, _)| arc_process.integer(u).unwrap()),
                    )
                }),
                |(function, arity)| {
                    prop_assert_eq!(native(function, arity), Ok(false.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_function_arity_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::function::arity_usize().prop_flat_map(|arity_usize| {
                    (
                        strategy::term::is_function_with_arity(
                            arc_process.clone(),
                            arity_usize as u8,
                        ),
                        Just(arc_process.integer(arity_usize).unwrap()),
                    )
                }),
                |(function, arity)| {
                    prop_assert_eq!(native(function, arity), Ok(true.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
