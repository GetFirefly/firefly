mod with_arity_2;

use super::*;

use proptest::prop_oneof;

#[test]
fn without_arity_2_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    (1_usize..=3_usize),
                    strategy::term(arc_process.clone()),
                    (Just(arc_process.clone()), prop_oneof![Just(1), Just(3)])
                        .prop_flat_map(|(arc_process, len)| {
                            (
                                Just(arc_process.clone()),
                                strategy::term::tuple::intermediate(
                                    strategy::term(arc_process.clone()),
                                    (len..=len).into(),
                                    arc_process.clone(),
                                ),
                            )
                        })
                        .prop_map(|(arc_process, element)| {
                            arc_process.list_from_slice(&[element]).unwrap()
                        }),
                )
            }),
            |(arc_process, arity_usize, default_value, init_list)| {
                let arity = arc_process.integer(arity_usize).unwrap();

                prop_assert_eq!(
                    native(&arc_process, arity, default_value, init_list),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}
