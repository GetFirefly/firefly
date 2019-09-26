mod with_tuples_in_init_list;

use super::*;

#[test]
fn without_tuple_in_init_list_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    (0_usize..255_usize),
                    strategy::term(arc_process.clone()),
                    (
                        Just(arc_process.clone()),
                        strategy::term::is_not_tuple(arc_process.clone()),
                    )
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
