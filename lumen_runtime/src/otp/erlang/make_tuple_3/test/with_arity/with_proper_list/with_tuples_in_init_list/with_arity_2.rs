mod with_positive_index;

use super::*;

#[test]
fn without_positive_index_errors_badarg_because_indexes_are_one_based() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    (1_usize..3_usize),
                    strategy::term(arc_process.clone()),
                    (
                        Just(arc_process.clone()),
                        strategy::term(arc_process.clone()),
                    )
                        .prop_filter("Index must not be a positive index", |(_, index)| {
                            !index.is_integer() || index <= &fixnum!(0)
                        })
                        .prop_flat_map(|(arc_process, index)| {
                            (
                                Just(arc_process.clone()),
                                Just(index),
                                strategy::term(arc_process.clone()),
                            )
                                .prop_map(
                                    |(arc_process, index, element)| {
                                        (
                                            arc_process.clone(),
                                            arc_process.list_from_slice(&[index, element]).unwrap(),
                                        )
                                    },
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
                    Err(badarg!(&arc_process).into())
                );

                Ok(())
            },
        )
        .unwrap();
}
