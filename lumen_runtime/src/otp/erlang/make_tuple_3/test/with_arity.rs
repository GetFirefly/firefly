mod with_proper_list;

use super::*;

#[test]
fn without_proper_list_init_list_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    (0_usize..255_usize),
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process),
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

#[test]
fn with_empty_list_init_list_returns_tuple_with_arity_copies_of_default_value() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    (0_usize..255_usize),
                    strategy::term(arc_process),
                )
            }),
            |(arc_process, arity_usize, default_value)| {
                let arity = arc_process.integer(arity_usize).unwrap();
                let init_list = Term::NIL;

                let result = native(&arc_process, arity, default_value, init_list);

                prop_assert!(result.is_ok());

                let tuple_term = result.unwrap();

                prop_assert!(tuple_term.is_boxed());

                let boxed_tuple: Result<Boxed<Tuple>, _> = tuple_term.try_into();
                prop_assert!(boxed_tuple.is_ok());

                let tuple = boxed_tuple.unwrap();
                prop_assert_eq!(tuple.len(), arity_usize);

                for element in tuple.iter() {
                    prop_assert_eq!(element, &default_value);
                }

                Ok(())
            },
        )
        .unwrap();
}
