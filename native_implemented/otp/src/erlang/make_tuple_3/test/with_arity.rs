mod with_proper_list;

use super::*;

#[test]
fn without_proper_list_init_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (1_usize..255_usize),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term::is_not_list(arc_process),
            )
        },
        |(arc_process, arity_usize, default_value, element, tail)| {
            let arity = arc_process.integer(arity_usize).unwrap();
            let init_list = arc_process
                .cons(
                    arc_process
                        .tuple_from_slice(&[arc_process.integer(1).unwrap(), element])
                        .unwrap(),
                    tail,
                )
                .unwrap();

            prop_assert_badarg!(
                result(&arc_process, arity, default_value, init_list),
                format!("init_list ({}) is improper", init_list)
            );

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_init_list_returns_tuple_with_arity_copies_of_default_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                (0_usize..255_usize),
                strategy::term(arc_process),
            )
        },
        |(arc_process, arity_usize, default_value)| {
            let arity = arc_process.integer(arity_usize).unwrap();
            let init_list = Term::NIL;

            let result = result(&arc_process, arity, default_value, init_list);

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
    );
}
