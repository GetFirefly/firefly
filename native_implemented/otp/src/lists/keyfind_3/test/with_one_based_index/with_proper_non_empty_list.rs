use super::*;

#[test]
fn without_found_returns_false() {
    with_process_arc(|arc_process| {
        let key = Atom::str_to_term("not_found");
        let one_based_index = arc_process.integer(1).unwrap();
        let slice = &[arc_process.tuple_from_slice(&[]).unwrap()];
        let tuple_list = arc_process.list_from_slice(slice).unwrap();

        assert_eq!(result(key, one_based_index, tuple_list), Ok(false.into()));
    });
}

#[test]
fn with_non_tuple_in_list_with_found_returns_tuple() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                proptest::collection::vec(strategy::term(arc_process.clone()), 0..=1),
                strategy::term(arc_process.clone()),
                proptest::collection::vec(strategy::term(arc_process.clone()), 0..=1),
                strategy::term::is_not_tuple(arc_process),
            )
                .prop_map(
                    |(arc_process, before_key_vec, key, after_key_vec, non_tuple)| {
                        let index_zero_based_usize = before_key_vec.len() + 1;
                        let index_one_based_term =
                            arc_process.integer(index_zero_based_usize).unwrap();

                        let tuple_with_key = arc_process
                            .tuple_from_slices(&[&before_key_vec, &[key], &after_key_vec])
                            .unwrap();

                        let tuple_list = arc_process
                            .list_from_slice(&[non_tuple, tuple_with_key])
                            .unwrap();

                        (key, index_one_based_term, tuple_list, tuple_with_key)
                    },
                )
        },
        |(key, one_based_index, tuple_list, tuple_with_key)| {
            prop_assert_eq!(result(key, one_based_index, tuple_list), Ok(tuple_with_key));

            Ok(())
        },
    );
}

#[test]
fn with_shorter_tuple_in_list_with_found_returns_tuple() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                proptest::collection::vec(strategy::term(arc_process.clone()), 0..=1),
                strategy::term(arc_process.clone()),
                proptest::collection::vec(strategy::term(arc_process.clone()), 0..=1),
            )
                .prop_flat_map(|(arc_process, before_key_vec, key, after_key_vec)| {
                    // so it does not possess the index being searched
                    let short_tuple_max_len = before_key_vec.len();

                    (
                        Just(arc_process.clone()),
                        Just(before_key_vec),
                        Just(key),
                        Just(after_key_vec),
                        strategy::term::tuple::intermediate(
                            strategy::term(arc_process.clone()),
                            (0..=short_tuple_max_len).into(),
                            arc_process.clone(),
                        ),
                    )
                })
                .prop_map(
                    |(arc_process, before_key_vec, key, after_key_vec, short_tuple)| {
                        let index_zero_based_usize = before_key_vec.len() + 1;
                        let index_one_based_term =
                            arc_process.integer(index_zero_based_usize).unwrap();

                        let tuple_with_key = arc_process
                            .tuple_from_slices(&[&before_key_vec, &[key], &after_key_vec])
                            .unwrap();

                        let tuple_list = arc_process
                            .list_from_slice(&[short_tuple, tuple_with_key])
                            .unwrap();

                        (key, index_one_based_term, tuple_list, tuple_with_key)
                    },
                )
        },
        |(key, one_based_index, tuple_list, tuple_with_key)| {
            prop_assert_eq!(result(key, one_based_index, tuple_list), Ok(tuple_with_key));

            Ok(())
        },
    );
}

#[test]
fn with_found_returns_tuple() {
    with_process_arc(|arc_process| {
        let key = Atom::str_to_term("found");
        let one_based_index = arc_process.integer(1).unwrap();
        let element = arc_process.tuple_from_slice(&[key]).unwrap();
        let slice = &[element];
        let tuple_list = arc_process.list_from_slice(slice).unwrap();

        assert_eq!(result(key, one_based_index, tuple_list), Ok(element));
    });
}
