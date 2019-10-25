use super::*;

use proptest::strategy::Just;

#[test]
fn without_proper_list_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                )
            }),
            |(arc_process, list)| {
                prop_assert_eq!(native(&arc_process, list), Err(badarg!().into()));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn without_tuple_list_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term::is_not_tuple(arc_process.clone()),
                )
            }),
            |(arc_process, term)| {
                let list = arc_process.list_from_slice(&[term]).unwrap();
                prop_assert_eq!(native(&arc_process, list), Err(badarg!().into()));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_two_element_tuple_list_returns_value() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
            }),
            |(arc_process, key)| {
                let value = Atom::str_to_term("value");
                let tuple = arc_process.tuple_from_slice(&[key, value]).unwrap();
                let list = arc_process.list_from_slice(&[tuple]).unwrap();
                let map = arc_process.map_from_slice(&[(key, value)]).unwrap();
                prop_assert_eq!(native(&arc_process, list), Ok(map));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_duplicate_keys_preserves_last_value() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
            }),
            |(arc_process, key)| {
                let value1 = Atom::str_to_term("value1");
                let value2 = Atom::str_to_term("value2");
                let tuple1 = arc_process.tuple_from_slice(&[key, value1]).unwrap();
                let tuple2 = arc_process.tuple_from_slice(&[key, value2]).unwrap();
                let list = arc_process.list_from_slice(&[tuple1, tuple2]).unwrap();
                let map = arc_process.map_from_slice(&[(key, value2)]).unwrap();
                prop_assert_eq!(native(&arc_process, list), Ok(map));

                Ok(())
            },
        )
        .unwrap();
}
