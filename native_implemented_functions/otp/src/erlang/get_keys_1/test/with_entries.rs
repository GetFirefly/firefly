use super::*;

use std::convert::TryInto;

use proptest::strategy::{Just, Strategy};

use crate::runtime::scheduler::Spawned;

use crate::test;

#[test]
fn without_value_returns_empty_list() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (Just(arc_process.clone()), strategy::term(arc_process))
                })
                .prop_flat_map(|(arc_process, value)| {
                    (
                        Just(arc_process.clone()),
                        Just(value),
                        strategy::term(arc_process.clone()),
                        strategy::term(arc_process).prop_filter(
                            "Value in process dictionary cannot match value passed to get_keys/1",
                            move |process_dictionary_value| process_dictionary_value != &value,
                        ),
                    )
                }),
            |(arc_process, value, entry_key, entry_value)| {
                arc_process.put(entry_key, entry_value).unwrap();

                prop_assert_eq!(native(&arc_process, value), Ok(Term::NIL));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_value_returns_keys_with_value_in_list() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (
                        Just(arc_process.clone()),
                        strategy::term(arc_process.clone()),
                        strategy::term(arc_process),
                    )
                })
                .prop_flat_map(|(arc_process, first_key_with_value, value)| {
                    (
                        Just(arc_process.clone()),
                        Just(first_key_with_value),
                        Just(value),
                        strategy::term(arc_process.clone()).prop_filter(
                            "Second key cannot match first key",
                            move |key_with_other_value| {
                                key_with_other_value != &first_key_with_value
                            },
                        ),
                        strategy::term(arc_process).prop_filter(
                            "Second value cannot match value passed to get_keys/1",
                            move |other_value| other_value != &value,
                        ),
                    )
                })
                .prop_flat_map(
                    |(
                        arc_process,
                        first_key_with_value,
                        value,
                        key_with_other_value,
                        other_value,
                    )| {
                        (
                            Just(arc_process.clone()),
                            Just(first_key_with_value),
                            Just(value),
                            Just(key_with_other_value),
                            Just(other_value),
                            strategy::term(arc_process).prop_filter(
                                "Third key cannot match first or second key",
                                move |second_key_with_value| {
                                    second_key_with_value != &first_key_with_value
                                        && second_key_with_value != &key_with_other_value
                                },
                            ),
                        )
                    },
                ),
            |(
                arc_process,
                first_key_with_value,
                value,
                key_with_other_value,
                other_value,
                second_key_with_value,
            )| {
                arc_process.erase_entries().unwrap();
                arc_process.put(first_key_with_value, value).unwrap();
                arc_process.put(key_with_other_value, other_value).unwrap();
                arc_process.put(second_key_with_value, value).unwrap();

                let list = native(&arc_process, value).unwrap();

                assert!(list.is_list());

                let boxed_cons: Boxed<Cons> = list.try_into().unwrap();

                let vec: Vec<Term> = boxed_cons
                    .into_iter()
                    .map(|result| result.unwrap())
                    .collect();

                assert_eq!(vec.len(), 2);
                assert!(vec.contains(&first_key_with_value));
                assert!(!vec.contains(&key_with_other_value));
                assert!(vec.contains(&second_key_with_value));

                Ok(())
            },
        )
        .unwrap();
}

// From https://github.com/erlang/otp/blob/a62aed81c56c724f7dd7040adecaa28a78e5d37f/erts/doc/src/erlang.xml#L2104-L2112
#[test]
fn doc_test() {
    let init_arc_process = test::process::init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let one = arc_process.integer(1).unwrap();
    let value = arc_process
        .tuple_from_slice(&[one, arc_process.integer(2).unwrap()])
        .unwrap();
    let other_value = arc_process
        .tuple_from_slice(&[one, arc_process.integer(3).unwrap()])
        .unwrap();

    let mary = Atom::str_to_term("mary");
    arc_process.put(mary, value).unwrap();
    let had = Atom::str_to_term("had");
    arc_process.put(had, value).unwrap();
    let a = Atom::str_to_term("a");
    arc_process.put(a, value).unwrap();
    let little = Atom::str_to_term("little");
    arc_process.put(little, value).unwrap();
    let dog = Atom::str_to_term("dog");
    arc_process.put(dog, other_value).unwrap();
    let lamb = Atom::str_to_term("lamb");
    arc_process.put(lamb, value).unwrap();

    let list = native(&arc_process, value).unwrap();

    assert!(list.is_list());

    let boxed_cons: Boxed<Cons> = list.try_into().unwrap();

    let vec: Vec<Term> = boxed_cons
        .into_iter()
        .map(|result| result.unwrap())
        .collect();

    assert_eq!(vec.len(), 5);
    assert!(vec.contains(&mary));
    assert!(vec.contains(&had));
    assert!(vec.contains(&a));
    assert!(vec.contains(&little));
    assert!(vec.contains(&lamb));
}
