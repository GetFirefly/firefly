use super::*;

use proptest::strategy::Just;

#[test]
fn without_proper_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, key, value, tail)| {
            let entry = arc_process.tuple_from_slice(&[key, value]).unwrap();
            let list = arc_process
                .improper_list_from_slice(&[entry], tail)
                .unwrap();

            prop_assert_badarg!(
                result(&arc_process, list),
                format!("list ({}) is improper", list)
            );

            Ok(())
        },
    );
}

#[test]
fn without_tuple_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_tuple(arc_process.clone()),
            )
        },
        |(arc_process, element)| {
            let list = arc_process.list_from_slice(&[element]).unwrap();

            prop_assert_badarg!(
                result(&arc_process, list),
                format!("element ({}) of list ({}) is not a tuple", element, list)
            );

            Ok(())
        },
    );
}

#[test]
fn with_two_element_tuple_list_returns_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, key)| {
            let value = atom!("value");
            let tuple = arc_process.tuple_from_slice(&[key, value]).unwrap();
            let list = arc_process.list_from_slice(&[tuple]).unwrap();
            let map = arc_process.map_from_slice(&[(key, value)]).unwrap();
            prop_assert_eq!(result(&arc_process, list), Ok(map));

            Ok(())
        },
    );
}

#[test]
fn with_duplicate_keys_preserves_last_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, key)| {
            let value1 = atom!("value1");
            let value2 = atom!("value2");
            let tuple1 = arc_process.tuple_from_slice(&[key, value1]).unwrap();
            let tuple2 = arc_process.tuple_from_slice(&[key, value2]).unwrap();
            let list = arc_process.list_from_slice(&[tuple1, tuple2]).unwrap();
            let map = arc_process.map_from_slice(&[(key, value2)]).unwrap();
            prop_assert_eq!(result(&arc_process, list), Ok(map));

            Ok(())
        },
    );
}
