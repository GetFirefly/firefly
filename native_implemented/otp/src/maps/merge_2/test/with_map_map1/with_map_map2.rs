use super::*;

use std::convert::TryInto;

use proptest::prop_assert;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::{Boxed, Map};

#[test]
fn with_same_key_in_map1_and_map2_uses_value_from_map2() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, key, value1, value2)| {
            let map1 = arc_process.map_from_slice(&[(key, value1)]).unwrap();
            let map2 = arc_process.map_from_slice(&[(key, value2)]).unwrap();

            let result_map3 = result(&arc_process, map1, map2);

            prop_assert!(result_map3.is_ok());

            let map3 = result_map3.unwrap();

            let map3_map: Boxed<Map> = map3.try_into().unwrap();

            prop_assert_eq!(map3_map.get(key), Some(value2));

            Ok(())
        },
    );
}

#[test]
fn with_different_keys_in_map2_and_map2_combines_keys() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (Just(arc_process.clone()), strategy::term(arc_process))
                })
                .prop_flat_map(|(arc_process, key1)| {
                    (
                        Just(arc_process.clone()),
                        Just(key1),
                        strategy::term(arc_process)
                            .prop_filter("Key1 and Key2 must be different", move |key2| {
                                *key2 != key1
                            }),
                    )
                })
                .prop_flat_map(|(arc_process, key1, key2)| {
                    (
                        Just(arc_process.clone()),
                        Just(key1),
                        strategy::term(arc_process.clone()),
                        Just(key2),
                        strategy::term(arc_process),
                    )
                }),
            |(arc_process, key1, value1, key2, value2)| {
                let map1 = arc_process.map_from_slice(&[(key1, value1)]).unwrap();
                let map2 = arc_process.map_from_slice(&[(key2, value2)]).unwrap();

                let result_map3 = result(&arc_process, map1, map2);

                prop_assert!(result_map3.is_ok());

                let map3 = result_map3.unwrap();

                let map3_map: Boxed<Map> = map3.try_into().unwrap();

                prop_assert_eq!(map3_map.get(key1), Some(value1));
                prop_assert_eq!(map3_map.get(key2), Some(value2));

                Ok(())
            },
        )
        .unwrap();
}
