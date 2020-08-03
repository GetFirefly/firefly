use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::map_get_2::result;
use crate::test::strategy;

#[test]
fn without_map_errors_badmap() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_map(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, map, key)| {
            prop_assert_badmap!(
                result(&arc_process, key, map),
                &arc_process,
                map,
                format!("map ({}) is not a map", map)
            );

            Ok(())
        },
    );
}

#[test]
fn with_map_without_key_errors_badkey() {
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
                .prop_filter(
                    "Key being tested must not be a key in map",
                    |(_, key, non_key)| key != non_key,
                )
                .prop_flat_map(|(arc_process, key, non_key)| {
                    (
                        Just(arc_process.clone()),
                        Just(key),
                        Just(non_key),
                        strategy::term(arc_process),
                    )
                })
                .prop_map(|(arc_process, key, non_key, value)| {
                    (
                        arc_process.clone(),
                        arc_process.map_from_slice(&[(key, value)]).unwrap(),
                        non_key,
                    )
                }),
            |(arc_process, map, key)| {
                prop_assert_badkey!(
                    result(&arc_process, key, map),
                    &arc_process,
                    key,
                    format!("key ({}) does not exist in map ({})", key, map)
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_map_with_key_returns_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
                .prop_map(|(arc_process, key, value)| {
                    (
                        arc_process.clone(),
                        arc_process.map_from_slice(&[(key, value)]).unwrap(),
                        key,
                        value,
                    )
                })
        },
        |(arc_process, map, key, value)| {
            prop_assert_eq!(result(&arc_process, key, map), Ok(value));

            Ok(())
        },
    );
}
