use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::{badkey, badmap};

use crate::otp::erlang::map_get_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_map_errors_badmap() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_map(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(map, key)| {
                    prop_assert_eq!(
                        native(&arc_process, key, map),
                        Err(badmap!(&arc_process, map))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
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
                prop_assert_eq!(
                    native(&arc_process, key, map),
                    Err(badkey!(&arc_process, key))
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_map_with_key_returns_value() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
                    .prop_map(|(key, value)| {
                        (
                            arc_process.map_from_slice(&[(key, value)]).unwrap(),
                            key,
                            value,
                        )
                    }),
                |(map, key, value)| {
                    prop_assert_eq!(native(&arc_process, key, map), Ok(value));

                    Ok(())
                },
            )
            .unwrap();
    });
}
