use super::*;

use proptest::strategy::Strategy;

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
                        erlang::map_get_2(key, map, &arc_process),
                        Err(badmap!(map, &arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_map_without_key_errors_badkey() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
                    .prop_filter(
                        "Key being tested must not be a key in map",
                        |(key, non_key)| key != non_key,
                    )
                    .prop_flat_map(|(key, non_key)| {
                        (
                            Just(key),
                            Just(non_key),
                            strategy::term(arc_process.clone()),
                        )
                    })
                    .prop_map(|(key, non_key, value)| {
                        (Term::slice_to_map(&[(key, value)], &arc_process), non_key)
                    }),
                |(map, key)| {
                    prop_assert_eq!(
                        erlang::map_get_2(key, map, &arc_process),
                        Err(badkey!(key, &arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
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
                            Term::slice_to_map(&[(key, value)], &arc_process),
                            key,
                            value,
                        )
                    }),
                |(map, key, value)| {
                    prop_assert_eq!(erlang::map_get_2(key, map, &arc_process), Ok(value));

                    Ok(())
                },
            )
            .unwrap();
    });
}
