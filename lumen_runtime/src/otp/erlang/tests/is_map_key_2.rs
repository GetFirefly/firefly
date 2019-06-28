use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_map_errors_badmap() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_map(arc_process.clone()),
                ),
                |(key, map)| {
                    prop_assert_eq!(
                        erlang::is_map_key_2(key, map, &arc_process),
                        Err(badmap!(map, &arc_process))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_map_without_key_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                )
                    .prop_filter("Key and non-key must be different", |(key, non_key)| {
                        key != non_key
                    })
                    .prop_map(|(key, non_key)| {
                        let value = Term::str_to_atom("value", DoNotCare).unwrap();

                        (non_key, Term::slice_to_map(&[(key, value)], &arc_process))
                    }),
                |(key, map)| {
                    prop_assert_eq!(
                        erlang::is_map_key_2(key, map, &arc_process),
                        Ok(false.into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_map_with_key_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone()).prop_map(|key| {
                    let value = Term::str_to_atom("value", DoNotCare).unwrap();

                    (key, Term::slice_to_map(&[(key, value)], &arc_process))
                }),
                |(key, map)| {
                    prop_assert_eq!(
                        erlang::is_map_key_2(key, map, &arc_process),
                        Ok(true.into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
