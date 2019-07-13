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
                        Err(badmap!(&mut arc_process.acquire_heap(), map))
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
                        let value = atom_unchecked("value");

                        (
                            non_key,
                            arc_process.map_from_slice(&[(key, value)]).unwrap(),
                        )
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
                    let value = atom_unchecked("value");

                    (key, arc_process.map_from_slice(&[(key, value)]).unwrap())
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
