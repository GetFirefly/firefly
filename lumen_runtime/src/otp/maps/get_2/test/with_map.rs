use super::*;

use liblumen_alloc::badkey;

#[test]
fn without_key_errors_badkey() {
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
                        let value = atom!("value");

                        (
                            non_key,
                            arc_process.map_from_slice(&[(key, value)]).unwrap(),
                        )
                    }),
                |(key, map)| {
                    prop_assert_eq!(
                        native(&arc_process, key, map),
                        Err(badkey!(&arc_process, key))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_key_returns_value() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone()).prop_map(|key| {
                    let value = atom!("value");

                    (key, arc_process.map_from_slice(&[(key, value)]).unwrap())
                }),
                |(key, map)| {
                    let value = atom!("value");
                    prop_assert_eq!(native(&arc_process, key, map), Ok(value.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
