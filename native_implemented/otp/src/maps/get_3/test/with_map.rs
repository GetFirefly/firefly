use super::*;

#[test]
fn without_key_returns_default() {
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
                        let value = Atom::str_to_term("value").into();

                        (non_key, arc_process.map_from_slice(&[(key, value)]))
                    }),
                |(key, map)| {
                    let default = Atom::str_to_term("default").into();
                    prop_assert_eq!(result(&arc_process, key, map, default), Ok(default.into()));

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
                    let value = Atom::str_to_term("value").into();

                    (key, arc_process.map_from_slice(&[(key, value)]))
                }),
                |(key, map)| {
                    let default = Atom::str_to_term("default").into();
                    let value = Atom::str_to_term("value").into();
                    prop_assert_eq!(result(&arc_process, key, map, default), Ok(value.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
