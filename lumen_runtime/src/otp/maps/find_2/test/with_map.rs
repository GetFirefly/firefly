use super::*;

#[test]
fn without_key_returns_error_atom() {
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
                        let value = Atom::str_to_term("value");

                        (
                            non_key,
                            arc_process.map_from_slice(&[(key, value)]).unwrap(),
                        )
                    }),
                |(key, map)| {
                    let error = Atom::str_to_term("error");

                    prop_assert_eq!(native(&arc_process, key, map), Ok(error.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_key_returns_success_tuple() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone()).prop_map(|key| {
                    let value = Atom::str_to_term("value");

                    (key, arc_process.map_from_slice(&[(key, value)]).unwrap())
                }),
                |(key, map)| {
                    let ok = Atom::str_to_term("ok");
                    let value = Atom::str_to_term("value");
                    let success_tuple = arc_process.tuple_from_slice(&[ok, value]).unwrap();

                    prop_assert_eq!(native(&arc_process, key, map), Ok(success_tuple.into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
