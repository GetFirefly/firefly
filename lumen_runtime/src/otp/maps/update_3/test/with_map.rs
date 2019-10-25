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
                ),
                |(key, value)| {
                    let empty_map = arc_process.map_from_slice(&[]).unwrap();
                    prop_assert_eq!(
                        native(&arc_process, key, value, empty_map),
                        Err(badkey!(&arc_process, key).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_key_updates_replacement_value() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone()).prop_map(|key| {
                    let value = Atom::str_to_term("value");

                    (key, arc_process.map_from_slice(&[(key, value)]).unwrap())
                }),
                |(key, map)| {
                    let value2 = Atom::str_to_term("value2");
                    let updated_map = arc_process.map_from_slice(&[(key, value2)]).unwrap();
                    prop_assert_eq!(
                        native(&arc_process, key, value2, map),
                        Ok(updated_map.into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
