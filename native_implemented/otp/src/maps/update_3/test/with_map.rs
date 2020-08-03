use super::*;

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

                    prop_assert_badkey!(
                        result(&arc_process, key, value, empty_map),
                        &arc_process,
                        key,
                        format!("key ({}) does not exist in map ({})", key, empty_map)
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
                    let value = atom!("value");

                    (key, arc_process.map_from_slice(&[(key, value)]).unwrap())
                }),
                |(key, map)| {
                    let value2 = atom!("value2");
                    let updated_map = arc_process.map_from_slice(&[(key, value2)]).unwrap();
                    prop_assert_eq!(
                        result(&arc_process, key, value2, map),
                        Ok(updated_map.into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
