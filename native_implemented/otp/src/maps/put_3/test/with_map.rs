use super::*;

use proptest::strategy::Just;

#[test]
fn without_key_puts_new_value() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, key, value)| {
            let empty_map = arc_process.map_from_slice(&[]).unwrap();
            let updated_map = arc_process.map_from_slice(&[(key, value)]).unwrap();
            prop_assert_eq!(
                result(&arc_process, key, value, empty_map),
                Ok(updated_map.into())
            );

            Ok(())
        },
    );
}

#[test]
fn with_key_puts_replacement_value() {
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
