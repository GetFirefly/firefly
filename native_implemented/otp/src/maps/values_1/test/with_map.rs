use super::*;

#[test]
fn returns_empty_list_of_values() {
    with_process_arc(|arc_process| {
        let empty_map = arc_process.map_from_slice(&[]);

        assert_eq!(result(&arc_process, empty_map), Ok(Term::Nil));
    });
}

#[test]
fn returns_list_of_values() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term(arc_process.clone())).prop_map(|value| {
                    let key = Atom::str_to_term("key").into();

                    (
                        arc_process.list_from_slice(&[value]).unwrap(),
                        arc_process.map_from_slice(&[(key, value)]),
                    )
                }),
                |(values, map)| {
                    prop_assert_eq!(result(&arc_process, map), Ok(values));

                    Ok(())
                },
            )
            .unwrap();
    });
}
