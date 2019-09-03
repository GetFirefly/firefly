use super::*;

use liblumen_alloc::erts::term::atom_unchecked;
use liblumen_alloc::Term;

#[test]
fn returns_empty_list_of_values() {
    with_process_arc(|arc_process| {
        let empty_map = arc_process.map_from_slice(&[]).unwrap();

        assert_eq!(native(&arc_process, empty_map), Ok(Term::NIL));
    });
}

#[test]
fn returns_list_of_values() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term(arc_process.clone())).prop_map(|value| {
                    let key = atom_unchecked("key");

                    (
                        arc_process.list_from_slice(&[value]).unwrap(),
                        arc_process.map_from_slice(&[(key, value)]).unwrap(),
                    )
                }),
                |(values, map)| {
                    prop_assert_eq!(native(&arc_process, map), Ok(values));

                    Ok(())
                },
            )
            .unwrap();
    });
}
