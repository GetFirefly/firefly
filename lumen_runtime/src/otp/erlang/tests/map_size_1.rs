use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_map_errors_badmap() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_map(arc_process.clone()), |map| {
                prop_assert_eq!(
                    erlang::map_size_1(map, &arc_process),
                    Err(badmap!(&mut arc_process.acquire_heap(), map))
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_map_returns_number_of_entries() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::collection::hash_map(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    strategy::size_range(),
                )
                .prop_map(|mut hash_map| {
                    let entry_vec: Vec<(Term, Term)> = hash_map.drain().collect();

                    (
                        arc_process.map_from_slice(&entry_vec).unwrap(),
                        arc_process.integer(entry_vec.len()),
                    )
                }),
                |(map, size)| {
                    prop_assert_eq!(erlang::map_size_1(map, &arc_process), Ok(size));

                    Ok(())
                },
            )
            .unwrap();
    });
}
