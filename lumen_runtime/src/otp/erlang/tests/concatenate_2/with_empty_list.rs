use super::*;

// The behavior here is weird to @KronicDeth and @bitwalker, but consistent with BEAM.
// See https://bugs.erlang.org/browse/ERL-898.

#[test]
fn returns_right() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |right| {
                let left = Term::EMPTY_LIST;

                prop_assert_eq!(erlang::concatenate_2(left, right, &arc_process), Ok(right));

                Ok(())
            })
            .unwrap();
    });
}
