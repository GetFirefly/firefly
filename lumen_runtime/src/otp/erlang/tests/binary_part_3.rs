use super::*;

mod with_bitstring;

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |binary| {
                    let start = arc_process.integer(0).unwrap();
                    let length = arc_process.integer(0).unwrap();

                    prop_assert_eq!(
                        erlang::binary_part_3(binary, start, length, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
