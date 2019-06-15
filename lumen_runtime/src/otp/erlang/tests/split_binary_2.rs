use super::*;

mod with_bitstring_binary;

#[test]
fn without_bitstring_binary_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_bitstring(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(binary, position)| {
                    prop_assert_eq!(
                        erlang::split_binary_2(binary, position, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
