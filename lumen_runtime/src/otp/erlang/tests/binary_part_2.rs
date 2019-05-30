use super::*;

use crate::process::IntoProcess;

mod with_bitstring;

#[test]
fn without_bitstring_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_bitstring(arc_process.clone()),
                |binary| {
                    let start_length = Term::slice_to_tuple(
                        &[0.into_process(&arc_process), 0.into_process(&arc_process)],
                        &arc_process,
                    );

                    prop_assert_eq!(
                        erlang::binary_part_2(binary, start_length, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
