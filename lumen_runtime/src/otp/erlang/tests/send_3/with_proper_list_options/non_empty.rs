use super::*;

use proptest::strategy::Strategy;

mod with_noconnect;
mod with_noconnect_and_nosuspend;
mod with_nosuspend;

#[test]
fn with_invalid_option_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Option must be invalid",
                        |option| match option.tag() {
                            Atom => match unsafe { option.atom_to_string() }.as_ref().as_ref() {
                                "noconnect" | "nosuspend" => false,
                                _ => true,
                            },
                            _ => true,
                        },
                    ),
                ),
                |(message, option)| {
                    let destination = arc_process.pid;
                    let options = Term::slice_to_list(&[option], &arc_process);

                    prop_assert_eq!(
                        erlang::send_3(destination, message, options, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
