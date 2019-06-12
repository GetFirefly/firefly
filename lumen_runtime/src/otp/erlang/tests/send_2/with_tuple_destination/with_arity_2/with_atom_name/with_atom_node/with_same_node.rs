use super::*;

mod registered;

#[test]
fn unregistered_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::atom(), strategy::term(arc_process.clone())),
                |(name, message)| {
                    let destination = Term::slice_to_tuple(&[name, erlang::node_0()], &arc_process);

                    prop_assert_eq!(
                        erlang::send_2(destination, message, &arc_process),
                        Err(badarg!())
                    );
                    assert_badarg!(erlang::send_2(destination, message, &arc_process));

                    Ok(())
                },
            )
            .unwrap();
    });
}
