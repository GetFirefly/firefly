use super::*;

mod registered;

#[test]
fn unregistered_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::atom(), strategy::term(arc_process.clone())),
                |(name, message)| {
                    let destination = arc_process
                        .tuple_from_slice(&[name, erlang::node_0()])
                        .unwrap();

                    prop_assert_eq!(
                        erlang::send_2(destination, message, &arc_process),
                        Err(badarg!().into())
                    );
                    assert_badarg!(erlang::send_2(destination, message, &arc_process));

                    Ok(())
                },
            )
            .unwrap();
    });
}
