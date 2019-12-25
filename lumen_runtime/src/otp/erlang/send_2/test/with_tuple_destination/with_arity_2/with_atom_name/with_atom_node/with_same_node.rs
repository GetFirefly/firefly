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
                        .tuple_from_slice(&[name, erlang::node_0::native()])
                        .unwrap();

                    prop_assert_badarg!(
                        native(&arc_process, destination, message),
                        format!("name ({}) not registered", name)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
