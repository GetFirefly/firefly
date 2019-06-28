use super::*;

use proptest::strategy::Strategy;

mod with_local_pid;

#[test]
fn without_pid_or_port_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term(arc_process.clone())
                        .prop_filter("Cannot be pid or port", |pid_or_port| {
                            !(pid_or_port.is_pid() || pid_or_port.is_port())
                        }),
                ),
                |(name, pid_or_port)| {
                    prop_assert_eq!(
                        erlang::register_2(name, pid_or_port, arc_process.clone()),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
