use super::*;

use proptest::strategy::Strategy;

mod with_local_pid;

#[test]
fn without_pid_or_port_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::atom(),
                strategy::term(arc_process.clone())
                    .prop_filter("Cannot be pid or port", |pid_or_port| {
                        !(pid_or_port.is_pid() || pid_or_port.is_port())
                    }),
            )
        },
        |(arc_process, name, pid_or_port)| {
            prop_assert_badarg!(
                erlang::register_2::result(arc_process.clone(), name, pid_or_port),
                format!("{} must be a local pid or port", pid_or_port)
            );

            Ok(())
        },
    );
}
