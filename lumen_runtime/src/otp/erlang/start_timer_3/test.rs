mod with_small_integer_time;

use proptest::strategy::{BoxedStrategy, Just};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::{atom_unchecked, next_pid};

use crate::otp::erlang;
use crate::otp::erlang::start_timer_3::native;
use crate::scheduler::with_process_arc;
use crate::test::{has_message, registered_name, strategy, timeout_message};
use crate::time::monotonic::Milliseconds;
use crate::{process, timer};

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super shot soon and later wheel sizes used for `cfg(test)`

#[test]
fn without_non_negative_integer_time_error_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_non_negative_integer(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(time, message)| {
                    let destination = arc_process.pid_term();

                    prop_assert_eq!(
                        native(arc_process.clone(), time, destination, message),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
