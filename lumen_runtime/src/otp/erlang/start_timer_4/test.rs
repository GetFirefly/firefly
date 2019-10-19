mod with_proper_list_options;

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use proptest::strategy::{BoxedStrategy, Just};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, next_pid, Term};

use crate::otp::erlang;
use crate::otp::erlang::start_timer_4::native;
use crate::scheduler::with_process_arc;
use crate::test::{has_message, registered_name, strategy};
use crate::time::Milliseconds;
use crate::{process, timer};

#[test]
fn without_proper_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::non_negative(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(time, message, options)| {
                    let destination = arc_process.pid_term();

                    prop_assert_eq!(
                        native(arc_process.clone(), time, destination, message, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
