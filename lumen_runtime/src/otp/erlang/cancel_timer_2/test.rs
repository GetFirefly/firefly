mod with_reference_timer_reference;

use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::cancel_timer_2::native;
use crate::process::SchedulerDependentAlloc;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::{cancel_timer_message, has_message, receive_message, strategy, timeout_message};
use crate::time::Milliseconds;
use crate::{process, timer};

#[test]
fn without_reference_timer_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_reference(arc_process.clone()),
                |timer_reference| {
                    let options = Term::NIL;

                    prop_assert_eq!(
                        native(&arc_process, timer_reference, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_reference_timer_reference_without_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_reference(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(timer_reference, options)| {
                    prop_assert_eq!(
                        native(&arc_process, timer_reference, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
