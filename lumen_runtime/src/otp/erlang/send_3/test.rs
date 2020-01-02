mod with_proper_list_options;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::send_3::native;
use crate::process;
use crate::scheduler::with_process_arc;
use crate::test::{has_heap_message, has_process_message, registered_name, strategy};

#[test]
fn without_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(message, options)| {
                    prop_assert_badarg!(
                        native(&arc_process, arc_process.pid_term(), message, options),
                        "improper list"
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
