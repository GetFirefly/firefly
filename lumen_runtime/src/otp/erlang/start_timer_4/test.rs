mod with_proper_list_options;

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use proptest::strategy::{BoxedStrategy, Just};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::start_timer_4::native;
use crate::test::{has_message, registered_name, strategy};
use crate::time::Milliseconds;
use crate::{process, timer};

#[test]
fn without_proper_list_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::non_negative(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, time, message, tail)| {
            let destination = arc_process.pid_term();
            let option = arc_process
                .tuple_from_slice(&[atom!("abs"), true.into()])
                .unwrap();
            let options = arc_process
                .improper_list_from_slice(&[option], tail)
                .unwrap();

            prop_assert_badarg!(
                native(arc_process.clone(), time, destination, message, options),
                "improper list"
            );

            Ok(())
        },
    );
}
