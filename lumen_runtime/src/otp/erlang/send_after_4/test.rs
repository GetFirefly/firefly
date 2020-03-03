mod with_proper_list_options;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::send_after_4::native;
use crate::test::{has_message, registered_name, strategy, timeout_after};
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
                (
                    Just(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                )
                    .prop_map(|(arc_process, tail)| {
                        arc_process
                            .cons(
                                arc_process
                                    .tuple_from_slice(&[atom!("abs"), false.into()])
                                    .unwrap(),
                                tail,
                            )
                            .unwrap()
                    }),
            )
        },
        |(arc_process, time, message, options)| {
            let destination = arc_process.pid_term();

            prop_assert_badarg!(
                native(arc_process.clone(), time, destination, message, options,),
                "improper list"
            );

            Ok(())
        },
    );
}
