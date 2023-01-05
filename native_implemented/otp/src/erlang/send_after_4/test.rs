mod with_proper_list_options;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use firefly_rt::process::Process;
use firefly_rt::term::{atoms, Term};

use crate::erlang;
use crate::erlang::send_after_4::result;
use crate::test;
use crate::test::strategy::milliseconds;
use crate::test::{freeze_at_timeout, freeze_timeout, has_message, registered_name, strategy};

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
                        arc_process.cons(
                            arc_process.tuple_term_from_term_slice(&[atoms::Abs.into(), false.into()]),
                            tail,
                        )
                    }),
            )
        },
        |(arc_process, time, message, options)| {
            let destination = arc_process.pid_term().unwrap();

            prop_assert_badarg!(
                result(arc_process.clone(), time, destination, message, options,),
                "improper list"
            );

            Ok(())
        },
    );
}
