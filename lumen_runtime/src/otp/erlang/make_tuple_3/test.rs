mod with_arity;

use std::convert::TryInto;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{badarg, fixnum};

use crate::otp::erlang::make_tuple_3::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_arity_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_arity(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(arity, default_value)| {
                    let init_list = Term::NIL;

                    prop_assert_eq!(
                        native(&arc_process, arity, default_value, init_list),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
