mod with_arity;

use std::convert::TryInto;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::fixnum;

use crate::otp::erlang::make_tuple_3::native;
use crate::test::{run, strategy};

#[test]
fn without_arity_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_arity(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, arity, default_value)| {
            let init_list = Term::NIL;

            prop_assert_is_not_arity!(native(&arc_process, arity, default_value, init_list), arity);

            Ok(())
        },
    );
}
