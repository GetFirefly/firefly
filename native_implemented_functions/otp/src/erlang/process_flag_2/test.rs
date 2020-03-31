mod with_atom_flag;

use std::convert::TryInto;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::erlang::process_flag_2::native;
use crate::runtime::scheduler;
use crate::test::{self, has_message, has_no_message, strategy, with_process};

#[test]
fn without_atom_flag_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, flag, value)| {
            prop_assert_is_not_atom!(native(&arc_process, flag, value), flag);

            Ok(())
        },
    );
}
