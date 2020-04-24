mod with_list;

use proptest::strategy::{Just, Strategy};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::list_to_integer_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, list)| {
            prop_assert_badarg!(
                result(&arc_process, list),
                format!("list ({}) is not a list", list)
            );

            Ok(())
        },
    );
}
