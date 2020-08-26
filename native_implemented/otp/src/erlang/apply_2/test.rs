mod with_function;

use std::mem;
use std::sync::Arc;

use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::apply_2::result;
use crate::test::*;

#[test]
fn without_function_errors_badarg() {
    run!(
        |arc_process| (
            Just(arc_process.clone()),
            strategy::term::is_not_function(arc_process.clone())
        ),
        |(arc_process, function)| {
            let result = result(&arc_process, function, Term::NIL);

            prop_assert_badarg!(result, format!("function ({}) is not a function", function));

            Ok(())
        },
    );
}
