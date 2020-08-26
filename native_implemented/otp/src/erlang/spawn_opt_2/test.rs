mod with_function;

use proptest::strategy::Just;

use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::spawn_opt_2::result;
use crate::test::*;

#[test]
fn without_function_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_function(arc_process.clone()),
            )
        },
        |(arc_process, function)| {
            let options = Term::NIL;

            prop_assert_badarg!(
                result(&arc_process, function, options),
                format!("function ({}) is not a function", function)
            );

            Ok(())
        },
    );
}
