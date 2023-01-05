mod with_function;

use proptest::strategy::Just;

use firefly_rt::term::Term;

use crate::erlang::spawn_opt_2::result;
use crate::test::*;

#[test]
fn without_function_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_closure(arc_process.clone()),
            )
        },
        |(arc_process, function)| {
            let options = Term::Nil;

            prop_assert_badarg!(
                result(&arc_process, function, options),
                format!("function ({}) is not a function", function)
            );

            Ok(())
        },
    );
}
