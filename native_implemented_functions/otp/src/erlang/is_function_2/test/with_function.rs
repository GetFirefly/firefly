mod with_non_negative_arity;

use super::*;

#[test]
fn without_non_negative_arity_errors_badarg() {
    run!(
        |arc_process| {
            (
                strategy::term::is_function(arc_process.clone()),
                strategy::term::integer::negative(arc_process.clone()),
            )
        },
        |(function, arity)| {
            prop_assert_is_not_arity!(result(function, arity), arity);

            Ok(())
        },
    );
}
