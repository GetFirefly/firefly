mod with_empty_list_options;
mod with_proper_list_options;

use super::*;

use crate::erlang::binary_to_float_1;

#[test]
fn without_proper_list_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, float, tail)| {
            let options = arc_process.improper_list_from_slice(&[atoms::Compact.into()], tail);

            prop_assert_badarg!(result(&arc_process, float, options), "improper list");

            Ok(())
        },
    );
}
