mod with_proper_list_options;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Just};
use proptest::{prop_assert, prop_assert_eq};

use firefly_rt::process::Process;
use firefly_rt::term::{Atom, Term};

use crate::erlang;
use crate::erlang::send_3::result;
use crate::test;
use crate::test::{has_heap_message, has_process_message, registered_name, strategy};

#[test]
fn without_list_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, message, options)| {
            prop_assert_badarg!(
                result(&arc_process, arc_process.pid_term().unwrap(), message, options),
                "improper list"
            );

            Ok(())
        },
    );
}
