use super::*;

#[test]
fn without_boolean_value_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_boolean(arc_process.clone()),
            )
        },
        |(arc_process, value)| {
            prop_assert_is_not_boolean!(
                result(&arc_process, flag(), value),
                "trap_exit value",
                value
            );

            Ok(())
        },
    );
}

// `with_boolean_returns_original_value_false` in integration tests
// `with_true_value_then_boolean_value_returns_old_value_true` in integration tests
// `with_true_value_with_linked_and_does_not_exit_when_linked_process_exits_normal` in integration
// tests
// `with_true_value_with_linked_receive_exit_message_and_does_not_exit_when_linked_process_exits` in
// integration tests `with_true_value_then_false_value_exits_when_linked_process_exits` in
// integration tests

fn flag() -> Term {
    Atom::str_to_term("trap_exit")
}
