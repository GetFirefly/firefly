mod with_arity_zero;

use super::*;

use proptest::prop_assert;
use proptest::strategy::Strategy;

use crate::runtime::scheduler;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
                (1_u8..=255_u8),
            )
                .prop_map(|(arc_process, module, function, arity)| {
                    (
                        arc_process.clone(),
                        strategy::term::export_closure(&arc_process, module, function, arity),
                        arity,
                    )
                })
        },
        |(arc_process, function, arity)| {
            let result = native(&arc_process, function);

            prop_assert!(result.is_ok());

            let child_pid_term = result.unwrap();

            prop_assert!(child_pid_term.is_pid());

            let child_pid: Pid = child_pid_term.try_into().unwrap();
            let child_arc_process = pid_to_process(&child_pid).unwrap();

            prop_assert!(scheduler::run_through(&child_arc_process));

            let args = Term::NIL;
            let source_substring = format!(
                "arguments ([]) length (0) does not match arity ({}) of function ({})",
                arity, function
            );
            prop_assert_exits_badarity(&child_arc_process, function, args, &source_substring)
        },
    );
}
