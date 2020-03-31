mod with_arity_zero;

use super::*;

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
                        arity,
                        strategy::term::export_closure(&arc_process, module, function, arity),
                    )
                })
        },
        |(arc_process, arity, function)| {
            let result = native(&arc_process, function, options(&arc_process));

            prop_assert!(result.is_ok());

            let child_pid_term = result.unwrap();

            prop_assert!(child_pid_term.is_pid());

            let child_pid: Pid = child_pid_term.try_into().unwrap();
            let child_arc_process = pid_to_process(&child_pid).unwrap();

            prop_assert!(scheduler::run_through(&child_arc_process));

            prop_assert_exits_badarity(
                &child_arc_process,
                function,
                Term::NIL,
                format!(
                    "arguments ([]) length (0) does not match arity ({}) of function ({})",
                    arity, function
                ),
            )?;

            Ok(())
        },
    );
}

pub fn options(_: &Process) -> Term {
    Term::NIL
}
