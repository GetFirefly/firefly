mod with_arity_zero;

use super::*;

use proptest::prop_assert;
use proptest::strategy::Strategy;

use crate::process;
use crate::scheduler::Scheduler;
use crate::test::prop_assert_exits_badarity;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity_which_exits_linked_parent(
) {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(
                strategy::module_function_arity::module(),
                strategy::module_function_arity::function(),
                (1_u8..=255_u8),
            ),
            |(module, function, arity)| {
                let parent_arc_process = process::test_init();
                let function =
                    strategy::term::export_closure(&parent_arc_process, module, function, arity);

                let result = native(&parent_arc_process, function);

                prop_assert!(result.is_ok());

                let child_pid_term = result.unwrap();

                prop_assert!(child_pid_term.is_pid());

                let child_pid: Pid = child_pid_term.try_into().unwrap();

                let child_arc_process = pid_to_process(&child_pid).unwrap();

                let scheduler = Scheduler::current();

                prop_assert!(scheduler.run_once());
                prop_assert!(scheduler.run_once());

                let source_substring = format!(
                    "arguments ([]) length (0) does not match arity ({}) of function ({})",
                    arity, function
                );
                let args = Term::NIL;
                prop_assert_exits_badarity(&child_arc_process, function, args, &source_substring)?;
                prop_assert_exits_badarity(&parent_arc_process, function, args, &source_substring)
            },
        )
        .unwrap();
}
