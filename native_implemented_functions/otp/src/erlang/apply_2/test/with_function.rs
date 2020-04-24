mod with_empty_list_arguments;
mod with_non_empty_proper_list_arguments;

use super::*;

#[test]
fn without_list_arguments_errors_badarg() {
    errors_badarg(strategy::term::is_not_list, |_| format!("is not a list"));
}

#[test]
fn with_list_without_proper_arguments_errors_badarg() {
    errors_badarg(strategy::term::list::improper, |arguments| {
        format!("arguments ({}) is not a proper list", arguments)
    });
}

fn errors_badarg<F, S>(arguments_strategy: F, source_substring: fn(Term) -> String)
where
    F: Fn(Arc<Process>) -> S,
    S: Strategy<Value = Term>,
{
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_function(arc_process.clone()),
                arguments_strategy(arc_process.clone()),
            )
        },
        |(arc_process, function, arguments)| {
            let result = result(&arc_process, function, arguments);

            prop_assert_badarg!(result, source_substring(arguments));

            mem::drop(arc_process);

            Ok(())
        },
    );
}
