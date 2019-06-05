use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_or_port_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Right must be number, atom, reference, function, or port",
                        |right| {
                            right.is_number()
                                || right.is_atom()
                                || right.is_reference()
                                || right.is_function()
                                || right.is_port()
                        },
                    ),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_equal_or_less_than_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_lesser_local_pid_right_returns_false() {
    is_equal_or_less_than(|_, _| Term::local_pid(0, 0).unwrap(), false);
}

#[test]
fn with_same_value_local_pid_right_returns_true() {
    is_equal_or_less_than(|_, _| Term::local_pid(0, 1).unwrap(), true);
}

#[test]
fn with_greater_local_pid_right_returns_true() {
    is_equal_or_less_than(|_, _| Term::local_pid(1, 1).unwrap(), true);
}

#[test]
fn with_external_pid_right_returns_true() {
    is_equal_or_less_than(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        true,
    );
}

#[test]
fn with_list_or_bitstring_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must be tuple, map, list, or bitstring", |right| {
                            right.is_list() || right.is_bitstring()
                        }),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_equal_or_less_than_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_equal_or_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_equal_or_less_than(|_| Term::local_pid(0, 1).unwrap(), right, expected);
}
