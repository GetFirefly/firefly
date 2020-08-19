mod with_arity_zero;

use super::*;

#[test]
fn without_arity_zero_returns_pid_to_parent_and_child_process_exits_badarity() {
    let arc_process = process::default();
    let module = Atom::from_str("module");
    let function = Atom::from_str("function");
    let arity = 1;

    assert_ne!(arity, 0);

    let function = arc_process.export_closure(module, function, arity, None);
    let result = result(&arc_process, function);

    assert!(result.is_ok());

    let child_pid_term = result.unwrap();

    assert!(child_pid_term.is_pid());

    let child_pid: Pid = child_pid_term.try_into().unwrap();
    let child_arc_process = pid_to_process(&child_pid).unwrap();

    assert!(scheduler::run_through(&child_arc_process));

    assert_exits_badarity(&child_arc_process, function, arity, Term::NIL);
}
