use super::*;

use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use crate::process;
use crate::scheduler::Spawned;

#[test]
fn different_processes_have_different_pids() {
    let erlang = Atom::try_from_str("erlang").unwrap();
    let exit = Atom::try_from_str("exit").unwrap();
    let normal = atom_unchecked("normal");
    let parent_arc_process = process::test_init();

    let first_process_arguments = parent_arc_process.list_from_slice(&[normal]).unwrap();
    let Spawned {
        arc_process: first_arc_process,
        ..
    } = Scheduler::spawn_apply_3(
        &parent_arc_process,
        Default::default(),
        erlang,
        exit,
        first_process_arguments,
    )
    .unwrap();

    let second_process_arguments = parent_arc_process.list_from_slice(&[normal]).unwrap();
    let Spawned {
        arc_process: second_arc_process,
        ..
    } = Scheduler::spawn_apply_3(
        &first_arc_process,
        Default::default(),
        erlang,
        exit,
        second_process_arguments,
    )
    .unwrap();

    assert_ne!(first_arc_process.pid_term(), second_arc_process.pid_term());
}
