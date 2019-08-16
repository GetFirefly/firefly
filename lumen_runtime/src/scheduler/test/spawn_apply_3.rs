use super::*;

use liblumen_alloc::erts::process::default_heap;
use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use crate::process;

#[test]
fn different_processes_have_different_pids() {
    let erlang = Atom::try_from_str("erlang").unwrap();
    let exit = Atom::try_from_str("exit").unwrap();
    let normal = atom_unchecked("normal");
    let parent_arc_process_control_block = process::test_init();

    let first_process_arguments = parent_arc_process_control_block
        .list_from_slice(&[normal])
        .unwrap();
    let (first_heap, first_heap_size) = default_heap().unwrap();
    let first_process = Scheduler::spawn_apply_3(
        &parent_arc_process_control_block,
        erlang,
        exit,
        first_process_arguments,
        first_heap,
        first_heap_size,
    )
    .unwrap();

    let second_process_arguments = parent_arc_process_control_block
        .list_from_slice(&[normal])
        .unwrap();
    let (second_heap, second_heap_size) = default_heap().unwrap();
    let second_process = Scheduler::spawn_apply_3(
        &first_process,
        erlang,
        exit,
        second_process_arguments,
        second_heap,
        second_heap_size,
    )
    .unwrap();

    assert_ne!(first_process.pid_term(), second_process.pid_term());
}
