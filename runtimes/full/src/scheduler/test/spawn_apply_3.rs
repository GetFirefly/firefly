use super::*;

#[test]
fn different_processes_have_different_pids() {
    let erlang = Atom::try_from_str("erlang").unwrap();
    let apply = Atom::try_from_str("apply").unwrap();

    lumen_rt_core::code::export::insert(erlang.clone(), apply, 3, erlang_apply_3_code);

    let exit = Atom::try_from_str("exit").unwrap();
    let normal = Atom::str_to_term("normal");
    let parent_arc_process = test::process::init();

    let first_process_arguments = parent_arc_process.list_from_slice(&[normal]).unwrap();
    let Spawned {
        arc_process: first_arc_process,
        ..
    } = scheduler::spawn_apply_3(
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
    } = scheduler::spawn_apply_3(
        &first_arc_process,
        Default::default(),
        erlang,
        exit,
        second_process_arguments,
    )
    .unwrap();

    assert_ne!(first_arc_process.pid_term(), second_arc_process.pid_term());
}

fn erlang_apply_3_code(_arc_process: &Arc<Process>) -> frames::Result {
    unimplemented!()
}
