mod with_local_pid;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::atom_unchecked;

use crate::otp::erlang::process_info_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_local_pid_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_local_pid(arc_process.clone()),
                |pid| {
                    let item = atom_unchecked("registered_name");

                    prop_assert_eq!(native(&arc_process, pid, item), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}
