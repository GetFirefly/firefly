use super::*;

use crate::scheduler::with_process;

mod with_registered_name;

#[test]
fn without_registered_name_errors_badarg() {
    let name = registered_name();

    with_process(|process| {
        assert_eq!(native(process, name), Err(badarg!(process).into()));
    });
}
