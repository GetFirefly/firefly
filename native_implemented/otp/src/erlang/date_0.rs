use crate::runtime::time::datetime;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:date/0)]
pub fn result(process: &Process) -> Term {
    let date: [usize; 3] = datetime::local_date();

    process.tuple_from_slice(&[
        process.integer(date[0]),
        process.integer(date[1]),
        process.integer(date[2]),
    ])
}
