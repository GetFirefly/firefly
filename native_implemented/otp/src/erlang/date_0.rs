use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::time::datetime;

#[native_implemented::function(erlang:date/0)]
pub fn result(process: &Process) -> Term {
    let date: [usize; 3] = datetime::local_date();

    process.tuple_term_from_term_slice(&[
        process.integer(date[0]).unwrap(),
        process.integer(date[1]).unwrap(),
        process.integer(date[2]).unwrap(),
    ])
}
