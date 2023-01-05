#[cfg(test)]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::time::datetime;

#[native_implemented::function(erlang:time/0)]
pub fn result(process: &Process) -> Term {
    let time: [usize; 3] = datetime::local_time();

    process.tuple_term_from_term_slice(&[
        process.integer(time[0]),
        process.integer(time[1]),
        process.integer(time[2]),
    ])
}
