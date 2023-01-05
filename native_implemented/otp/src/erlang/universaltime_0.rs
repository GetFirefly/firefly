#[cfg(test)]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::time::datetime;

#[native_implemented::function(erlang:universaltime/0)]
pub fn result(process: &Process) -> Term {
    let now: [usize; 6] = datetime::utc_now();

    let date_tuple = process.tuple_term_from_term_slice(&[
        process.integer(now[0]).unwrap(),
        process.integer(now[1]).unwrap(),
        process.integer(now[2]).unwrap(),
    ]);
    let time_tuple = process.tuple_term_from_term_slice(&[
        process.integer(now[3]).unwrap(),
        process.integer(now[4]).unwrap(),
        process.integer(now[5]).unwrap(),
    ]);

    process.tuple_term_from_term_slice(&[date_tuple, time_tuple])
}
