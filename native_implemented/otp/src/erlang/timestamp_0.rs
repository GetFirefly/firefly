#[cfg(test)]
mod test;

use num_bigint::{BigInt, ToBigInt};
use num_traits::cast::ToPrimitive;

use firefly_rt::process::Process;
use firefly_rt::*;
use firefly_rt::term::Term;

use crate::runtime::time::{system, Unit::Microsecond};

#[native_implemented::function(erlang:timestamp/0)]
pub fn result(process: &Process) -> Term {
    let big_int = system::time_in_unit(Microsecond);
    let erlang_timestamp = ErlangTimestamp::from_microseconds(&big_int);

    process.tuple_term_from_term_slice(&[
        process.integer(erlang_timestamp.megaseconds as usize).unwrap(),
        process.integer(erlang_timestamp.seconds as usize).unwrap(),
        process.integer(erlang_timestamp.microseconds as usize).unwrap(),
    ])
}

struct ErlangTimestamp {
    pub megaseconds: u32,
    pub seconds: u32,
    pub microseconds: u32,
}

impl ErlangTimestamp {
    pub fn from_microseconds(system_time: &BigInt) -> Self {
        // algorithm taken from http://erlang.org/doc/man/erlang.html#timestamp-0
        let megaseconds: BigInt = system_time / ((1000000000000 as u64).to_bigint().unwrap());
        let seconds: BigInt = system_time / 1000000 - &megaseconds * 1000000;
        let microseconds: BigInt = system_time % 1000000;

        Self {
            megaseconds: megaseconds.to_u32().unwrap(),
            seconds: seconds.to_u32().unwrap(),
            microseconds: microseconds.to_u32().unwrap(),
        }
    }
}
