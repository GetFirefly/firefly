#![allow(dead_code)]
#![allow(unused_imports)]
/// ## Algorithms
///
/// * 'exrop' - Xoroshiro116+, 58 bits precision and period of 2^116-1 (jump equivalent to 2^64
///   calls)
/// * 'exs1024s' - Xorshift1024*, 64 bits precision and period of 2^1024-1 (jump equivalent to
///   2^512)
/// * 'exsp' - Xorshift116+, 58 bits precision and period of 2^116-1 (jump equivalent to 2^64)
///
/// Default is 'exrop'
///
/// ## Implementation Overview
///
/// Every time a random number is requested, a state is used to calculate it and a new state is
/// produced. The state can either be implicit or be an explicit argument and return value.
///
/// The functions with implicit state use the process dictionary variable rand_seed to remember
/// the current state.
///
/// If a process calls uniform/0, uniform/1 or uniform_real/0 without setting a seed first,
/// seed/1 is called automatically with the default algorithm and creates a non-constant seed.
///
/// The functions with explicit state never use the process dictionary.
use xorshift::*;

pub extern "C" fn uniform() {
    unimplemented!()
}
