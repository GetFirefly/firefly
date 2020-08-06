//! All modules under the `liblumen_otp` crate should mirror modules shipped with C-BEAM OTP
#![feature(backtrace)]
#![feature(thread_local)]
// for `liblumen_otp/src/erlang/subtract_list_2`.
#![feature(vec_remove_item)]

#[macro_use]
mod macros;

pub mod binary;
pub mod erlang;
pub mod lists;
pub mod lumen;
pub mod maps;
pub mod number;
#[cfg(not(test))]
use lumen_rt_core as runtime;
#[cfg(test)]
use lumen_rt_full as runtime;
pub mod timer;

#[cfg(test)]
mod test;
