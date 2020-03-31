mod options;

use std::sync::atomic::{AtomicU64, Ordering};

use lazy_static::lazy_static;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::scheduler;

pub use options::*;

/// There are two types of unique integers both created using the erlang:unique_integer() BIF:
///
/// 1. Unique integers created with the monotonic modifier consist of a set of 2⁶⁴ - 1 unique
///    integers.
/// 2. Unique integers created without the monotonic modifier consist of a set of 2⁶⁴ - 1 unique
///    integers per scheduler thread and a set of 2⁶⁴ - 1 unique integers shared by other threads.
///    That is, the total amount of unique integers without the monotonic modifier is
///    (NoSchedulers + 1) × (2⁶⁴ - 1).
///
/// If a unique integer is created each nanosecond, unique integers will at earliest be reused after
/// more than 584 years. That is, for the foreseeable future they are unique enough.
///
/// - http://erlang.org/doc/efficiency_guide/advanced.html#unique_integers
pub fn unique_integer(process: &Process, options: Options) -> exception::Result<Term> {
    if options.monotonic {
        let u = MONOTONIC.fetch_add(1, Ordering::SeqCst);

        // See https://github.com/erlang/otp/blob/769ff22c750d939fdc9cb45fae1e44817ec04307/erts/emulator/beam/erl_bif_unique.c#L669-L697
        if options.positive {
            process.integer(u)
        } else {
            // When not positive allow for negative and positive even though the counter is unsigned
            // by subtracting counter value down into signed range.
            let i = if u < NEGATED_I64_MIN_U64 {
                (u as i64) + std::i64::MIN
            } else {
                (u - NEGATED_I64_MIN_U64) as i64
            };

            process.integer(i)
        }
    } else {
        // Non-monotonic unique integers are per-scheduler (https://github.com/erlang/otp/blob/769ff22c750d939fdc9cb45fae1e44817ec04307/erts/emulator/beam/erl_bif_unique.c#L572-L584)
        // Instead of being u64, they are u128 with the first u64 is the scheduler ID
        let scheduler_id = process.scheduler_id().unwrap();
        let scheduler_id_u128: u128 = scheduler_id.into();

        let arc_scheduler = scheduler::from_id(&scheduler_id).unwrap();
        let scheduler_unique_integer = arc_scheduler.next_unique_integer() as u128;

        let u: u128 = (scheduler_id_u128 << 64) | scheduler_unique_integer;

        if options.positive {
            process.integer(u)
        } else {
            let i = if u < NEGATED_I128_MIN_U128 {
                (u as i128) + std::i128::MIN
            } else {
                (u - NEGATED_I128_MIN_U128) as i128
            };

            process.integer(i)
        }
    }
    .map_err(|alloc| alloc.into())
}

// have to add and then subtract to prevent overflow
const NEGATED_I64_MIN_U64: u64 = ((-(std::i64::MIN + 1)) - 1) as u64;
// have to add and then subtract to prevent overflow
const NEGATED_I128_MIN_U128: u128 = ((-(std::i128::MIN + 1)) - 1) as u128;

lazy_static! {
    static ref MONOTONIC: AtomicU64 = Default::default();
}
