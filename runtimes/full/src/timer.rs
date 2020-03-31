pub mod cancel;
pub mod read;
pub mod start;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Process;

use lumen_rt_core::time::Milliseconds;
pub use lumen_rt_core::timer::{Destination, Timeout};

use crate::scheduler::{Scheduled, Scheduler};

pub fn cancel(timer_reference: &Reference) -> Option<Milliseconds> {
    timer_reference
        .scheduler()
        .and_then(|scheduler| scheduler.hierarchy.write().cancel(timer_reference.number()))
}

pub fn read(timer_reference: &Reference) -> Option<Milliseconds> {
    timer_reference
        .scheduler()
        .and_then(|scheduler| scheduler.hierarchy.read().read(timer_reference.number()))
}

pub fn start(
    monotonic_time_milliseconds: Milliseconds,
    destination: Destination,
    timeout: Timeout,
    process_message: Term,
    process: &Process,
) -> AllocResult<Term> {
    let scheduler = Scheduler::current();

    let result = scheduler.hierarchy.write().start(
        monotonic_time_milliseconds,
        destination,
        timeout,
        process_message,
        process,
        scheduler.next_reference_number(),
        scheduler.id,
    );

    result
}

/// Times out the timers for the thread that have timed out since the last time `timeout` was
/// called.
pub fn timeout() {
    Scheduler::current().hierarchy.write().timeout();
}
