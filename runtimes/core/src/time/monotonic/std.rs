use std::cell::RefCell;
use std::time::Instant;

use lazy_static::lazy_static;

use super::Milliseconds;

pub fn freeze_time_in_milliseconds() -> Milliseconds {
    FROZEN.with(|frozen| {
        *frozen
            .borrow_mut()
            .get_or_insert_with(|| elapsed_time_in_milliseconds())
    })
}

pub fn freeze_at_time_in_milliseconds(milliseconds: Milliseconds) {
    FROZEN.with(|frozen| *frozen.borrow_mut() = Some(milliseconds));
}

pub fn time_in_milliseconds() -> Milliseconds {
    FROZEN.with(|frozen| {
        frozen
            .borrow()
            .unwrap_or_else(|| elapsed_time_in_milliseconds())
    })
}

fn elapsed_time_in_milliseconds() -> Milliseconds {
    START.elapsed().as_millis() as Milliseconds
}

// The time frozen at a specific time for testing
thread_local! {
    static FROZEN: RefCell<Option<Milliseconds>> = RefCell::new(None);
}

lazy_static! {
    static ref START: Instant = Instant::now();
}
