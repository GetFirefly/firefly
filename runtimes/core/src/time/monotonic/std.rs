use std::cell::RefCell;
use std::time::Instant;

use lazy_static::lazy_static;

use super::Monotonic;

pub fn freeze() -> Monotonic {
    FROZEN.with(|frozen| *frozen.borrow_mut().get_or_insert_with(|| elapsed()))
}

pub fn freeze_at(monotonic: Monotonic) {
    FROZEN.with(|frozen| *frozen.borrow_mut() = Some(monotonic));
}

pub fn time() -> Monotonic {
    FROZEN.with(|frozen| frozen.borrow().unwrap_or_else(|| elapsed()))
}

fn elapsed() -> Monotonic {
    Monotonic::from_millis(START.elapsed().as_millis() as u64)
}

// The time frozen at a specific time for testing
thread_local! {
    static FROZEN: RefCell<Option<Monotonic>> = RefCell::new(None);
}

lazy_static! {
    static ref START: Instant = Instant::now();
}
