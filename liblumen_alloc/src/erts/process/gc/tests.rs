use core::alloc::Layout;
use core::mem;

use liblumen_core::sys::sysconf::MIN_ALIGN;

use crate::erts::testing::DEFAULT_HEAP_SIZE;
use crate::erts::process::alloc::*;
use super::*;

mod sweep;
mod simple_collector;
mod collector;

#[inline]
fn default_heap_layout() -> Layout {
    // Allocate enough space for most tests
    Layout::from_size_align(DEFAULT_HEAP_SIZE, MIN_ALIGN)
        .expect("invalid size/alignment for DEFAULT_HEAP_SIZE")
}
