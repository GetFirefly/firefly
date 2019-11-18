use core::alloc::Layout;
use core::mem;

use liblumen_core::sys::sysconf::MIN_ALIGN;

use super::*;
use crate::erts::process::alloc::*;
use crate::erts::testing::DEFAULT_HEAP_SIZE;

mod collector;
mod simple_collector;
mod sweep;

#[inline]
fn default_heap_layout() -> Layout {
    // Allocate enough space for most tests
    Layout::from_size_align(DEFAULT_HEAP_SIZE, MIN_ALIGN)
        .expect("invalid size/alignment for DEFAULT_HEAP_SIZE")
}
