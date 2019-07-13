pub mod heap_alloc;
mod process_heap_alloc;
mod stack_alloc;
mod stack_primitives;
mod virtual_alloc;

use core::alloc::Layout;
use core::mem;

use crate::erts::term::Term;

pub use self::heap_alloc::HeapAlloc;
pub use self::process_heap_alloc::*;
pub use self::stack_alloc::StackAlloc;
pub use self::stack_primitives::StackPrimitives;
pub use self::virtual_alloc::VirtualAlloc;

#[inline]
pub fn layout_to_words(layout: Layout) -> usize {
    let size = layout.size();
    let mut words = size / mem::size_of::<Term>();
    if size % mem::size_of::<Term>() != 0 {
        words += 1;
    }
    words
}
