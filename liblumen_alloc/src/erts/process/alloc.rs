pub mod heap_alloc;
mod process_heap_alloc;
mod stack_alloc;
mod stack_primitives;
mod virtual_alloc;

pub use self::heap_alloc::HeapAlloc;
pub use self::process_heap_alloc::*;
pub use self::stack_alloc::StackAlloc;
pub use self::stack_primitives::StackPrimitives;
pub use self::virtual_alloc::VirtualAlloc;
