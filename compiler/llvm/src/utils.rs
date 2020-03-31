mod memory_buffer;
pub mod strings;

pub use self::memory_buffer::{MemoryBuffer, MemoryBufferRef};
pub use self::strings::{LLVMString, RustString};
