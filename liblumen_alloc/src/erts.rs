pub mod exception;
mod fragment;
pub mod message;
mod module_function_arity;
mod node;
pub mod process;
pub mod scheduler;
pub mod term;
pub mod string;
#[cfg(test)]
pub mod testing;

pub use fragment::{HeapFragment, HeapFragmentAdapter};
pub use message::Message;
pub use module_function_arity::ModuleFunctionArity;
pub use node::*;
pub use process::*;

/// Given a number of bytes `bytes`, returns the number of words
/// needed to hold that number of bytes, rounding up if necessary
#[inline]
pub fn to_word_size(bytes: usize) -> usize {
    use core::mem;
    use liblumen_core::alloc::utils::round_up_to_multiple_of;

    round_up_to_multiple_of(bytes, mem::size_of::<usize>()) / mem::size_of::<usize>()
}

#[allow(unused)]
#[inline]
pub fn to_arch64_word_size(bytes: usize) -> usize {
    use liblumen_core::alloc::utils::round_up_to_multiple_of;

    round_up_to_multiple_of(bytes, 8) / 8
}

#[allow(unused)]
#[inline]
pub fn to_arch32_word_size(bytes: usize) -> usize {
    use liblumen_core::alloc::utils::round_up_to_multiple_of;

    round_up_to_multiple_of(bytes, 4) / 4
}
