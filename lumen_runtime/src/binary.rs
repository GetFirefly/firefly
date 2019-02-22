use crate::process::Process;
use crate::term::BadArgument;

pub mod heap;
pub mod sub;

pub enum Binary<'a> {
    Heap(&'a heap::Binary),
    Sub(&'a sub::Binary),
}

impl<'a> Binary<'a> {
    pub fn from_slice(bytes: &[u8], process: &mut Process) -> Self {
        // TODO use reference counted binaries for bytes.len() > 64
        let heap_binary = heap::Binary::from_slice(
            bytes,
            &mut process.heap_binary_arena,
            &mut process.byte_arena,
        );

        Binary::Heap(heap_binary)
    }
}

pub trait Part<'a, S, L, T> {
    fn part(&'a self, start: S, length: L, process: &mut Process) -> Result<T, BadArgument>;
}
