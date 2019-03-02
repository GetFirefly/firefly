use crate::list::ToList;
use crate::process::Process;
use crate::term::{BadArgument, Term};

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

trait ByteIterator: ExactSizeIterator + DoubleEndedIterator + Iterator<Item = u8>
where
    Self: Sized,
{
    fn part_range(
        &mut self,
        PartRange {
            byte_offset,
            byte_count,
        }: PartRange,
    ) -> &mut Self {
        // skip byte_offset
        for _ in 0..byte_offset {
            self.next();
        }

        for _ in byte_count..self.len() {
            self.next_back();
        }

        self
    }
}

pub trait Part<'a, S, L, T> {
    fn part(&'a self, start: S, length: L, process: &mut Process) -> Result<T, BadArgument>;
}

pub struct PartRange {
    byte_offset: usize,
    byte_count: usize,
}

fn start_length_to_part_range(
    start: usize,
    length: isize,
    available_byte_count: usize,
) -> Result<PartRange, BadArgument> {
    if length >= 0 {
        let non_negative_length = length as usize;

        if (start < available_byte_count) & (start + non_negative_length <= available_byte_count) {
            Ok(PartRange {
                byte_offset: start,
                byte_count: non_negative_length,
            })
        } else {
            Err(BadArgument)
        }
    } else {
        let start_isize = start as isize;

        if (start <= available_byte_count) & (0 <= start_isize + length) {
            let byte_offset = (start_isize + length) as usize;
            let byte_count = (-length) as usize;

            Ok(PartRange {
                byte_offset,
                byte_count,
            })
        } else {
            Err(BadArgument)
        }
    }
}

fn part_range_to_list<T: ByteIterator>(
    mut byte_iterator: T,
    part_range: PartRange,
    mut process: &mut Process,
) -> Term {
    byte_iterator.part_range(part_range).to_list(&mut process)
}

pub trait PartToList<S, L> {
    fn part_to_list(&self, start: S, length: L, process: &mut Process)
        -> Result<Term, BadArgument>;
}
