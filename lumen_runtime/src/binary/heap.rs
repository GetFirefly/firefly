use std::cmp::Ordering;

use liblumen_arena::TypedArena;

use crate::process::{DebugInProcess, OrderInProcess, Process};
use crate::term::Term;

pub struct Binary {
    header: Term,
    bytes: *const u8,
}

impl<'binary, 'bytes: 'binary> Binary {
    pub fn from_slice(
        bytes: &[u8],
        binary_arena: &'binary mut TypedArena<Binary>,
        byte_arena: &'bytes mut TypedArena<u8>,
    ) -> &'static Self {
        let arena_bytes: &[u8] = if bytes.len() != 0 {
            byte_arena.alloc_slice(bytes)
        } else {
            &[]
        };

        let pointer = binary_arena.alloc(Binary::new(arena_bytes)) as *const Binary;

        unsafe { &*pointer }
    }

    fn new(bytes: &[u8]) -> Self {
        Binary {
            header: Term::heap_binary(bytes.len()),
            bytes: bytes.as_ptr(),
        }
    }

    fn iter(&self) -> Iter {
        let byte_count = Term::heap_binary_to_byte_count(&self.header);

        unsafe {
            Iter {
                pointer: self.bytes,
                limit: self.bytes.offset(byte_count as isize),
            }
        }
    }

    pub fn size(&self) -> Term {
        // The `header` field is not the same as `size` because `sie` is tagged as a small integer
        // while `header` is tagged as `HeapBinary` to mark the beginning of a heap binary.
        Term::heap_binary_to_integer(&self.header)
    }
}

impl DebugInProcess for Binary {
    fn format_in_process(&self, _process: &Process) -> String {
        let mut strings: Vec<String> = Vec::new();
        strings.push("Binary::from_slice(&[".to_string());

        let mut iter = self.iter();

        if let Some(first_byte) = iter.next() {
            strings.push(first_byte.to_string());

            for element in iter {
                strings.push(", ".to_string());
                strings.push(element.to_string());
            }
        }

        strings.push("])".to_string());
        strings.join("")
    }
}

pub struct Iter {
    pointer: *const u8,
    limit: *const u8,
}

impl Iterator for Iter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.pointer == self.limit {
            None
        } else {
            let old_pointer = self.pointer;

            unsafe {
                self.pointer = self.pointer.offset(1);
                old_pointer.as_ref().map(|r| *r)
            }
        }
    }
}

impl OrderInProcess for Binary {
    fn cmp_in_process(&self, other: &Binary, process: &Process) -> Ordering {
        match self.header.cmp_in_process(&other.header, process) {
            Ordering::Equal => {
                let mut final_ordering = Ordering::Equal;

                for (self_element, other_element) in self.iter().zip(other.iter()) {
                    match self_element.cmp(&other_element) {
                        Ordering::Equal => continue,
                        ordering => {
                            final_ordering = ordering;
                            break;
                        }
                    }
                }

                final_ordering
            }
            ordering => ordering,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod from_slice {
        use super::*;

        #[test]
        fn without_bytes() {
            let mut byte_arena: TypedArena<u8> = Default::default();
            let mut binary_arena: TypedArena<Binary> = Default::default();

            let binary = Binary::from_slice(&[], &mut binary_arena, &mut byte_arena);

            assert_eq!(binary.header.tagged, Term::heap_binary(0).tagged);
        }

        #[test]
        fn with_bytes() {
            let mut byte_arena: TypedArena<u8> = Default::default();
            let mut binary_arena: TypedArena<Binary> = Default::default();

            let binary = Binary::from_slice(&[0, 1, 2, 3], &mut binary_arena, &mut byte_arena);

            assert_eq!(binary.header.tagged, Term::heap_binary(4).tagged);
            assert_eq!(unsafe { *binary.bytes.offset(0) }, 0);
            assert_eq!(unsafe { *binary.bytes.offset(1) }, 1);
            assert_eq!(unsafe { *binary.bytes.offset(2) }, 2);
            assert_eq!(unsafe { *binary.bytes.offset(3) }, 3);
        }
    }

    mod eq {
        use super::*;

        #[test]
        fn without_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[], &mut process.binary_arena, &mut process.byte_arena);
            let equal = Binary::from_slice(&[], &mut process.binary_arena, &mut process.byte_arena);

            assert_eq_in_process!(binary, binary, process);
            assert_eq_in_process!(binary, equal, process);
        }

        #[test]
        fn without_equal_length() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[0], &mut process.binary_arena, &mut process.byte_arena);
            let unequal =
                Binary::from_slice(&[0, 1], &mut process.binary_arena, &mut process.byte_arena);

            assert_ne_in_process!(binary, unequal, process);
        }

        #[test]
        fn with_equal_length_without_same_byte() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[0], &mut process.binary_arena, &mut process.byte_arena);
            let unequal =
                Binary::from_slice(&[1], &mut process.binary_arena, &mut process.byte_arena);

            assert_eq_in_process!(binary, binary, process);
            assert_ne_in_process!(binary, unequal, process);
        }

        #[test]
        fn with_equal_length_with_same_bytes() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[0], &mut process.binary_arena, &mut process.byte_arena);
            let unequal =
                Binary::from_slice(&[0], &mut process.binary_arena, &mut process.byte_arena);

            assert_eq_in_process!(binary, unequal, process);
        }
    }

    mod iter {
        use super::*;

        #[test]
        fn without_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[], &mut process.binary_arena, &mut process.byte_arena);

            assert_eq!(binary.iter().count(), 0);
            assert_eq!(binary.iter().count(), binary.size().into());
        }

        #[test]
        fn with_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[0], &mut process.binary_arena, &mut process.byte_arena);

            assert_eq!(binary.iter().count(), 1);
            assert_eq!(binary.iter().count(), binary.size().into());
        }
    }

    mod size {
        use super::*;

        #[test]
        fn without_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[], &mut process.binary_arena, &mut process.byte_arena);

            assert_eq_in_process!(binary.size(), &0.into(), process);
        }

        #[test]
        fn with_elements() {
            let mut process: Process = Default::default();
            let binary =
                Binary::from_slice(&[0], &mut process.binary_arena, &mut process.byte_arena);

            assert_eq_in_process!(binary.size(), &1.into(), process);
        }
    }
}
