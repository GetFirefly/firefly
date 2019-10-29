use core::convert::{TryFrom, TryInto};
use core::ops::Range;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;
use liblumen_alloc::erts::exception::{self, Exception};

pub(crate) struct PartRange {
    pub byte_offset: usize,
    pub byte_len: usize,
}

impl From<PartRange> for Range<usize> {
    fn from(part_range: PartRange) -> Self {
        part_range.byte_offset..part_range.byte_offset + part_range.byte_len
    }
}

pub(crate) fn start_length_to_part_range(
    start: usize,
    length: isize,
    available_byte_count: usize,
) -> exception::Result<PartRange> {
    if length >= 0 {
        let non_negative_length = length as usize;

        if (start <= available_byte_count) && (start + non_negative_length <= available_byte_count)
        {
            Ok(PartRange {
                byte_offset: start,
                byte_len: non_negative_length,
            })
        } else {
            Err(badarg!().into())
        }
    } else {
        let start_isize = start as isize;

        if (start <= available_byte_count) && (0 <= start_isize + length) {
            let byte_offset = (start_isize + length) as usize;
            let byte_len = (-length) as usize;

            Ok(PartRange {
                byte_offset,
                byte_len,
            })
        } else {
            Err(badarg!().into())
        }
    }
}

pub trait ToTerm {
    fn to_term(&self, options: ToTermOptions, process: &Process) -> exception::Result<Term>;
}

pub struct ToTermOptions {
    pub existing: bool,
    pub used: bool,
}

impl ToTermOptions {
    fn put_option_term(&mut self, option: Term) -> exception::Result<&ToTermOptions> {
        let atom: Atom = option.try_into()?;

        match atom.name() {
            "safe" => {
                self.existing = true;

                Ok(self)
            }
            "used" => {
                self.used = true;

                Ok(self)
            }
            _ => Err(badarg!().into()),
        }
    }
}

impl TryFrom<Term> for ToTermOptions {
    type Error = Exception;

    fn try_from(term: Term) -> Result<ToTermOptions, Self::Error> {
        let mut options: ToTermOptions = Default::default();
        let mut options_term = term;

        loop {
            match options_term.decode().unwrap() {
                TypedTerm::Nil => return Ok(options),
                TypedTerm::List(cons) => {
                    options.put_option_term(cons.head)?;
                    options_term = cons.tail;

                    continue;
                }
                _ => return Err(badarg!().into()),
            };
        }
    }
}

impl Default for ToTermOptions {
    fn default() -> ToTermOptions {
        ToTermOptions {
            existing: false,
            used: false,
        }
    }
}
