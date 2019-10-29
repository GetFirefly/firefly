use std::collections::VecDeque;
use std::convert::{TryFrom, TryInto};
use std::mem;

use num_bigint::{BigInt, Sign};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::exception::{self, Exception};

use crate::node;

pub fn term_to_binary(process: &Process, term: Term, _options: Options) -> exception::Result<Term> {
    let mut stack = VecDeque::new();
    stack.push_front(term);

    let mut byte_vec: Vec<u8> = vec![VERSION_NUMBER];

    while let Some(front_term) = stack.pop_front() {
        match front_term.decode()? {
            TypedTerm::Atom(atom) => {
                byte_vec.extend_from_slice(&atom_to_byte_vec(atom));
            }
            TypedTerm::List(cons) => {
                match try_cons_to_string_ext_byte_vec(&cons) {
                    Ok(mut string_ext_byte_vec) => byte_vec.append(&mut string_ext_byte_vec),
                    Err(_) => {
                        byte_vec.push(LIST_EXT);

                        let (element_vec, tail) = cons_to_element_vec_tail(&cons);

                        let len_usize = element_vec.len();
                        append_usize_as_u32(&mut byte_vec, len_usize);

                        stack.push_front(tail);

                        for element in element_vec.into_iter().rev() {
                            stack.push_front(element)
                        }
                    }
                };
            }
            TypedTerm::Nil => {
                byte_vec.push(NIL_EXT);
            }
            TypedTerm::Pid(pid) => {
                byte_vec.push(PID_EXT);
                byte_vec.extend_from_slice(&atom_to_byte_vec(node::atom()));

                let id = pid.number() as u32;
                byte_vec.extend_from_slice(&id.to_be_bytes());

                let serial = pid.serial() as u32;
                byte_vec.extend_from_slice(&serial.to_be_bytes());

                byte_vec.extend_from_slice(&CREATION.to_be_bytes());
            }
            TypedTerm::SmallInteger(small_integer) => {
                let small_integer_isize: isize = small_integer.into();

                if SMALL_INTEGER_EXT_MIN <= small_integer_isize
                    && small_integer_isize <= SMALL_INTEGER_EXT_MAX
                {
                    let small_integer_u8: u8 = small_integer_isize as u8;

                    byte_vec.push(SMALL_INTEGER_EXT);
                    byte_vec.extend_from_slice(&small_integer_u8.to_be_bytes());
                } else if INTEGER_EXT_MIN <= small_integer_isize
                    && small_integer_isize <= INTEGER_EXT_MAX
                {
                    let small_integer_i32: i32 = small_integer_isize as i32;

                    byte_vec.push(INTEGER_EXT);
                    byte_vec.extend_from_slice(&small_integer_i32.to_be_bytes());
                } else {
                    let small_integer_i64 = small_integer_isize as i64;
                    // convert to big int, so that the number of bytes is minimum instead of jumping
                    // to 8 to hold i64.
                    let small_integer_big_int: BigInt = small_integer_i64.into();

                    append_big_int(&mut byte_vec, &small_integer_big_int);
                }
            }
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();

                append_big_int(&mut byte_vec, big_int);
            }
            TypedTerm::Float(float) => {
                let float_f64: f64 = float.into();

                byte_vec.push(NEW_FLOAT_EXT);
                byte_vec.extend_from_slice(&float_f64.to_be_bytes());
            }
            TypedTerm::HeapBinary(heap_bin) => {
                byte_vec.push(BINARY_EXT);

                let len_usize = heap_bin.full_byte_len();
                append_usize_as_u32(&mut byte_vec, len_usize);

                byte_vec.extend_from_slice(heap_bin.as_bytes());
            }
            TypedTerm::Map(map) => {
                byte_vec.push(MAP_EXT);

                let len_usize = map.len();
                append_usize_as_u32(&mut byte_vec, len_usize);

                for (key, value) in map.iter() {
                    stack.push_front(*value);
                    stack.push_front(*key);
                }
            }
            TypedTerm::MatchContext(match_context) => {
                if match_context.is_binary() {
                    if match_context.is_aligned() {
                        append_binary_bytes(&mut byte_vec, unsafe { match_context.as_bytes_unchecked() });
                    } else {
                        unimplemented!()
                    }
                } else {
                    unimplemented!()
                }
            }
            TypedTerm::ProcBin(proc_bin) => {
                byte_vec.push(BINARY_EXT);

                let len_usize = proc_bin.full_byte_len();
                append_usize_as_u32(&mut byte_vec, len_usize);

                byte_vec.extend_from_slice(proc_bin.as_bytes());
            }
            TypedTerm::Reference(reference) => {
                let scheduler_id_u32: u32 = reference.scheduler_id().into();
                let number: u64 = reference.number().into();

                byte_vec.push(NEWER_REFERENCE_EXT);

                let u32_byte_len = mem::size_of::<u32>();
                let len_usize = (mem::size_of::<u32>() + mem::size_of::<u64>()) / u32_byte_len;
                // > Len - A 16-bit big endian unsigned integer not larger than 3.
                assert!(len_usize <= NEWER_REFERENCE_EXT_MAX_U32_LEN);
                append_usize_as_u16(&mut byte_vec, len_usize);

                byte_vec.extend_from_slice(&atom_to_byte_vec(node::atom()));

                let creation_u32 = CREATION as u32;
                byte_vec.extend_from_slice(&creation_u32.to_be_bytes());

                byte_vec.extend_from_slice(&scheduler_id_u32.to_be_bytes());
                byte_vec.extend_from_slice(&number.to_be_bytes());
            }
            TypedTerm::SubBinary(subbinary) => {
                if subbinary.is_binary() {
                    byte_vec.push(BINARY_EXT);

                    let len_usize = subbinary.full_byte_len();
                    append_usize_as_u32(&mut byte_vec, len_usize);

                    if subbinary.is_aligned() {
                        byte_vec.extend_from_slice(unsafe { subbinary.as_bytes_unchecked() });
                    } else {
                        byte_vec.extend(subbinary.full_byte_iter());
                    }
                } else {
                    byte_vec.push(BIT_BINARY_EXT);

                    let len_usize = subbinary.total_byte_len();
                    append_usize_as_u32(&mut byte_vec, len_usize);

                    let bits_u8 = subbinary.partial_byte_bit_len();
                    byte_vec.push(bits_u8);

                    if subbinary.is_aligned() {
                        byte_vec.extend_from_slice(unsafe { subbinary.as_bytes_unchecked() });
                    } else {
                        byte_vec.extend(subbinary.full_byte_iter());
                    }

                    let mut last_byte: u8 = 0;

                    for (index, bit) in subbinary.partial_byte_bit_iter().enumerate() {
                        last_byte |= bit << (7 - index);
                    }

                    byte_vec.push(last_byte);
                }
            }
            TypedTerm::Tuple(tuple) => {
                let len_usize = tuple.len();

                if len_usize <= SMALL_TUPLE_EXT_MAX_LEN {
                    byte_vec.push(SMALL_TUPLE_EXT);
                    byte_vec.push(len_usize as u8);
                } else {
                    byte_vec.push(LARGE_TUPLE_EXT);
                    append_usize_as_u32(&mut byte_vec, len_usize);
                }

                for element in tuple.iter().rev() {
                    stack.push_front(*element);
                }
            }
            _ => unimplemented!("term_to_binary({:?})", front_term),
        };
    }

    process
        .binary_from_bytes(&byte_vec)
        .map_err(|alloc| alloc.into())
}

// Private

// TODO implement creation rotation
// > A 32-bit big endian unsigned integer. All identifiers originating from the same node
// > incarnation must have identical Creation values. This makes it possible to separate identifiers
// > from old (crashed) nodes from a new one. The value zero should be avoided for normal operations
// > as it is used as a wild card for debug purpose (like a pid returned by erlang:list_to_pid/1).
const CREATION: u8 = 0;

const VERSION_NUMBER: u8 = 131;

const NEW_FLOAT_EXT: u8 = 70;

const BIT_BINARY_EXT: u8 = 77;

const NEWER_REFERENCE_EXT: u8 = 90;
const NEWER_REFERENCE_EXT_MAX_U32_LEN: usize = 3;

const SMALL_INTEGER_EXT: u8 = 97;
const SMALL_INTEGER_EXT_MIN: isize = std::u8::MIN as isize;
const SMALL_INTEGER_EXT_MAX: isize = std::u8::MAX as isize;

const INTEGER_EXT: u8 = 98;
const INTEGER_EXT_MIN: isize = std::i32::MIN as isize;
const INTEGER_EXT_MAX: isize = std::i32::MAX as isize;

// http://erlang.org/doc/apps/erts/erl_ext_dist.html#atom_ext--deprecated-
const ATOM_EXT: u8 = 100;

const PID_EXT: u8 = 103;

const SMALL_TUPLE_EXT: u8 = 104;
const SMALL_TUPLE_EXT_MAX_LEN: usize = std::u8::MAX as usize;

const LARGE_TUPLE_EXT: u8 = 105;

const NIL_EXT: u8 = 106;

const STRING_EXT: u8 = 107;
const STRING_EXT_MAX_LEN: usize = std::u16::MAX as usize;

const LIST_EXT: u8 = 108;

const BINARY_EXT: u8 = 109;

const SMALL_BIG_EXT: u8 = 110;
const SMALL_BIG_EXT_MAX_LEN: usize = std::u8::MAX as usize;

const LARGE_BIG_EXT: u8 = 111;

const MAP_EXT: u8 = 116;

const ATOM_UTF8_EXT: u8 = 118;

const SMALL_ATOM_UTF8_EXT: u8 = 119;
const SMALL_ATOM_UTF8_EXT_MAX_LEN: usize = std::u8::MAX as usize;

fn append_big_int(byte_vec: &mut Vec<u8>, big_int: &BigInt) {
    let (sign, mut little_endian_bytes) = big_int.to_bytes_le();

    let sign_byte: u8 = match sign {
        Sign::Minus => 1,
        _ => 0,
    };

    let len_usize = little_endian_bytes.len();

    if len_usize <= SMALL_BIG_EXT_MAX_LEN {
        byte_vec.push(SMALL_BIG_EXT);
        byte_vec.push(len_usize as u8);
    } else {
        byte_vec.push(LARGE_BIG_EXT);
        append_usize_as_u32(byte_vec, len_usize);
    }

    byte_vec.push(sign_byte);
    byte_vec.append(&mut little_endian_bytes);
}

fn append_binary_bytes(byte_vec: &mut Vec<u8>, binary_bytes: &[u8]) {
    byte_vec.extend_from_slice(binary_bytes)
}

fn append_usize_as_u16(byte_vec: &mut Vec<u8>, len_usize: usize) {
    assert!(len_usize <= (std::u16::MAX as usize));
    let len_u16 = len_usize as u16;
    byte_vec.extend_from_slice(&len_u16.to_be_bytes());
}

fn append_usize_as_u32(byte_vec: &mut Vec<u8>, len_usize: usize) {
    assert!(len_usize <= (std::u32::MAX as usize));
    let len_u32 = len_usize as u32;
    byte_vec.extend_from_slice(&len_u32.to_be_bytes());
}

fn atom_to_byte_vec(atom: Atom) -> Vec<u8> {
    let bytes = atom.name().as_bytes();
    let len_usize = bytes.len();
    let mut byte_vec: Vec<u8> = Vec::new();

    if bytes.iter().all(|byte| byte.is_ascii()) {
        byte_vec.push(ATOM_EXT);
        append_usize_as_u16(&mut byte_vec, len_usize);
    } else if len_usize <= SMALL_ATOM_UTF8_EXT_MAX_LEN {
        byte_vec.push(SMALL_ATOM_UTF8_EXT);

        let len_u8 = len_usize as u8;
        byte_vec.push(len_u8);
    } else {
        byte_vec.push(ATOM_UTF8_EXT);
        append_usize_as_u16(&mut byte_vec, len_usize);
    }

    byte_vec.extend_from_slice(bytes);

    byte_vec
}

// Tail is the final tail  of the list; it is NIL_EXT for a proper list, but can be any type if the
// list is improper (for example, [a|b]).
// -- http://erlang.org/doc/apps/erts/erl_ext_dist.html#list_ext
fn cons_to_element_vec_tail(cons: &Cons) -> (Vec<Term>, Term) {
    let mut element_vec: Vec<Term> = Vec::new();
    let mut tail = Term::NIL;

    for result in cons.into_iter() {
        match result {
            Ok(element) => element_vec.push(element),
            Err(ImproperList {
                tail: improper_list_tail,
            }) => tail = improper_list_tail,
        }
    }

    (element_vec, tail)
}

fn try_cons_to_string_ext_byte_vec(cons: &Cons) -> Result<Vec<u8>, TypeError> {
    let mut character_byte_vec: Vec<u8> = Vec::new();

    // STRING_EXT is used (https://github.com/erlang/otp/blob/e6a69b021bc2aee6aca42bd72583a96d06f4ba9d/erts/emulator/beam/external.c#L2893)
    // only after checking `is_external_string` (https://github.com/erlang/otp/blob/e6a69b021bc2aee6aca42bd72583a96d06f4ba9d/erts/emulator/beam/external.c#L2892).
    // `is_external_string` only checks if the element is an integer between 0 and 255.  It does not
    // care about printability. (https://github.com/erlang/otp/blob/e6a69b021bc2aee6aca42bd72583a96d06f4ba9d/erts/emulator/beam/external.c#L3164-L3191)
    for (index, result) in cons.into_iter().enumerate() {
        if index < STRING_EXT_MAX_LEN {
            match result {
                Ok(element) => {
                    let character_byte: u8 = element.try_into().map_err(|_| TypeError)?;
                    character_byte_vec.push(character_byte);
                }
                Err(_) => return Err(TypeError),
            }
        } else {
            return Err(TypeError);
        }
    }

    let mut byte_vec = vec![STRING_EXT];

    let len_usize = character_byte_vec.len();
    append_usize_as_u16(&mut byte_vec, len_usize);

    byte_vec.extend_from_slice(&character_byte_vec);

    Ok(byte_vec)
}

pub struct Compression(u8);

impl Compression {
    const MIN_U8: u8 = 0;
    const MAX_U8: u8 = 9;
}

impl Default for Compression {
    fn default() -> Self {
        // Default level when option compressed is provided.
        Self(6)
    }
}

impl TryFrom<Term> for Compression {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let term_u8: u8 = term.try_into()?;

        if Self::MIN_U8 <= term_u8 && term_u8 <= Self::MAX_U8 {
            Ok(Self(term_u8))
        } else {
            Err(badarg!().into())
        }
    }
}

pub struct MinorVersion(u8);

impl MinorVersion {
    const MIN_U8: u8 = 0;
    const MAX_U8: u8 = 2;
}

impl Default for MinorVersion {
    fn default() -> Self {
        Self(1)
    }
}

impl TryFrom<Term> for MinorVersion {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let term_u8: u8 = term.try_into()?;

        if Self::MIN_U8 <= term_u8 && term_u8 <= Self::MAX_U8 {
            Ok(Self(term_u8))
        } else {
            Err(badarg!().into())
        }
    }
}

pub struct Options {
    compression: Compression,
    minor_version: MinorVersion,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            // No compression is done (it is the same as giving no compressed option)
            compression: Compression(0),
            minor_version: Default::default(),
        }
    }
}

impl Options {
    fn put_option_term(&mut self, option: Term) -> exception::Result<&Self> {
        match option.decode()? {
            TypedTerm::Atom(atom) => match atom.name() {
                "compressed" => {
                    self.compression = Default::default();

                    Ok(self)
                }
                _ => Err(badarg!().into()),
            },
            TypedTerm::Tuple(tuple) => {
                if tuple.len() == 2 {
                    let atom: Atom = tuple[0].try_into()?;

                    match atom.name() {
                        "compressed" => {
                            self.compression = tuple[1].try_into()?;

                            Ok(self)
                        }
                        "minor_version" => {
                            self.minor_version = tuple[1].try_into()?;

                            Ok(self)
                        }
                        _ => Err(badarg!().into()),
                    }
                } else {
                    Err(badarg!().into())
                }
            }
            _ => Err(badarg!().into()),
        }
    }
}

impl TryFrom<Term> for Options {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let mut options: Options = Default::default();
        let mut options_term = term;

        loop {
            match options_term.decode()? {
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
