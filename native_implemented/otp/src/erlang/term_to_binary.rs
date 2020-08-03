mod options;

use std::collections::VecDeque;
use std::convert::TryInto;
use std::mem;
use std::sync::Arc;

use num_bigint::{BigInt, Sign};

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::{Creator, Definition};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Node;

use crate::runtime::distribution::nodes::node::{self, arc_node};

use crate::runtime::distribution::external_term_format::{version, Tag};

use options::*;

pub fn term_to_binary(process: &Process, term: Term, options: Options) -> exception::Result<Term> {
    let byte_vec = term_to_byte_vec(process, &options, term);

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

const NEWER_REFERENCE_EXT_MAX_U32_LEN: usize = 3;

const SMALL_INTEGER_EXT_MIN: isize = std::u8::MIN as isize;
const SMALL_INTEGER_EXT_MAX: isize = std::u8::MAX as isize;

const INTEGER_EXT_MIN: isize = std::i32::MIN as isize;
const INTEGER_EXT_MAX: isize = std::i32::MAX as isize;

const SMALL_TUPLE_EXT_MAX_LEN: usize = std::u8::MAX as usize;
const STRING_EXT_MAX_LEN: usize = std::u16::MAX as usize;
const SMALL_BIG_EXT_MAX_LEN: usize = std::u8::MAX as usize;
const SMALL_ATOM_UTF8_EXT_MAX_LEN: usize = std::u8::MAX as usize;

fn append_big_int(byte_vec: &mut Vec<u8>, big_int: &BigInt) {
    let (sign, mut little_endian_bytes) = big_int.to_bytes_le();

    let sign_byte: u8 = match sign {
        Sign::Minus => 1,
        _ => 0,
    };

    let len_usize = little_endian_bytes.len();

    if len_usize <= SMALL_BIG_EXT_MAX_LEN {
        push_tag(byte_vec, Tag::SmallBig);
        byte_vec.push(len_usize as u8);
    } else {
        push_tag(byte_vec, Tag::LargeBig);
        append_usize_as_u32(byte_vec, len_usize);
    }

    byte_vec.push(sign_byte);
    byte_vec.append(&mut little_endian_bytes);
}

fn append_binary_bytes(byte_vec: &mut Vec<u8>, binary_bytes: &[u8]) {
    byte_vec.extend_from_slice(binary_bytes)
}

fn append_creator(byte_vec: &mut Vec<u8>, creator: &Creator) {
    match creator {
        Creator::Local(pid) => append_pid(
            byte_vec,
            node::arc_node(),
            pid.number() as u32,
            pid.serial() as u32,
        ),
        Creator::External(external_pid) => append_pid(
            byte_vec,
            external_pid.arc_node(),
            external_pid.number() as u32,
            external_pid.serial() as u32,
        ),
    }
}

fn append_pid(byte_vec: &mut Vec<u8>, arc_node: Arc<Node>, id: u32, serial: u32) {
    let creation = arc_node.creation();

    let tag = if creation <= (std::u8::MAX as u32) {
        Tag::PID
    } else {
        Tag::NewPID
    };

    push_tag(byte_vec, tag);

    byte_vec.extend_from_slice(&atom_to_byte_vec(arc_node.name()));
    byte_vec.extend_from_slice(&id.to_be_bytes());
    byte_vec.extend_from_slice(&serial.to_be_bytes());

    if creation <= (std::u8::MAX as u32) {
        byte_vec.push(creation as u8);
    } else {
        byte_vec.extend_from_slice(&creation.to_be_bytes());
    };
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
        push_tag(&mut byte_vec, Tag::Atom);
        append_usize_as_u16(&mut byte_vec, len_usize);
    } else if len_usize <= SMALL_ATOM_UTF8_EXT_MAX_LEN {
        push_tag(&mut byte_vec, Tag::SmallAtomUTF8);

        let len_u8 = len_usize as u8;
        byte_vec.push(len_u8);
    } else {
        push_tag(&mut byte_vec, Tag::AtomUTF8);
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

fn push_tag(byte_vec: &mut Vec<u8>, tag: Tag) {
    byte_vec.push(tag.into());
}

fn term_to_byte_vec(process: &Process, options: &Options, term: Term) -> Vec<u8> {
    let mut stack = VecDeque::new();
    stack.push_front(term);

    let mut byte_vec: Vec<u8> = vec![version::NUMBER];

    while let Some(front_term) = stack.pop_front() {
        match front_term.decode().unwrap() {
            TypedTerm::Atom(atom) => {
                byte_vec.extend_from_slice(&atom_to_byte_vec(atom));
            }
            TypedTerm::List(cons) => {
                match try_cons_to_string_ext_byte_vec(&cons) {
                    Ok(mut string_ext_byte_vec) => byte_vec.append(&mut string_ext_byte_vec),
                    Err(_) => {
                        push_tag(&mut byte_vec, Tag::List);

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
                push_tag(&mut byte_vec, Tag::Nil);
            }
            TypedTerm::Pid(pid) => {
                append_pid(
                    &mut byte_vec,
                    arc_node(),
                    pid.number() as u32,
                    pid.serial() as u32,
                );
            }
            TypedTerm::SmallInteger(small_integer) => {
                let small_integer_isize: isize = small_integer.into();

                match try_append_isize_as_small_integer_or_integer(
                    &mut byte_vec,
                    small_integer_isize,
                ) {
                    Ok(()) => (),
                    Err(_) => {
                        let small_integer_i64 = small_integer_isize as i64;
                        // convert to big int, so that the number of bytes is minimum instead of
                        // jumping to 8 to hold i64.
                        let small_integer_big_int: BigInt = small_integer_i64.into();

                        append_big_int(&mut byte_vec, &small_integer_big_int);
                    }
                }
            }
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();

                append_big_int(&mut byte_vec, big_int);
            }
            TypedTerm::Float(float) => {
                let float_f64: f64 = float.into();

                push_tag(&mut byte_vec, Tag::NewFloat);
                byte_vec.extend_from_slice(&float_f64.to_be_bytes());
            }
            TypedTerm::Closure(closure) => {
                match closure.definition() {
                    Definition::Export { function } => {
                        push_tag(&mut byte_vec, Tag::Export);
                        byte_vec.append(&mut atom_to_byte_vec(closure.module()));
                        byte_vec.append(&mut atom_to_byte_vec(*function));
                        try_append_isize_as_small_integer_or_integer(
                            &mut byte_vec,
                            closure.arity() as isize,
                        )
                        .unwrap();
                    }
                    Definition::Anonymous {
                        index,
                        old_unique,
                        unique,
                        //creator,
                    } => {
                        let default_creator = Creator::Local(Pid::default());
                        let mut sized_byte_vec: Vec<u8> = Vec::new();

                        let module_function_arity = closure.module_function_arity();
                        sized_byte_vec.push(module_function_arity.arity);

                        sized_byte_vec.extend_from_slice(unique);
                        sized_byte_vec.extend_from_slice(&index.to_be_bytes());

                        let env_len_u32: u32 = closure.env_len().try_into().unwrap();
                        sized_byte_vec.extend_from_slice(&env_len_u32.to_be_bytes());

                        sized_byte_vec.append(&mut atom_to_byte_vec(module_function_arity.module));

                        // > [index] encoded using SMALL_INTEGER_EXT or INTEGER_EXT.
                        try_append_isize_as_small_integer_or_integer(
                            &mut sized_byte_vec,
                            (*index).try_into().unwrap(),
                        )
                        .unwrap();

                        // > An integer encoded using SMALL_INTEGER_EXT or INTEGER_EXT
                        // But this means OldUniq can't be the same a Uniq with a different
                        // encoding,
                        try_append_isize_as_small_integer_or_integer(
                            &mut sized_byte_vec,
                            (*old_unique).try_into().unwrap(),
                        )
                        .unwrap();

                        append_creator(&mut sized_byte_vec, &default_creator);

                        for term in closure.env_slice() {
                            sized_byte_vec.append(&mut term_to_byte_vec(process, options, *term));
                        }

                        const SIZE_BYTE_LEN: usize = mem::size_of::<u32>();
                        let size = (SIZE_BYTE_LEN + sized_byte_vec.len()) as u32;

                        push_tag(&mut byte_vec, Tag::NewFunction);
                        byte_vec.extend_from_slice(&size.to_be_bytes());
                        byte_vec.append(&mut sized_byte_vec);
                    }
                }
            }
            TypedTerm::ExternalPid(external_pid) => {
                append_pid(
                    &mut byte_vec,
                    external_pid.arc_node(),
                    external_pid.number() as u32,
                    external_pid.serial() as u32,
                );
            }
            TypedTerm::Map(map) => {
                push_tag(&mut byte_vec, Tag::Map);

                let len_usize = map.len();
                append_usize_as_u32(&mut byte_vec, len_usize);

                for (key, value) in map.iter() {
                    stack.push_front(*value);
                    stack.push_front(*key);
                }
            }
            TypedTerm::HeapBinary(heap_bin) => {
                push_tag(&mut byte_vec, Tag::Binary);

                let len_usize = heap_bin.full_byte_len();
                append_usize_as_u32(&mut byte_vec, len_usize);

                byte_vec.extend_from_slice(heap_bin.as_bytes());
            }
            TypedTerm::MatchContext(match_context) => {
                if match_context.is_binary() {
                    if match_context.is_aligned() {
                        append_binary_bytes(&mut byte_vec, unsafe {
                            match_context.as_bytes_unchecked()
                        });
                    } else {
                        unimplemented!()
                    }
                } else {
                    unimplemented!()
                }
            }
            TypedTerm::ProcBin(proc_bin) => {
                push_tag(&mut byte_vec, Tag::Binary);

                let len_usize = proc_bin.full_byte_len();
                append_usize_as_u32(&mut byte_vec, len_usize);

                byte_vec.extend_from_slice(proc_bin.as_bytes());
            }
            TypedTerm::Reference(reference) => {
                let scheduler_id_u32: u32 = reference.scheduler_id().into();
                let number: u64 = reference.number().into();

                push_tag(&mut byte_vec, Tag::NewerReference);

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
                    push_tag(&mut byte_vec, Tag::Binary);

                    let len_usize = subbinary.full_byte_len();
                    append_usize_as_u32(&mut byte_vec, len_usize);

                    if subbinary.is_aligned() {
                        byte_vec.extend_from_slice(unsafe { subbinary.as_bytes_unchecked() });
                    } else {
                        byte_vec.extend(subbinary.full_byte_iter());
                    }
                } else {
                    push_tag(&mut byte_vec, Tag::BitBinary);

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
                    push_tag(&mut byte_vec, Tag::SmallTuple);
                    byte_vec.push(len_usize as u8);
                } else {
                    push_tag(&mut byte_vec, Tag::LargeTuple);
                    append_usize_as_u32(&mut byte_vec, len_usize);
                }

                for element in tuple.iter().rev() {
                    stack.push_front(*element);
                }
            }
            _ => unimplemented!("term_to_binary({:?})", front_term),
        };
    }

    byte_vec
}

fn try_append_isize_as_small_integer_or_integer(
    mut byte_vec: &mut Vec<u8>,
    integer: isize,
) -> Result<(), TypeError> {
    if SMALL_INTEGER_EXT_MIN <= integer && integer <= SMALL_INTEGER_EXT_MAX {
        let integer_u8: u8 = integer as u8;

        push_tag(&mut byte_vec, Tag::SmallInteger);
        byte_vec.extend_from_slice(&integer_u8.to_be_bytes());

        Ok(())
    } else if INTEGER_EXT_MIN <= integer && integer <= INTEGER_EXT_MAX {
        let small_integer_i32: i32 = integer as i32;

        push_tag(&mut byte_vec, Tag::Integer);
        byte_vec.extend_from_slice(&small_integer_i32.to_be_bytes());

        Ok(())
    } else {
        Err(TypeError)
    }
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

    let mut byte_vec = vec![Tag::String.into()];

    let len_usize = character_byte_vec.len();
    append_usize_as_u16(&mut byte_vec, len_usize);

    byte_vec.extend_from_slice(&character_byte_vec);

    Ok(byte_vec)
}
