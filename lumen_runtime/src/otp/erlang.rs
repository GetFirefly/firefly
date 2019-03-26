//! Mirrors [erlang](http://erlang::org/doc/man/erlang::html) module

#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering;
use std::num::FpCategory;

use num_bigint::BigInt;
use num_traits::Zero;

use crate::atom::{Existence, Existence::*};
use crate::binary::{heap, sub, Part, ToTerm, ToTermOptions};
use crate::exception::Result;
use crate::float::Float;
use crate::integer::{big, small};
use crate::list::Cons;
use crate::map::Map;
use crate::otp;
use crate::process::{IntoProcess, Process, TryIntoInProcess};
use crate::term::{Tag, Tag::*, Term};
use crate::time;
use crate::tuple::{Tuple, ZeroBasedIndex};

#[cfg(test)]
mod tests;

pub fn abs_1(number: Term, mut process: &mut Process) -> Result {
    match number.tag() {
        SmallInteger => {
            if unsafe { number.small_integer_is_negative() } {
                // cast first so that sign bit is extended on shift
                let signed = (number.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT;
                let positive = -signed;
                Ok(Term {
                    tagged: ((positive << Tag::SMALL_INTEGER_BIT_COUNT) as usize)
                        | (SmallInteger as usize),
                })
            } else {
                Ok(Term {
                    tagged: number.tagged,
                })
            }
        }
        Boxed => {
            let unboxed: &Term = number.unbox_reference();

            match unboxed.tag() {
                BigInteger => {
                    let big_integer: &big::Integer = number.unbox_reference();
                    let big_int = &big_integer.inner;
                    let zero_big_int: &BigInt = &Zero::zero();

                    let positive_term: Term = if big_int < zero_big_int {
                        let positive_big_int: BigInt = -1 * big_int;

                        positive_big_int.into_process(&mut process)
                    } else {
                        number
                    };

                    Ok(positive_term)
                }
                Float => {
                    let float: &Float = number.unbox_reference();
                    let inner = float.inner;

                    match inner.partial_cmp(&0.0).unwrap() {
                        Ordering::Less => {
                            let positive_inner = inner.abs();
                            let positive_number: Term = positive_inner.into_process(&mut process);

                            Ok(positive_number)
                        }
                        _ => Ok(number),
                    }
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn append_element_2(tuple: Term, element: Term, mut process: &mut Process) -> Result {
    let internal: &Tuple = tuple.try_into_in_process(&mut process)?;
    let new_tuple = internal.append_element(element, &mut process.term_arena);

    Ok(new_tuple.into())
}

pub fn atom_to_binary_2(atom: Term, encoding: Term, mut process: &mut Process) -> Result {
    if atom.tag() == Atom {
        encoding.atom_to_encoding(&mut process)?;
        let string = atom.atom_to_string(process);
        Ok(Term::slice_to_binary(string.as_bytes(), &mut process))
    } else {
        Err(bad_argument!(&mut process))
    }
}

pub fn atom_to_list_1(atom: Term, mut process: &mut Process) -> Result {
    if atom.tag() == Atom {
        let string = atom.atom_to_string(process);
        Ok(Term::chars_to_list(string.chars(), &mut process))
    } else {
        Err(bad_argument!(&mut process))
    }
}

pub fn binary_part_3(binary: Term, start: Term, length: Term, mut process: &mut Process) -> Result {
    match binary.tag() {
        Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary.part(start, length, &mut process)
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.part(start, length, &mut process)
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn binary_to_atom_2(binary: Term, encoding: Term, process: &mut Process) -> Result {
    binary_existence_to_atom(binary, encoding, DoNotCare, process)
}

pub fn binary_to_existing_atom_2(binary: Term, encoding: Term, process: &mut Process) -> Result {
    binary_existence_to_atom(binary, encoding, Exists, process)
}

pub fn binary_to_float_1(binary: Term, mut process: &mut Process) -> Result {
    let string: String = binary.try_into_in_process(&mut process)?;

    match string.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) & !string.chars().any(|b| b == '.') {
                        Err(bad_argument!(&mut process))
                    } else {
                        Ok(inner.into_process(&mut process))
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan | FpCategory::Infinite => Err(bad_argument!(&mut process)),
                FpCategory::Zero => {
                    // Erlang does not track the difference without +0 and -0.
                    Ok(inner.abs().into_process(&mut process))
                }
            }
        }
        Err(_) => Err(bad_argument!(&mut process)),
    }
}

pub fn binary_to_integer_1(binary: Term, mut process: &mut Process) -> Result {
    let string: String = binary.try_into_in_process(&mut process)?;
    let bytes = string.as_bytes();

    match BigInt::parse_bytes(bytes, 10) {
        Some(big_int) => {
            let term: Term = big_int.into_process(&mut process);

            Ok(term)
        }
        None => Err(bad_argument!(&mut process)),
    }
}

pub fn binary_to_integer_2(binary: Term, base: Term, mut process: &mut Process) -> Result {
    let string: String = binary.try_into_in_process(&mut process)?;
    let radix: usize = base.try_into_in_process(&mut process)?;

    if 2 <= radix && radix <= 36 {
        let bytes = string.as_bytes();

        match BigInt::parse_bytes(bytes, radix as u32) {
            Some(big_int) => {
                let term: Term = big_int.into_process(&mut process);

                Ok(term)
            }
            None => Err(bad_argument!(&mut process)),
        }
    } else {
        Err(bad_argument!(&mut process))
    }
}

pub fn binary_to_list_1(binary: Term, mut process: &mut Process) -> Result {
    match binary.tag() {
        Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    Ok(heap_binary.to_list(&mut process))
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.to_list(&mut process)
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

/// The one-based indexing for binaries used by this function is deprecated. New code is to use
/// [crate::otp::binary::bin_to_list] instead. All functions in module [crate::otp::binary]
/// consistently use zero-based indexing.
pub fn binary_to_list_3(
    binary: Term,
    start: Term,
    stop: Term,
    mut process: &mut Process,
) -> Result {
    let one_based_start_usize: usize = start.try_into_in_process(&mut process)?;

    if 1 <= one_based_start_usize {
        let one_based_stop_usize: usize = stop.try_into_in_process(&mut process)?;

        if one_based_start_usize <= one_based_stop_usize {
            let zero_based_start_usize = one_based_start_usize - 1;
            let zero_based_stop_usize = one_based_stop_usize - 1;

            let length_usize = zero_based_stop_usize - zero_based_stop_usize + 1;

            otp::binary::bin_to_list(
                binary,
                zero_based_start_usize.into_process(&mut process),
                length_usize.into_process(&mut process),
                &mut process,
            )
        } else {
            Err(bad_argument!(&mut process))
        }
    } else {
        Err(bad_argument!(&mut process))
    }
}

pub fn binary_to_term_1(binary: Term, process: &mut Process) -> Result {
    binary_to_term_2(binary, Term::EMPTY_LIST, process)
}

pub fn binary_to_term_2(binary: Term, options: Term, mut process: &mut Process) -> Result {
    let to_term_options: ToTermOptions = options.try_into_in_process(process)?;

    match binary.tag() {
        Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary.to_term(to_term_options, &mut process)
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.to_term(to_term_options, &mut process)
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn bit_size_1(bit_string: Term, mut process: &mut Process) -> Result {
    match bit_string.tag() {
        Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.bit_size())
                }
                Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.bit_size())
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
    .map(|bit_size_usize| bit_size_usize.into_process(&mut process))
}

pub fn bitstring_to_list_1(bit_string: Term, mut process: &mut Process) -> Result {
    match bit_string.tag() {
        Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.to_bitstring_list(&mut process))
                }
                Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.to_bitstring_list(&mut process))
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn byte_size_1(bit_string: Term, mut process: &mut Process) -> Result {
    match bit_string.tag() {
        Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.byte_size())
                }
                Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.byte_size())
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
    .map(|byte_size_usize| byte_size_usize.into_process(&mut process))
}

pub fn ceil_1(number: Term, mut process: &mut Process) -> Result {
    match number.tag() {
        SmallInteger => Ok(number),
        Boxed => {
            let unboxed: &Term = number.unbox_reference();

            match unboxed.tag() {
                BigInteger => Ok(number),
                Float => {
                    let float: &Float = number.unbox_reference();
                    let inner = float.inner;
                    let ceil_inner = inner.ceil();

                    // skip creating a rug::Integer if float can fit in small integer.
                    let ceil_term =
                        if (small::MIN as f64) <= ceil_inner && ceil_inner <= (small::MAX as f64) {
                            (ceil_inner as usize).into_process(&mut process)
                        } else {
                            let ceil_string = ceil_inner.to_string();
                            let ceil_bytes = ceil_string.as_bytes();
                            let big_int = BigInt::parse_bytes(ceil_bytes, 10).unwrap();

                            big_int.into_process(&mut process)
                        };

                    Ok(ceil_term)
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

/// `++/2`
pub fn concatenate_2(list: Term, term: Term, mut process: &mut Process) -> Result {
    match list.tag() {
        EmptyList => Ok(term),
        List => {
            let cons: &Cons = unsafe { list.as_ref_cons_unchecked() };

            cons.concatenate(term, &mut process)
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn convert_time_unit_3(
    time: Term,
    from_unit: Term,
    to_unit: Term,
    mut process: &mut Process,
) -> Result {
    let time_big_int: BigInt = time.try_into_in_process(&mut process)?;
    let from_unit_unit: crate::time::Unit = from_unit.try_into_in_process(&mut process)?;
    let to_unit_unit: crate::time::Unit = to_unit.try_into_in_process(&mut process)?;
    let converted =
        time::convert(time_big_int, from_unit_unit, to_unit_unit).into_process(&mut process);

    Ok(converted)
}

pub fn delete_element_2(tuple: Term, index: Term, mut process: &mut Process) -> Result {
    let initial_inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into_in_process(&mut process)?;

    initial_inner_tuple
        .delete_element(index_zero_based, &mut process)
        .map(|final_inner_tuple| final_inner_tuple.into())
}

pub fn element_2(tuple: Term, index: Term, mut process: &mut Process) -> Result {
    let inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into_in_process(&mut process)?;

    inner_tuple.element(index_zero_based, &mut process)
}

pub fn error_1(reason: Term) -> Result {
    Err(error!(reason))
}

pub fn error_2(reason: Term, arguments: Term) -> Result {
    Err(error!(reason, arguments))
}

pub fn hd_1(list: Term, mut process: &mut Process) -> Result {
    let cons: &Cons = list.try_into_in_process(&mut process)?;

    Ok(cons.head())
}

pub fn insert_element_3(
    tuple: Term,
    index: Term,
    element: Term,
    mut process: &mut Process,
) -> Result {
    let initial_inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into_in_process(&mut process)?;

    initial_inner_tuple
        .insert_element(index_zero_based, element, &mut process)
        .map(|final_inner_tuple| final_inner_tuple.into())
}

pub fn is_atom_1(term: Term, mut process: &mut Process) -> Term {
    (term.tag() == Atom).into_process(&mut process)
}

pub fn is_binary_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                HeapBinary => true,
                Subbinary => {
                    let subbinary: &sub::Binary = term.unbox_reference();

                    subbinary.is_binary()
                }
                _ => false,
            }
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_boolean_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Atom => match term.atom_to_string(&mut process).as_ref() {
            "false" | "true" => true,
            _ => false,
        },
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_float_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                Float => true,
                _ => false,
            }
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_integer_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        SmallInteger => true,
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            unboxed.tag() == BigInteger
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_list_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        EmptyList | List => true,
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_map_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                Map => true,
                _ => false,
            }
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_map_key_2(key: Term, map: Term, mut process: &mut Process) -> Result {
    let map_map: &Map = map.try_into_in_process(&mut process)?;

    Ok(map_map.is_key(key).into_process(&mut process))
}

pub fn is_number_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        SmallInteger => true,
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                BigInteger | Float => true,
                _ => false,
            }
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_pid_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        LocalPid => true,
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                ExternalPid => true,
                _ => false,
            }
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_record_2(term: Term, record_tag: Term, mut process: &mut Process) -> Result {
    is_record(term, record_tag, None, &mut process)
}

pub fn is_record_3(term: Term, record_tag: Term, size: Term, mut process: &mut Process) -> Result {
    is_record(term, record_tag, Some(size), &mut process)
}

pub fn is_reference_1(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                LocalReference | ExternalReference => true,
                _ => false,
            }
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_tuple_1(term: Term, mut process: &mut Process) -> Term {
    (term.tag() == Boxed && term.unbox_reference::<Term>().tag() == Arity)
        .into_process(&mut process)
}

pub fn length_1(list: Term, mut process: &mut Process) -> Result {
    let mut length: usize = 0;
    let mut tail = list;

    loop {
        match tail.tag() {
            EmptyList => break Ok(length.into_process(&mut process)),
            List => {
                tail = crate::otp::erlang::tl_1(tail, &mut process).unwrap();
                length += 1;
            }
            _ => break Err(bad_argument!(&mut process)),
        }
    }
}

pub fn list_to_atom_1(string: Term, process: &mut Process) -> Result {
    list_to_atom(string, DoNotCare, process)
}

pub fn list_to_existing_atom_1(string: Term, process: &mut Process) -> Result {
    list_to_atom(string, Exists, process)
}

pub fn list_to_pid_1(string: Term, mut process: &mut Process) -> Result {
    let cons: &Cons = string.try_into_in_process(&mut process)?;

    cons.to_pid(&mut process)
}

pub fn list_to_tuple_1(list: Term, mut process: &mut Process) -> Result {
    match list.tag() {
        EmptyList => Ok(Term::slice_to_tuple(&[], &mut process)),
        List => {
            let cons: &Cons = unsafe { list.as_ref_cons_unchecked() };

            cons.to_tuple(&mut process)
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn make_ref_0(mut process: &mut Process) -> Term {
    Term::local_reference(&mut process)
}

pub fn map_get_2(key: Term, map: Term, mut process: &mut Process) -> Result {
    let map_map: &Map = map.try_into_in_process(&mut process)?;

    map_map.get(key, &mut process)
}

pub fn self_0(process: &Process) -> Term {
    process.pid
}

pub fn setelement_3(index: Term, tuple: Term, value: Term, mut process: &mut Process) -> Result {
    let inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into_in_process(&mut process)?;

    inner_tuple
        .setelement(index_zero_based, value, &mut process)
        .map(|new_inner_tuple| new_inner_tuple.into())
}

pub fn size_1(binary_or_tuple: Term, mut process: &mut Process) -> Result {
    match binary_or_tuple.tag() {
        Boxed => {
            let unboxed: &Term = binary_or_tuple.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = binary_or_tuple.unbox_reference();

                    Ok(tuple.size())
                }
                HeapBinary => {
                    let heap_binary: &heap::Binary = binary_or_tuple.unbox_reference();

                    Ok(heap_binary.size())
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary_or_tuple.unbox_reference();

                    Ok(subbinary.size())
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
    .map(|integer| integer.into_process(&mut process))
}

pub fn subtract_list_2(minuend: Term, subtrahend: Term, mut process: &mut Process) -> Result {
    match (minuend.tag(), subtrahend.tag()) {
        (EmptyList, EmptyList) => Ok(minuend),
        (EmptyList, List) => {
            let subtrahend_cons: &Cons = unsafe { subtrahend.as_ref_cons_unchecked() };

            if subtrahend_cons.is_proper() {
                Ok(minuend)
            } else {
                Err(bad_argument!(&mut process))
            }
        }
        (List, EmptyList) => Ok(minuend),
        (List, List) => {
            let minuend_cons: &Cons = unsafe { minuend.as_ref_cons_unchecked() };
            let subtrahend_cons: &Cons = unsafe { subtrahend.as_ref_cons_unchecked() };

            minuend_cons.subtract(subtrahend_cons, &mut process)
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn tl_1(list: Term, process: &mut Process) -> Result {
    let cons: &Cons = list.try_into_in_process(process)?;

    Ok(cons.tail())
}

pub fn tuple_size_1(tuple: Term, mut process: &mut Process) -> Result {
    match tuple.tag() {
        Boxed => {
            let unboxed: &Term = tuple.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = tuple.unbox_reference();

                    Ok(tuple.size().into_process(&mut process))
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

pub fn tuple_to_list_1(tuple: Term, mut process: &mut Process) -> Result {
    match tuple.tag() {
        Boxed => {
            let unboxed: &Term = tuple.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = tuple.unbox_reference();

                    Ok(tuple.to_list(&mut process))
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
}

// Private Functions

fn binary_existence_to_atom(
    binary: Term,
    encoding: Term,
    existence: Existence,
    mut process: &mut Process,
) -> Result {
    encoding.atom_to_encoding(&mut process)?;

    match binary.tag() {
        Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary.to_atom_index(existence, &mut process)
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.to_atom_index(existence, &mut process)
                }
                _ => Err(bad_argument!(&mut process)),
            }
        }
        _ => Err(bad_argument!(&mut process)),
    }
    .map(|atom_index| atom_index.into())
}

fn is_record(
    term: Term,
    record_tag: Term,
    size: Option<Term>,
    mut process: &mut Process,
) -> Result {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = term.unbox_reference();

                    tuple.is_record(record_tag, size, &mut process)
                }
                _ => Ok(false.into_process(&mut process)),
            }
        }
        _ => Ok(false.into_process(&mut process)),
    }
}

fn list_to_atom(string: Term, existence: Existence, mut process: &mut Process) -> Result {
    match string.tag() {
        EmptyList => Term::str_to_atom("", existence, &mut process),
        List => {
            let cons: &Cons = unsafe { string.as_ref_cons_unchecked() };

            cons.to_atom(existence, &mut process)
        }
        _ => Err(bad_argument!(&mut process)),
    }
}
