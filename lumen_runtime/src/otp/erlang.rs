//! Mirrors [erlang](http://erlang::org/doc/man/erlang::html) module

use std::cmp::Ordering;
use std::convert::TryInto;
use std::num::FpCategory;

use num_bigint::BigInt;
use num_traits::Zero;

use crate::atom::{Existence, Existence::*};
use crate::binary::{heap, sub, Part, ToTerm, ToTermOptions};
use crate::exception::{Class, Result};
use crate::float::Float;
use crate::integer::{big, small};
use crate::list::Cons;
use crate::map::Map;
use crate::otp;
use crate::process::{IntoProcess, Process, TryIntoInProcess};
use crate::stacktrace;
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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
}

/// `+/2` infix operator
pub fn add_2(augend: Term, addend: Term, mut process: &mut Process) -> Result {
    number_infix_operator!(augend, addend, process, checked_add, +)
}

pub fn append_element_2(tuple: Term, element: Term, mut process: &mut Process) -> Result {
    let internal: &Tuple = tuple.try_into_in_process(&mut process)?;
    let new_tuple = internal.append_element(element, &mut process.term_arena);

    Ok(new_tuple.into())
}

pub fn atom_to_binary_2(atom: Term, encoding: Term, mut process: &mut Process) -> Result {
    if atom.tag() == Atom {
        encoding.atom_to_encoding()?;
        let string = unsafe { atom.atom_to_string() };
        Ok(Term::slice_to_binary(string.as_bytes(), &mut process))
    } else {
        Err(badarg!())
    }
}

pub fn atom_to_list_1(atom: Term, mut process: &mut Process) -> Result {
    if atom.tag() == Atom {
        let string = unsafe { atom.atom_to_string() };
        Ok(Term::chars_to_list(string.chars(), &mut process))
    } else {
        Err(badarg!())
    }
}

// `band/2` infix operator.
pub fn band_2(left_integer: Term, right_integer: Term, mut process: &mut Process) -> Result {
    bitwise_infix_operator!(left_integer, right_integer, process, &)
}

pub fn binary_part_2(binary: Term, start_length: Term, mut process: &mut Process) -> Result {
    match start_length.tag() {
        Boxed => {
            let unboxed: &Term = start_length.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = start_length.unbox_reference();

                    if tuple.len() == 2 {
                        binary_part_3(binary, tuple[0], tuple[1], &mut process)
                    } else {
                        Err(badarg!())
                    }
                }
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
}

pub fn binary_to_atom_2(binary: Term, encoding: Term) -> Result {
    binary_existence_to_atom(binary, encoding, DoNotCare)
}

pub fn binary_to_existing_atom_2(binary: Term, encoding: Term) -> Result {
    binary_existence_to_atom(binary, encoding, Exists)
}

pub fn binary_to_float_1(binary: Term, mut process: &mut Process) -> Result {
    let string: String = binary.try_into()?;

    match string.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) & !string.chars().any(|b| b == '.') {
                        Err(badarg!())
                    } else {
                        Ok(inner.into_process(&mut process))
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan | FpCategory::Infinite => Err(badarg!()),
                FpCategory::Zero => {
                    // Erlang does not track the difference without +0 and -0.
                    Ok(inner.abs().into_process(&mut process))
                }
            }
        }
        Err(_) => Err(badarg!()),
    }
}

pub fn binary_to_integer_1(binary: Term, mut process: &mut Process) -> Result {
    let string: String = binary.try_into()?;
    let bytes = string.as_bytes();

    match BigInt::parse_bytes(bytes, 10) {
        Some(big_int) => {
            let term: Term = big_int.into_process(&mut process);

            Ok(term)
        }
        None => Err(badarg!()),
    }
}

pub fn binary_to_integer_2(binary: Term, base: Term, mut process: &mut Process) -> Result {
    let string: String = binary.try_into()?;
    let radix: usize = base.try_into()?;

    if 2 <= radix && radix <= 36 {
        let bytes = string.as_bytes();

        match BigInt::parse_bytes(bytes, radix as u32) {
            Some(big_int) => {
                let term: Term = big_int.into_process(&mut process);

                Ok(term)
            }
            None => Err(badarg!()),
        }
    } else {
        Err(badarg!())
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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
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
    let one_based_start_usize: usize = start.try_into()?;

    if 1 <= one_based_start_usize {
        let one_based_stop_usize: usize = stop.try_into()?;

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
            Err(badarg!())
        }
    } else {
        Err(badarg!())
    }
}

pub fn binary_to_term_1(binary: Term, process: &mut Process) -> Result {
    binary_to_term_2(binary, Term::EMPTY_LIST, process)
}

pub fn binary_to_term_2(binary: Term, options: Term, mut process: &mut Process) -> Result {
    let to_term_options: ToTermOptions = options.try_into()?;

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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
}

pub fn bit_size_1(bit_string: Term, mut process: &mut Process) -> Result {
    match bit_string.tag() {
        Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.bit_len())
                }
                Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.bit_len())
                }
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
}

pub fn byte_size_1(bit_string: Term, mut process: &mut Process) -> Result {
    match bit_string.tag() {
        Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.byte_len())
                }
                Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.byte_len())
                }
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
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
        _ => Err(badarg!()),
    }
}

pub fn convert_time_unit_3(
    time: Term,
    from_unit: Term,
    to_unit: Term,
    mut process: &mut Process,
) -> Result {
    let time_big_int: BigInt = time.try_into()?;
    let from_unit_unit: crate::time::Unit = from_unit.try_into()?;
    let to_unit_unit: crate::time::Unit = to_unit.try_into()?;
    let converted =
        time::convert(time_big_int, from_unit_unit, to_unit_unit).into_process(&mut process);

    Ok(converted)
}

pub fn delete_element_2(tuple: Term, index: Term, mut process: &mut Process) -> Result {
    let initial_inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into()?;

    initial_inner_tuple
        .delete_element(index_zero_based, &mut process)
        .map(|final_inner_tuple| final_inner_tuple.into())
}

/// `div/2` infix operator.  Integer division.
pub fn div_2(dividend: Term, divisor: Term, mut process: &mut Process) -> Result {
    integer_infix_operator!(dividend, divisor, process, /)
}

/// `//2` infix operator.  Unlike `+/2`, `-/2` and `*/2` always promotes to `float` returns the
/// `float`.
pub fn divide_2(dividend: Term, divisor: Term, mut process: &mut Process) -> Result {
    let dividend_f64: f64 = dividend.try_into()?;
    let divisor_f64: f64 = divisor.try_into()?;

    if divisor_f64 == 0.0 {
        Err(badarith!())
    } else {
        let quotient_f64 = dividend_f64 / divisor_f64;

        Ok(quotient_f64.into_process(&mut process))
    }
}

pub fn element_2(tuple: Term, index: Term, mut process: &mut Process) -> Result {
    let inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into()?;

    inner_tuple.element(index_zero_based)
}

pub fn error_1(reason: Term) -> Result {
    Err(error!(reason))
}

pub fn error_2(reason: Term, arguments: Term) -> Result {
    Err(error!(reason, Some(arguments)))
}

pub fn exit_1(reason: Term) -> Result {
    Err(exit!(reason))
}

pub fn hd_1(list: Term) -> Result {
    let cons: &Cons = list.try_into()?;

    Ok(cons.head())
}

pub fn insert_element_3(
    tuple: Term,
    index: Term,
    element: Term,
    mut process: &mut Process,
) -> Result {
    let initial_inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into()?;

    initial_inner_tuple
        .insert_element(index_zero_based, element, &mut process)
        .map(|final_inner_tuple| final_inner_tuple.into())
}

/// Distribution is not supported at this time.  Always returns `false`.
pub fn is_alive_0() -> Term {
    false.into()
}

pub fn is_atom_1(term: Term) -> Term {
    term.is_atom().into()
}

pub fn is_binary_1(term: Term) -> Term {
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
    .into()
}

pub fn is_bitstring_1(term: Term) -> Term {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                HeapBinary | Subbinary => true,
                _ => false,
            }
        }
        _ => false,
    }
    .into()
}

pub fn is_boolean_1(term: Term) -> Term {
    match term.tag() {
        Atom => match unsafe { term.atom_to_string() }.as_ref().as_ref() {
            "false" | "true" => true,
            _ => false,
        },
        _ => false,
    }
    .into()
}

pub fn is_float_1(term: Term) -> Term {
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
    .into()
}

pub fn is_integer_1(term: Term) -> Term {
    match term.tag() {
        SmallInteger => true,
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            unboxed.tag() == BigInteger
        }
        _ => false,
    }
    .into()
}

pub fn is_list_1(term: Term) -> Term {
    match term.tag() {
        EmptyList | List => true,
        _ => false,
    }
    .into()
}

pub fn is_map_1(term: Term) -> Term {
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
    .into()
}

pub fn is_map_key_2(key: Term, map: Term, mut process: &mut Process) -> Result {
    let map_map: &Map = map.try_into_in_process(&mut process)?;

    Ok(map_map.is_key(key).into())
}

pub fn is_number_1(term: Term) -> Term {
    term.is_number().into()
}

pub fn is_pid_1(term: Term) -> Term {
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
    .into()
}

pub fn is_record_2(term: Term, record_tag: Term) -> Result {
    is_record(term, record_tag, None)
}

pub fn is_record_3(term: Term, record_tag: Term, size: Term) -> Result {
    is_record(term, record_tag, Some(size))
}

pub fn is_reference_1(term: Term) -> Term {
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
    .into()
}

pub fn is_tuple_1(term: Term) -> Term {
    (term.tag() == Boxed && term.unbox_reference::<Term>().tag() == Arity).into()
}

pub fn length_1(list: Term, mut process: &mut Process) -> Result {
    let mut length: usize = 0;
    let mut tail = list;

    loop {
        match tail.tag() {
            EmptyList => break Ok(length.into_process(&mut process)),
            List => {
                let cons: &Cons = unsafe { tail.as_ref_cons_unchecked() };
                tail = cons.tail();
                length += 1;
            }
            _ => break Err(badarg!()),
        }
    }
}

pub fn list_to_atom_1(string: Term) -> Result {
    list_to_atom(string, DoNotCare)
}

pub fn list_to_existing_atom_1(string: Term) -> Result {
    list_to_atom(string, Exists)
}

pub fn list_to_binary_1(iolist: Term, mut process: &mut Process) -> Result {
    match iolist.tag() {
        EmptyList | List => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut stack: Vec<Term> = vec![iolist];

            while let Some(top) = stack.pop() {
                match top.tag() {
                    SmallInteger => {
                        let top_isize = unsafe { top.small_integer_to_isize() };
                        let top_byte = top_isize.try_into().map_err(|_| badarg!())?;

                        byte_vec.push(top_byte);
                    }
                    EmptyList => (),
                    List => {
                        let cons: &Cons = unsafe { top.as_ref_cons_unchecked() };

                        // @type iolist :: maybe_improper_list(byte() | binary() | iolist(),
                        // binary() | []) means that `byte()` isn't allowed
                        // for `tail`s unlike `head`.

                        let tail = cons.tail();

                        if tail.tag() == SmallInteger {
                            return Err(badarg!());
                        } else {
                            stack.push(tail);
                        }

                        stack.push(cons.head());
                    }
                    Boxed => {
                        let unboxed: &Term = top.unbox_reference();

                        match unboxed.tag() {
                            HeapBinary => {
                                let heap_binary: &heap::Binary = top.unbox_reference();

                                byte_vec.extend_from_slice(heap_binary.as_slice());
                            }
                            Subbinary => {
                                let subbinary: &sub::Binary = top.unbox_reference();

                                if subbinary.bit_count == 0 {
                                    byte_vec.extend(subbinary.byte_iter());
                                } else {
                                    return Err(badarg!());
                                }
                            }
                            _ => return Err(badarg!()),
                        }
                    }
                    _ => return Err(badarg!()),
                }
            }

            Ok(Term::slice_to_binary(byte_vec.as_slice(), &mut process))
        }
        _ => Err(badarg!()),
    }
}

pub fn list_to_bitstring_1(iolist: Term, mut process: &mut Process) -> Result {
    match iolist.tag() {
        EmptyList | List => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut bit_offset = 0;
            let mut tail_byte = 0;
            let mut stack: Vec<Term> = vec![iolist];

            while let Some(top) = stack.pop() {
                match top.tag() {
                    SmallInteger => {
                        let top_isize = unsafe { top.small_integer_to_isize() };
                        let top_byte = top_isize.try_into().map_err(|_| badarg!())?;

                        if bit_offset == 0 {
                            byte_vec.push(top_byte);
                        } else {
                            tail_byte |= top_byte >> bit_offset;
                            byte_vec.push(tail_byte);

                            tail_byte = top_byte << (8 - bit_offset);
                        }
                    }
                    EmptyList => (),
                    List => {
                        let cons: &Cons = unsafe { top.as_ref_cons_unchecked() };

                        // @type bitstring_list ::
                        //   maybe_improper_list(byte() | bitstring() | bitstring_list(),
                        //                       bitstring() | [])
                        // means that `byte()` isn't allowed for `tail`s unlike `head`.

                        let tail = cons.tail();

                        if tail.tag() == SmallInteger {
                            return Err(badarg!());
                        } else {
                            stack.push(tail);
                        }

                        stack.push(cons.head());
                    }
                    Boxed => {
                        let unboxed: &Term = top.unbox_reference();

                        match unboxed.tag() {
                            HeapBinary => {
                                let heap_binary: &heap::Binary = top.unbox_reference();

                                if bit_offset == 0 {
                                    byte_vec.extend_from_slice(heap_binary.as_slice());
                                } else {
                                    for byte in heap_binary.byte_iter() {
                                        tail_byte |= byte >> bit_offset;
                                        byte_vec.push(tail_byte);

                                        tail_byte = byte << (8 - bit_offset);
                                    }
                                }
                            }
                            Subbinary => {
                                let subbinary: &sub::Binary = top.unbox_reference();

                                if bit_offset == 0 {
                                    byte_vec.extend(subbinary.byte_iter());
                                } else {
                                    for byte in subbinary.byte_iter() {
                                        tail_byte |= byte >> bit_offset;
                                        byte_vec.push(tail_byte);

                                        tail_byte = byte << (8 - bit_offset);
                                    }
                                }

                                if 0 < subbinary.bit_count {
                                    for bit in subbinary.bit_count_iter() {
                                        tail_byte |= bit << (7 - bit_offset);

                                        if bit_offset == 7 {
                                            byte_vec.push(tail_byte);
                                            bit_offset = 0;
                                            tail_byte = 0;
                                        } else {
                                            bit_offset += 1;
                                        }
                                    }
                                }
                            }
                            _ => return Err(badarg!()),
                        }
                    }
                    _ => return Err(badarg!()),
                }
            }

            if bit_offset == 0 {
                Ok(Term::slice_to_binary(byte_vec.as_slice(), &mut process))
            } else {
                let byte_count = byte_vec.len();
                byte_vec.push(tail_byte);
                let original = Term::slice_to_binary(byte_vec.as_slice(), &mut process);

                Ok(Term::subbinary(
                    original,
                    0,
                    0,
                    byte_count,
                    bit_offset,
                    &mut process,
                ))
            }
        }
        _ => Err(badarg!()),
    }
}

pub fn list_to_pid_1(string: Term, mut process: &mut Process) -> Result {
    let cons: &Cons = string.try_into()?;

    cons.to_pid(&mut process)
}

pub fn list_to_tuple_1(list: Term, mut process: &mut Process) -> Result {
    match list.tag() {
        EmptyList => Ok(Term::slice_to_tuple(&[], &mut process)),
        List => {
            let cons: &Cons = unsafe { list.as_ref_cons_unchecked() };

            cons.to_tuple(&mut process)
        }
        _ => Err(badarg!()),
    }
}

pub fn make_ref_0(mut process: &mut Process) -> Term {
    Term::local_reference(&mut process)
}

pub fn map_get_2(key: Term, map: Term, mut process: &mut Process) -> Result {
    let map_map: &Map = map.try_into_in_process(&mut process)?;

    map_map.get(key, &mut process)
}

pub fn map_size_1(map: Term, mut process: &mut Process) -> Result {
    let map_map: &Map = map.try_into_in_process(&mut process)?;

    Ok(map_map.size().into_process(&mut process))
}

/// `*/2` infix operator
pub fn multiply_2(multiplier: Term, multiplicand: Term, mut process: &mut Process) -> Result {
    number_infix_operator!(multiplier, multiplicand, process, checked_mul, *)
}

/// `-/1` prefix operator.
pub fn negate_1(number: Term, mut process: &mut Process) -> Result {
    match number.tag() {
        SmallInteger => {
            let number_isize = unsafe { number.small_integer_to_isize() };
            let negated_isize = -number_isize;

            Ok(negated_isize.into_process(&mut process))
        }
        Boxed => {
            let unboxed: &Term = number.unbox_reference();

            match unboxed.tag() {
                BigInteger => {
                    let big_integer: &big::Integer = number.unbox_reference();
                    let negated_big_int = -&big_integer.inner;

                    Ok(negated_big_int.into_process(&mut process))
                }
                Float => {
                    let float: &Float = number.unbox_reference();
                    let negated_f64 = -float.inner;

                    Ok(negated_f64.into_process(&mut process))
                }
                _ => Err(badarith!()),
            }
        }
        _ => Err(badarith!()),
    }
}

pub fn node_0() -> Term {
    Term::str_to_atom("nonode@nohost", DoNotCare).unwrap()
}

/// `+/1` prefix operator.
pub fn number_or_badarith_1(term: Term) -> Result {
    if term.is_number() {
        Ok(term)
    } else {
        Err(badarith!())
    }
}

pub fn raise_3(class: Term, reason: Term, stacktrace: Term) -> Result {
    let class_class: Class = class.try_into()?;

    if stacktrace::is(stacktrace) {
        Err(raise!(class_class, reason, Some(stacktrace)))
    } else {
        Err(badarg!())
    }
}

/// `rem/2` infix operator.  Integer remainder.
pub fn rem_2(dividend: Term, divisor: Term, mut process: &mut Process) -> Result {
    integer_infix_operator!(dividend, divisor, process, %)
}

pub fn self_0(process: &Process) -> Term {
    process.pid
}

pub fn setelement_3(index: Term, tuple: Term, value: Term, mut process: &mut Process) -> Result {
    let inner_tuple: &Tuple = tuple.try_into_in_process(&mut process)?;
    let index_zero_based: ZeroBasedIndex = index.try_into()?;

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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
    .map(|integer| integer.into_process(&mut process))
}

pub fn split_binary_2(binary: Term, position: Term, mut process: &mut Process) -> Result {
    let index: usize = position.try_into()?;

    match binary.tag() {
        Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    if index == 0 {
                        let empty_prefix = Term::subbinary(binary, index, 0, 0, 0, &mut process);

                        // Don't make a subbinary of the suffix since it is the same as the
                        // `binary`.
                        Ok(Term::slice_to_tuple(&[empty_prefix, binary], &mut process))
                    } else {
                        let heap_binary: &heap::Binary = binary.unbox_reference();
                        let byte_length = heap_binary.byte_len();

                        if index < byte_length {
                            let prefix = Term::subbinary(binary, 0, 0, index, 0, &mut process);
                            let suffix = Term::subbinary(
                                binary,
                                index,
                                0,
                                byte_length - index,
                                0,
                                &mut process,
                            );

                            Ok(Term::slice_to_tuple(&[prefix, suffix], &mut process))
                        } else if index == byte_length {
                            let empty_suffix =
                                Term::subbinary(binary, index, 0, 0, 0, &mut process);

                            // Don't make a subbinary of the prefix since it is the same as the
                            // `binary`.
                            Ok(Term::slice_to_tuple(&[binary, empty_suffix], &mut process))
                        } else {
                            Err(badarg!())
                        }
                    }
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    if index == 0 {
                        let empty_prefix = Term::subbinary(
                            subbinary.original,
                            subbinary.byte_offset + index,
                            subbinary.bit_offset,
                            0,
                            0,
                            &mut process,
                        );

                        // Don't make a subbinary of the suffix since it is the same as the
                        // `binary`.
                        Ok(Term::slice_to_tuple(&[empty_prefix, binary], &mut process))
                    } else {
                        // byte_length includes +1 byte if bits
                        let byte_length = subbinary.byte_len();

                        if index < byte_length {
                            let original = subbinary.original;
                            let byte_offset = subbinary.byte_offset;
                            let bit_offset = subbinary.bit_offset;
                            let prefix = Term::subbinary(
                                original,
                                byte_offset,
                                bit_offset,
                                index,
                                0,
                                &mut process,
                            );
                            let suffix = Term::subbinary(
                                original,
                                byte_offset + index,
                                bit_offset,
                                // byte_count does not include bits
                                subbinary.byte_count - index,
                                subbinary.bit_count,
                                &mut process,
                            );

                            Ok(Term::slice_to_tuple(&[prefix, suffix], &mut process))
                        } else if (index == byte_length) & (subbinary.bit_count == 0) {
                            let empty_suffix = Term::subbinary(
                                subbinary.original,
                                subbinary.byte_offset + index,
                                subbinary.bit_offset,
                                0,
                                0,
                                &mut process,
                            );

                            Ok(Term::slice_to_tuple(&[binary, empty_suffix], &mut process))
                        } else {
                            Err(badarg!())
                        }
                    }
                }
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
}

/// `-/2` infix operator
pub fn subtract_2(minuend: Term, subtrahend: Term, mut process: &mut Process) -> Result {
    number_infix_operator!(minuend, subtrahend, process, checked_sub, -)
}

pub fn subtract_list_2(minuend: Term, subtrahend: Term, mut process: &mut Process) -> Result {
    match (minuend.tag(), subtrahend.tag()) {
        (EmptyList, EmptyList) => Ok(minuend),
        (EmptyList, List) => {
            let subtrahend_cons: &Cons = unsafe { subtrahend.as_ref_cons_unchecked() };

            if subtrahend_cons.is_proper() {
                Ok(minuend)
            } else {
                Err(badarg!())
            }
        }
        (List, EmptyList) => Ok(minuend),
        (List, List) => {
            let minuend_cons: &Cons = unsafe { minuend.as_ref_cons_unchecked() };
            let subtrahend_cons: &Cons = unsafe { subtrahend.as_ref_cons_unchecked() };

            minuend_cons.subtract(subtrahend_cons, &mut process)
        }
        _ => Err(badarg!()),
    }
}

pub fn throw_1(reason: Term) -> Result {
    Err(throw!(reason))
}

pub fn tl_1(list: Term) -> Result {
    let cons: &Cons = list.try_into()?;

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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
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
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
}

// Private Functions

fn binary_existence_to_atom(binary: Term, encoding: Term, existence: Existence) -> Result {
    encoding.atom_to_encoding()?;

    match binary.tag() {
        Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary
                        .to_atom_index(existence)
                        .ok_or_else(|| badarg!())
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.to_atom_index(existence)
                }
                _ => Err(badarg!()),
            }
        }
        _ => Err(badarg!()),
    }
    .map(|atom_index| atom_index.into())
}

fn is_record(term: Term, record_tag: Term, size: Option<Term>) -> Result {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = term.unbox_reference();

                    tuple.is_record(record_tag, size)
                }
                _ => Ok(false.into()),
            }
        }
        _ => Ok(false.into()),
    }
}

fn list_to_atom(string: Term, existence: Existence) -> Result {
    match string.tag() {
        EmptyList => Term::str_to_atom("", existence).ok_or_else(|| badarg!()),
        List => {
            let cons: &Cons = unsafe { string.as_ref_cons_unchecked() };

            cons.to_atom(existence)
        }
        _ => Err(badarg!()),
    }
}
