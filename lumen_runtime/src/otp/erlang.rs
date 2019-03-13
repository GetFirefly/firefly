//! Mirrors [erlang](http://erlang::org/doc/man/erlang::html) module

#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering;
use std::convert::TryInto;
use std::num::FpCategory;

use num_bigint::BigInt;
use num_traits::Zero;

use crate::atom::Existence;
use crate::bad_argument::BadArgument;
use crate::binary::{heap, sub, Part, ToTerm, ToTermOptions};
use crate::float::Float;
use crate::integer::{big, small};
use crate::list::Cons;
use crate::otp;
use crate::process::{IntoProcess, Process};
use crate::term::{Tag, Term};
use crate::time;
use crate::tuple::Tuple;

pub fn abs(number: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    match number.tag() {
        Tag::SmallInteger => {
            if unsafe { number.small_integer_is_negative() } {
                // cast first so that sign bit is extended on shift
                let signed = (number.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT;
                let positive = -signed;
                Ok(Term {
                    tagged: ((positive << Tag::SMALL_INTEGER_BIT_COUNT) as usize)
                        | (Tag::SmallInteger as usize),
                })
            } else {
                Ok(Term {
                    tagged: number.tagged,
                })
            }
        }
        Tag::Boxed => {
            let unboxed: &Term = number.unbox_reference();

            match unboxed.tag() {
                Tag::BigInteger => {
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
                Tag::Float => {
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
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
}

pub fn append_element(
    tuple: Term,
    element: Term,
    process: &mut Process,
) -> Result<Term, BadArgument> {
    let internal: &Tuple = tuple.try_into()?;
    let new_tuple = internal.append_element(element, &mut process.term_arena);

    Ok(new_tuple.into())
}

pub fn atom_to_binary(
    atom: Term,
    encoding: Term,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
    if let Tag::Atom = atom.tag() {
        encoding.atom_to_encoding(&mut process)?;
        let string = atom.atom_to_string(process);
        Ok(Term::slice_to_binary(string.as_bytes(), &mut process))
    } else {
        Err(bad_argument!())
    }
}

pub fn atom_to_list(
    atom: Term,
    encoding: Term,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
    if let Tag::Atom = atom.tag() {
        encoding.atom_to_encoding(&mut process)?;
        let string = atom.atom_to_string(process);
        Ok(Term::chars_to_list(string.chars(), &mut process))
    } else {
        Err(bad_argument!())
    }
}

pub fn binary_part(
    binary: Term,
    start: Term,
    length: Term,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
    match binary.tag() {
        Tag::Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary.part(start, length, &mut process)
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.part(start, length, &mut process)
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
}

pub fn binary_to_atom(
    binary: Term,
    encoding: Term,
    process: &mut Process,
) -> Result<Term, BadArgument> {
    binary_existence_to_atom(binary, encoding, Existence::DoNotCare, process)
}

pub fn binary_to_existing_atom(
    binary: Term,
    encoding: Term,
    process: &mut Process,
) -> Result<Term, BadArgument> {
    binary_existence_to_atom(binary, encoding, Existence::Exists, process)
}

pub fn binary_to_float(binary: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    let string: String = binary.try_into()?;

    match string.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) & !string.chars().any(|b| b == '.') {
                        Err(bad_argument!())
                    } else {
                        Ok(inner.into_process(&mut process))
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan | FpCategory::Infinite => Err(bad_argument!()),
                FpCategory::Zero => {
                    // Erlang does not track the difference without +0 and -0.
                    Ok(inner.abs().into_process(&mut process))
                }
            }
        }
        Err(_) => Err(bad_argument!()),
    }
}

/// `binary_to_integer/1`
pub fn binary_to_integer(binary: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    let string: String = binary.try_into()?;
    let bytes = string.as_bytes();

    match BigInt::parse_bytes(bytes, 10) {
        Some(big_int) => {
            let term: Term = big_int.into_process(&mut process);

            Ok(term)
        }
        None => Err(bad_argument!()),
    }
}

/// `binary_to_integer/2`
pub fn binary_in_base_to_integer(
    binary: Term,
    base: Term,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
    let string: String = binary.try_into()?;
    let radix: usize = base.try_into()?;

    if 2 <= radix && radix <= 36 {
        let bytes = string.as_bytes();

        match BigInt::parse_bytes(bytes, radix as u32) {
            Some(big_int) => {
                let term: Term = big_int.into_process(&mut process);

                Ok(term)
            }
            None => Err(bad_argument!()),
        }
    } else {
        Err(bad_argument!())
    }
}

/// `binary_to_list/1`
pub fn binary_to_list(binary: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    match binary.tag() {
        Tag::Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    Ok(heap_binary.to_list(&mut process))
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.to_list(&mut process)
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
}

/// `binary_to_list/3`
///
/// The one-based indexing for binaries used by this function is deprecated. New code is to use
/// [crate::otp::binary::bin_to_list] instead. All functions in module [crate::otp::binary]
/// consistently use zero-based indexing.
pub fn binary_byte_range_to_list(
    binary: Term,
    start: Term,
    stop: Term,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
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
            Err(bad_argument!())
        }
    } else {
        Err(bad_argument!())
    }
}

/// `binary_to_term/1`
pub fn binary_to_term(binary: Term, process: &mut Process) -> Result<Term, BadArgument> {
    binary_options_to_term(binary, Term::EMPTY_LIST, process)
}

/// `binary_to_term/2`
pub fn binary_options_to_term(
    binary: Term,
    options: Term,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
    let to_term_options = ToTermOptions::try_from(options, process)?;

    match binary.tag() {
        Tag::Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary.to_term(to_term_options, &mut process)
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.to_term(to_term_options, &mut process)
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
}

pub fn bit_size(bit_string: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    match bit_string.tag() {
        Tag::Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.bit_size())
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.bit_size())
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
    .map(|bit_size_usize| bit_size_usize.into_process(&mut process))
}

pub fn bitstring_to_list(bit_string: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    match bit_string.tag() {
        Tag::Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.to_bitstring_list(&mut process))
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.to_bitstring_list(&mut process))
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
}

pub fn byte_size(bit_string: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    match bit_string.tag() {
        Tag::Boxed => {
            let unboxed: &Term = bit_string.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = bit_string.unbox_reference();

                    Ok(heap_binary.byte_size())
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = bit_string.unbox_reference();

                    Ok(subbinary.byte_size())
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
    .map(|byte_size_usize| byte_size_usize.into_process(&mut process))
}

pub fn ceil(number: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    match number.tag() {
        Tag::SmallInteger => Ok(number),
        Tag::Boxed => {
            let unboxed: &Term = number.unbox_reference();

            match unboxed.tag() {
                Tag::BigInteger => Ok(number),
                Tag::Float => {
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
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
}

pub fn convert_time_unit(
    time: Term,
    from_unit: Term,
    to_unit: Term,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
    let time_big_int: BigInt = time.try_into()?;
    let from_unit_unit = crate::time::Unit::try_from(from_unit, &mut process)?;
    let to_unit_unit = crate::time::Unit::try_from(to_unit, &mut process)?;
    let converted =
        time::convert(time_big_int, from_unit_unit, to_unit_unit).into_process(&mut process);

    Ok(converted)
}

pub fn delete_element(
    tuple: Term,
    index: Term,
    process: &mut Process,
) -> Result<Term, BadArgument> {
    let initial_inner_tuple: &Tuple = tuple.try_into()?;
    let inner_index: usize = index.try_into()?;

    initial_inner_tuple
        .delete_element(inner_index, &mut process.term_arena)
        .map(|final_inner_tuple| final_inner_tuple.into())
}

pub fn element(tuple: Term, index: Term) -> Result<Term, BadArgument> {
    let inner_tuple: &Tuple = tuple.try_into()?;
    let inner_index: usize = index.try_into()?;

    inner_tuple.element(inner_index)
}

pub fn head(list: Term) -> Result<Term, BadArgument> {
    let cons: &Cons = list.try_into()?;

    Ok(cons.head())
}

pub fn insert_element(
    tuple: Term,
    index: Term,
    element: Term,
    process: &mut Process,
) -> Result<Term, BadArgument> {
    let initial_inner_tuple: &Tuple = tuple.try_into()?;
    let inner_index: usize = index.try_into()?;

    initial_inner_tuple
        .insert_element(inner_index, element, &mut process.term_arena)
        .map(|final_inner_tuple| final_inner_tuple.into())
}

pub fn is_atom(term: Term, mut process: &mut Process) -> Term {
    (term.tag() == Tag::Atom).into_process(&mut process)
}

pub fn is_binary(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Tag::Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => true,
                Tag::Subbinary => {
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

pub fn is_integer(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Tag::SmallInteger => true,
        Tag::Boxed => {
            let unboxed: &Term = term.unbox_reference();

            unboxed.tag() == Tag::BigInteger
        }
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_list(term: Term, mut process: &mut Process) -> Term {
    match term.tag() {
        Tag::EmptyList | Tag::List => true,
        _ => false,
    }
    .into_process(&mut process)
}

pub fn is_tuple(term: Term, mut process: &mut Process) -> Term {
    (term.tag() == Tag::Boxed && term.unbox_reference::<Term>().tag() == Tag::Arity)
        .into_process(&mut process)
}

pub fn length(list: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    let mut length: usize = 0;
    let mut tail = list;

    loop {
        match tail.tag() {
            Tag::EmptyList => break Ok(length.into_process(&mut process)),
            Tag::List => {
                tail = crate::otp::erlang::tail(tail).unwrap();
                length += 1;
            }
            _ => break Err(bad_argument!()),
        }
    }
}

pub fn list_to_pid(string: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    let cons: &Cons = string.try_into()?;

    cons.to_pid(&mut process)
}

pub fn size(binary_or_tuple: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
    match binary_or_tuple.tag() {
        Tag::Boxed => {
            let unboxed: &Term = binary_or_tuple.unbox_reference();

            match unboxed.tag() {
                Tag::Arity => {
                    let tuple: &Tuple = binary_or_tuple.unbox_reference();

                    Ok(tuple.size())
                }
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = binary_or_tuple.unbox_reference();

                    Ok(heap_binary.size())
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = binary_or_tuple.unbox_reference();

                    Ok(subbinary.size())
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
    .map(|integer| integer.into_process(&mut process))
}

pub fn tail(list: Term) -> Result<Term, BadArgument> {
    let cons: &Cons = list.try_into()?;

    Ok(cons.tail())
}

// Private Functions

fn binary_existence_to_atom(
    binary: Term,
    encoding: Term,
    existence: Existence,
    mut process: &mut Process,
) -> Result<Term, BadArgument> {
    encoding.atom_to_encoding(&mut process)?;

    match binary.tag() {
        Tag::Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                Tag::HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary.to_atom_index(existence, &mut process)
                }
                Tag::Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.to_atom_index(existence, &mut process)
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
    .map(|atom_index| atom_index.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cmp::Ordering;

    use crate::otp::erlang;

    mod abs {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::abs(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0], &mut process);

            assert_eq_in_process!(
                erlang::abs(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(
                erlang::abs(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::abs(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                Default::default()
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::abs(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_that_is_negative_returns_positive() {
            let mut process: Process = Default::default();

            let negative: isize = -1;
            let negative_term = negative.into_process(&mut process);

            let positive = -negative;
            let positive_term = positive.into_process(&mut process);

            assert_eq_in_process!(
                erlang::abs(negative_term, &mut process),
                Ok(positive_term),
                process
            );
        }

        #[test]
        fn with_small_integer_that_is_positive_returns_self() {
            let mut process: Process = Default::default();
            let positive_term = 1usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::abs(positive_term, &mut process),
                Ok(positive_term),
                process
            );
        }

        #[test]
        fn with_big_integer_that_is_negative_returns_positive() {
            let mut process: Process = Default::default();

            let negative: isize = small::MIN - 1;
            let negative_term = negative.into_process(&mut process);

            assert_eq!(negative_term.tag(), Tag::Boxed);

            let unboxed_negative_term: &Term = negative_term.unbox_reference();

            assert_eq!(unboxed_negative_term.tag(), Tag::BigInteger);

            let positive = -negative;
            let positive_term = positive.into_process(&mut process);

            assert_eq_in_process!(
                erlang::abs(negative_term, &mut process),
                Ok(positive_term),
                process
            );
        }

        #[test]
        fn with_big_integer_that_is_positive_return_self() {
            let mut process: Process = Default::default();
            let positive_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq!(positive_term.tag(), Tag::Boxed);

            let unboxed_positive_term: &Term = positive_term.unbox_reference();

            assert_eq!(unboxed_positive_term.tag(), Tag::BigInteger);

            assert_eq_in_process!(
                erlang::abs(positive_term, &mut process),
                Ok(positive_term),
                process
            );
        }

        #[test]
        fn with_float_that_is_negative_returns_positive() {
            let mut process: Process = Default::default();

            let negative = -1.0;
            let negative_term = negative.into_process(&mut process);

            assert_eq!(negative_term.tag(), Tag::Boxed);

            let unboxed_negative_term: &Term = negative_term.unbox_reference();

            assert_eq!(unboxed_negative_term.tag(), Tag::Float);

            let positive = -negative;
            let positive_term = positive.into_process(&mut process);

            assert_eq_in_process!(
                erlang::abs(negative_term, &mut process),
                Ok(positive_term),
                process
            );
        }

        #[test]
        fn with_float_that_is_positive_return_self() {
            let mut process: Process = Default::default();
            let positive_term: Term = 1.0.into_process(&mut process);

            assert_eq!(positive_term.tag(), Tag::Boxed);

            let unboxed_positive_term: &Term = positive_term.unbox_reference();

            assert_eq!(unboxed_positive_term.tag(), Tag::Float);

            assert_eq_in_process!(
                erlang::abs(positive_term, &mut process),
                Ok(positive_term),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::abs(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::abs(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::abs(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod append_element {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::append_element(atom_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::append_element(
                    Term::EMPTY_LIST,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::append_element(list_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::append_element(
                    small_integer_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::append_element(
                    big_integer_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term: Term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::append_element(float_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::append_element(local_pid_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::append_element(
                    external_pid_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_returns_tuple_with_new_element_at_end() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 1.into_process(&mut process)],
                &mut process,
            );

            assert_eq_in_process!(
                erlang::append_element(tuple_term, 2.into_process(&mut process), &mut process),
                Ok(Term::slice_to_tuple(
                    &[
                        0.into_process(&mut process),
                        1.into_process(&mut process),
                        2.into_process(&mut process)
                    ],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_at_size_return_tuples_with_new_element_at_end() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);

            assert_eq_in_process!(
                erlang::append_element(tuple_term, 1.into_process(&mut process), &mut process),
                Ok(Term::slice_to_tuple(
                    &[0.into_process(&mut process), 1.into_process(&mut process)],
                    &mut process
                )),
                process
            )
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::append_element(
                    heap_binary_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(
                erlang::append_element(subbinary_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            )
        }
    }

    mod atom_to_binary {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_without_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_name = "😈";
            let atom_term =
                Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(atom_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_atom_with_invalid_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_name = "😈";
            let atom_term =
                Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();
            let invalid_encoding_atom_term =
                Term::str_to_atom("invalid_encoding", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(atom_term, invalid_encoding_atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_atom_with_encoding_atom_returns_name_in_binary() {
            let mut process: Process = Default::default();
            let atom_name = "😈";
            let atom_term =
                Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();
            let latin1_atom_term =
                Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
            let utf8_atom_term =
                Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(atom_term, latin1_atom_term, &mut process),
                Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
                process
            );
            assert_eq_in_process!(
                erlang::atom_to_binary(atom_term, unicode_atom_term, &mut process),
                Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
                process
            );
            assert_eq_in_process!(
                erlang::atom_to_binary(atom_term, utf8_atom_term, &mut process),
                Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(Term::EMPTY_LIST, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(list_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(small_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(big_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(float_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(local_pid_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(external_pid_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 1.into_process(&mut process)],
                &mut process,
            );
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(tuple_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(heap_binary_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_binary(subbinary_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod atom_to_list {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_without_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_name = "😈🤘";
            let atom_term =
                Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(atom_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_atom_with_invalid_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_name = "😈🤘";
            let atom_term =
                Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();
            let invalid_encoding_atom_term =
                Term::str_to_atom("invalid_encoding", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(atom_term, invalid_encoding_atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_atom_with_encoding_atom_returns_chars_in_list() {
            let mut process: Process = Default::default();
            let atom_name = "😈🤘";
            let atom_term =
                Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();
            let latin1_atom_term =
                Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
            let utf8_atom_term =
                Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(atom_term, latin1_atom_term, &mut process),
                Ok(Term::cons(
                    128520.into_process(&mut process),
                    Term::cons(
                        129304.into_process(&mut process),
                        Term::EMPTY_LIST,
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
            assert_eq_in_process!(
                erlang::atom_to_list(atom_term, unicode_atom_term, &mut process),
                Ok(Term::cons(
                    128520.into_process(&mut process),
                    Term::cons(
                        129304.into_process(&mut process),
                        Term::EMPTY_LIST,
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
            assert_eq_in_process!(
                erlang::atom_to_list(atom_term, utf8_atom_term, &mut process),
                Ok(Term::cons(
                    128520.into_process(&mut process),
                    Term::cons(
                        129304.into_process(&mut process),
                        Term::EMPTY_LIST,
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(Term::EMPTY_LIST, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(list_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(small_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(big_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(float_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(local_pid_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(external_pid_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 1.into_process(&mut process)],
                &mut process,
            );
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(tuple_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(heap_binary_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::atom_to_list(subbinary_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    // binary_part/3
    mod binary_part {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(
                    atom_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::binary_part(
                    Term::EMPTY_LIST,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(
                    list_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(
                    small_integer_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(
                    big_integer_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term: Term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(
                    float_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(
                    local_pid_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(
                    external_pid_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 1.into_process(&mut process)],
                &mut process,
            );

            assert_eq_in_process!(
                erlang::binary_part(
                    tuple_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_without_integer_start_without_integer_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let start_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 0.into_process(&mut process)],
                &mut process,
            );
            let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_without_integer_start_with_integer_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_integer_start_without_integer_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_negative_start_with_valid_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let start_term = (-1isize).into_process(&mut process);
            let length_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_start_greater_than_size_with_non_negative_length_returns_bad_argument(
        ) {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let start_term = 1.into_process(&mut process);
            let length_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_start_less_than_size_with_negative_length_past_start_returns_bad_argument(
        ) {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = (-1isize).into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_start_less_than_size_with_positive_length_past_end_returns_bad_argument(
        ) {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = 2.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_zero_start_and_size_length_returns_binary() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = 1.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Ok(heap_binary_term),
                process
            );

            let returned_binary =
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process)
                    .unwrap();

            assert_eq!(returned_binary.tagged, heap_binary_term.tagged);
        }

        #[test]
        fn with_heap_binary_with_size_start_and_negative_size_length_returns_binary() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
            let start_term = 1.into_process(&mut process);
            let length_term = (-1isize).into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Ok(heap_binary_term),
                process
            );

            let returned_binary =
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process)
                    .unwrap();

            assert_eq!(returned_binary.tagged, heap_binary_term.tagged);
        }

        #[test]
        fn with_heap_binary_with_positive_start_and_negative_length_returns_subbinary() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0, 1], &mut process);
            let start_term = 1.into_process(&mut process);
            let length_term = (-1isize).into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Ok(Term::slice_to_binary(&[0], &mut process)),
                process
            );

            let returned_boxed =
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process)
                    .unwrap();

            assert_eq!(returned_boxed.tag(), Tag::Boxed);

            let returned_unboxed: &Term = returned_boxed.unbox_reference();

            assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
        }

        #[test]
        fn with_heap_binary_with_positive_start_and_positive_length_returns_subbinary() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0, 1], &mut process);
            let start_term = 1.into_process(&mut process);
            let length_term = 1.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process),
                Ok(Term::slice_to_binary(&[1], &mut process)),
                process
            );

            let returned_boxed =
                erlang::binary_part(heap_binary_term, start_term, length_term, &mut process)
                    .unwrap();

            assert_eq!(returned_boxed.tag(), Tag::Boxed);

            let returned_unboxed: &Term = returned_boxed.unbox_reference();

            assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
        }

        #[test]
        fn with_subbinary_without_integer_start_without_integer_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let start_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 0.into_process(&mut process)],
                &mut process,
            );
            let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_without_integer_start_with_integer_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_integer_start_without_integer_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_negative_start_with_valid_length_returns_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let start_term = (-1isize).into_process(&mut process);
            let length_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_start_greater_than_size_with_non_negative_length_returns_bad_argument(
        ) {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 0, 1, &mut process);
            let start_term = 1.into_process(&mut process);
            let length_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_start_less_than_size_with_negative_length_past_start_returns_bad_argument(
        ) {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = (-1isize).into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_start_less_than_size_with_positive_length_past_end_returns_bad_argument(
        ) {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 1, 1, &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = 2.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_zero_start_and_byte_count_length_returns_new_subbinary_with_bytes() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let start_term = 0.into_process(&mut process);
            let length_term = 2.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Ok(Term::subbinary(binary_term, 0, 7, 2, 0, &mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_with_byte_count_start_and_negative_byte_count_length_returns_new_subbinary_with_bytes(
        ) {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let start_term = 2.into_process(&mut process);
            let length_term = (-2isize).into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Ok(Term::subbinary(binary_term, 0, 7, 2, 0, &mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_with_positive_start_and_negative_length_returns_subbinary() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 1, 0, &mut process);
            let start_term = 1.into_process(&mut process);
            let length_term = (-1isize).into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Ok(Term::slice_to_binary(&[0b1111_1111], &mut process)),
                process
            );

            let returned_boxed =
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process).unwrap();

            assert_eq!(returned_boxed.tag(), Tag::Boxed);

            let returned_unboxed: &Term = returned_boxed.unbox_reference();

            assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
        }

        #[test]
        fn with_subbinary_with_positive_start_and_positive_length_returns_subbinary() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term: Term = process.subbinary(binary_term, 0, 7, 2, 1).into();
            let start_term = 1.into_process(&mut process);
            let length_term = 1.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process),
                Ok(Term::slice_to_binary(&[0b0101_0101], &mut process)),
                process
            );

            let returned_boxed =
                erlang::binary_part(subbinary_term, start_term, length_term, &mut process).unwrap();

            assert_eq!(returned_boxed.tag(), Tag::Boxed);

            let returned_unboxed: &Term = returned_boxed.unbox_reference();

            assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
        }
    }

    mod binary_to_atom {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(atom_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(Term::EMPTY_LIST, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(list_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(small_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(big_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(float_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(local_pid_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(external_pid_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 1.into_process(&mut process)],
                &mut process,
            );
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(tuple_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_without_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_atom(
                    heap_binary_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_invalid_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let invalid_encoding_term =
                Term::str_to_atom("invalid_encoding", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(heap_binary_term, invalid_encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_valid_encoding_returns_atom() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary("😈".as_bytes(), &mut process);
            let latin1_atom_term =
                Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
            let utf8_atom_term =
                Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();
            let atom_term = Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(heap_binary_term, latin1_atom_term, &mut process),
                Ok(atom_term),
                &mut process
            );
            assert_eq_in_process!(
                erlang::binary_to_atom(heap_binary_term, unicode_atom_term, &mut process),
                Ok(atom_term),
                &mut process
            );
            assert_eq_in_process!(
                erlang::binary_to_atom(heap_binary_term, utf8_atom_term, &mut process),
                Ok(atom_term),
                &mut process
            );
        }

        #[test]
        fn with_subbinary_with_bit_count_returns_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(subbinary_term, unicode_atom_term, &mut process),
                Err(bad_argument!()),
                &mut process
            )
        }

        #[test]
        fn with_subbinary_without_bit_count_returns_atom_with_bytes() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary("😈🤘".as_bytes(), &mut process);
            let subbinary_term = Term::subbinary(binary_term, 4, 0, 4, 0, &mut process);
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_atom(subbinary_term, unicode_atom_term, &mut process),
                Term::str_to_atom("🤘", Existence::DoNotCare, &mut process),
                &mut process
            )
        }
    }

    mod binary_to_existing_atom {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(atom_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(Term::EMPTY_LIST, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(list_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(small_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(big_integer_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(float_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 1.into_process(&mut process)],
                &mut process,
            );
            let encoding_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(tuple_term, encoding_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_without_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(
                    heap_binary_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_invalid_encoding_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let invalid_encoding_term =
                Term::str_to_atom("invalid_encoding", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(
                    heap_binary_term,
                    invalid_encoding_term,
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_valid_encoding_without_existing_atom_returns_atom() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary("😈".as_bytes(), &mut process);
            let latin1_atom_term =
                Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
            let utf8_atom_term =
                Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(heap_binary_term, latin1_atom_term, &mut process),
                Err(bad_argument!()),
                &mut process
            );
            assert_eq_in_process!(
                erlang::binary_to_existing_atom(heap_binary_term, unicode_atom_term, &mut process),
                Err(bad_argument!()),
                &mut process
            );
            assert_eq_in_process!(
                erlang::binary_to_existing_atom(heap_binary_term, utf8_atom_term, &mut process),
                Err(bad_argument!()),
                &mut process
            );
        }

        #[test]
        fn with_heap_binary_with_valid_encoding_with_existing_atom_returns_atom() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary("😈".as_bytes(), &mut process);
            let latin1_atom_term =
                Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
            let utf8_atom_term =
                Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();
            let atom_term = Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(heap_binary_term, latin1_atom_term, &mut process),
                Ok(atom_term),
                &mut process
            );
            assert_eq_in_process!(
                erlang::binary_to_existing_atom(heap_binary_term, unicode_atom_term, &mut process),
                Ok(atom_term),
                &mut process
            );
            assert_eq_in_process!(
                erlang::binary_to_existing_atom(heap_binary_term, utf8_atom_term, &mut process),
                Ok(atom_term),
                &mut process
            );
        }

        #[test]
        fn with_subbinary_with_bit_count_returns_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(subbinary_term, unicode_atom_term, &mut process),
                Err(bad_argument!()),
                &mut process
            )
        }

        #[test]
        fn with_subbinary_without_bit_count_without_existing_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary("😈🤘".as_bytes(), &mut process);
            let subbinary_term = Term::subbinary(binary_term, 4, 0, 4, 0, &mut process);
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(subbinary_term, unicode_atom_term, &mut process),
                Err(bad_argument!()),
                &mut process
            )
        }

        #[test]
        fn with_subbinary_without_bit_count_with_existing_atom_returns_atom_with_bytes() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary("😈🤘".as_bytes(), &mut process);
            let subbinary_term = Term::subbinary(binary_term, 4, 0, 4, 0, &mut process);
            let unicode_atom_term =
                Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
            let atom_term = Term::str_to_atom("🤘", Existence::DoNotCare, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_existing_atom(subbinary_term, unicode_atom_term, &mut process),
                atom_term,
                &mut process
            )
        }
    }

    mod binary_to_float {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term =
                Term::str_to_atom("😈🤘", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_float(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_returns_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::binary_to_float(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term =
                <BigInt as Num>::from_str_radix("18446744073709551616", 10)
                    .unwrap()
                    .into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_float(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_float(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_integer_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary("1".as_bytes(), &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            )
        }

        #[test]
        fn with_heap_binary_with_min_f64_returns_float() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("-179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0".as_bytes(), &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(heap_binary_term, &mut process),
                Ok((-179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0_f64).into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_max_f64_returns_float() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0".as_bytes(), &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(heap_binary_term, &mut process),
                Ok(179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0_f64.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_less_than_min_f64_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("-1797693134862315700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0".as_bytes(), &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_greater_than_max_f64_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("17976931348623157000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0".as_bytes(), &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_min_f64_returns_float() {
            let mut process: Process = Default::default();
            // <<1::1, "-179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    150,
                    152,
                    155,
                    156,
                    155,
                    155,
                    28,
                    153,
                    152,
                    153,
                    154,
                    28,
                    27,
                    25,
                    25,
                    152,
                    154,
                    155,
                    152,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    23,
                    24,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 312, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(subbinary_term, &mut process),
                Ok((-179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0_f64).into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_with_max_f64_returns_float() {
            let mut process: Process = Default::default();
            // <<1::1, "179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    152,
                    155,
                    156,
                    155,
                    155,
                    28,
                    153,
                    152,
                    153,
                    154,
                    28,
                    27,
                    25,
                    25,
                    152,
                    154,
                    155,
                    152,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    23,
                    24,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 311, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(subbinary_term, &mut process),
                Ok(179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0_f64.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_with_less_than_min_f64_returns_bag_argument() {
            let mut process: Process = Default::default();
            // <<1::1, "-1797693134862315700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    150,
                    152,
                    155,
                    156,
                    155,
                    155,
                    28,
                    153,
                    152,
                    153,
                    154,
                    28,
                    27,
                    25,
                    25,
                    152,
                    154,
                    155,
                    152,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    23,
                    24,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 313, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_greater_than_max_f64_returns_bad_argument() {
            let mut process: Process = Default::default();
            // <<1::1, "576460752303423488">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    150,
                    152,
                    155,
                    156,
                    155,
                    155,
                    28,
                    153,
                    152,
                    153,
                    154,
                    28,
                    27,
                    25,
                    25,
                    152,
                    154,
                    155,
                    152,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    23,
                    24,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 313, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_float(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod binary_to_integer {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term =
                Term::str_to_atom("😈🤘", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_integer(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_returns_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::binary_to_integer(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_integer(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_integer(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term =
                <BigInt as Num>::from_str_radix("18446744073709551616", 10)
                    .unwrap()
                    .into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_integer(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_integer(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_integer(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_min_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("-576460752303423488".as_bytes(), &mut process);

            let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_heap_binary_with_max_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("576460752303423487".as_bytes(), &mut process);

            let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_heap_binary_with_less_than_min_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("-576460752303423489".as_bytes(), &mut process);

            let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("-576460752303423489", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }

        #[test]
        fn with_heap_binary_with_greater_than_max_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("576460752303423488".as_bytes(), &mut process);

            let integer_result = erlang::binary_to_integer(heap_binary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }

        #[test]
        fn with_heap_binary_with_non_decimal_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary("FF".as_bytes(), &mut process);

            assert_eq_in_process!(
                erlang::binary_to_integer(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_min_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            // <<1::1, "-576460752303423488">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    150,
                    154,
                    155,
                    155,
                    26,
                    27,
                    24,
                    27,
                    154,
                    153,
                    25,
                    152,
                    25,
                    154,
                    25,
                    25,
                    154,
                    28,
                    28,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 19, 0, &mut process);

            let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_subbinary_with_max_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            // <<1::1, "576460752303423487">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    154,
                    155,
                    155,
                    26,
                    27,
                    24,
                    27,
                    154,
                    153,
                    25,
                    152,
                    25,
                    154,
                    25,
                    25,
                    154,
                    28,
                    27,
                    0b1000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 18, 0, &mut process);

            let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_subbinary_with_less_than_min_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            // <<1::1, "-576460752303423489">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    150,
                    154,
                    155,
                    155,
                    26,
                    27,
                    24,
                    27,
                    154,
                    153,
                    25,
                    152,
                    25,
                    154,
                    25,
                    25,
                    154,
                    28,
                    28,
                    0b1000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 19, 0, &mut process);

            let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("-576460752303423489", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }

        #[test]
        fn with_subbinary_with_greater_than_max_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            // <<1::1, "576460752303423488">>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    154,
                    155,
                    155,
                    26,
                    27,
                    24,
                    27,
                    154,
                    153,
                    25,
                    152,
                    25,
                    154,
                    25,
                    25,
                    154,
                    28,
                    28,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 18, 0, &mut process);

            let integer_result = erlang::binary_to_integer(subbinary_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }

        #[test]
        fn with_subbinary_with_non_decimal_returns_bad_argument() {
            let mut process: Process = Default::default();
            // <<1:1, "FF>>
            let heap_binary_term = Term::slice_to_binary(&[163, 35, 0b000_0000], &mut process);
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 2, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_integer(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod binary_in_base_to_integer {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term =
                Term::str_to_atom("😈🤘", Existence::DoNotCare, &mut process).unwrap();
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(atom_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_returns_bad_argument() {
            let mut process: Process = Default::default();
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(Term::EMPTY_LIST, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(list_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(small_integer_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term =
                <BigInt as Num>::from_str_radix("18446744073709551616", 10)
                    .unwrap()
                    .into_process(&mut process);
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(big_integer_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(float_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(local_pid_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(external_pid_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);
            let base_term: Term = 16.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_in_base_to_integer(tuple_term, base_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_min_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("-800000000000000".as_bytes(), &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(heap_binary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_heap_binary_with_max_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("7FFFFFFFFFFFFFF".as_bytes(), &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(heap_binary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_heap_binary_with_less_than_min_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("-800000000000001".as_bytes(), &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(heap_binary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("-576460752303423489", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }

        #[test]
        fn with_heap_binary_with_greater_than_max_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term =
                Term::slice_to_binary("800000000000000".as_bytes(), &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(heap_binary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }

        #[test]
        fn with_subbinary_with_min_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            // <<1::1, Integer.to_string(-576460752303423488, 16) :: binary>>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    150,
                    156,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 16, 0, &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(subbinary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_subbinary_with_max_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            // <<1::1, Integer.to_string(576460752303423487, 16) :: binary>>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    155,
                    163,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    35,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 15, 0, &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(subbinary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );
            assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
        }

        #[test]
        fn with_subbinary_with_less_than_min_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            // <<1::1, Integer.to_string(-576460752303423489, 16) :: binary>>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    150,
                    156,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    0b1000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 16, 0, &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(subbinary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(
                    <BigInt as Num>::from_str_radix("-5764_60_752_303_423_489", 10)
                        .unwrap()
                        .into_process(&mut process)
                ),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }

        #[test]
        fn with_subbinary_with_greater_than_max_small_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            // <<1::1, Integer.to_string(576460752303423488, 16) :: binary>>
            let heap_binary_term = Term::slice_to_binary(
                &[
                    156,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    0b0000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 15, 0, &mut process);
            let base_term: Term = 16.into_process(&mut process);

            let integer_result =
                erlang::binary_in_base_to_integer(subbinary_term, base_term, &mut process);

            assert_eq_in_process!(
                integer_result,
                Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
                    .unwrap()
                    .into_process(&mut process)),
                process
            );

            let integer = integer_result.unwrap();

            assert_eq!(integer.tag(), Tag::Boxed);

            let unboxed: &Term = integer.unbox_reference();

            assert_eq!(unboxed.tag(), Tag::BigInteger);
        }
    }

    mod binary_to_list {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_list(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::binary_to_list(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_list(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_list(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_returns_list_of_bytes() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(heap_binary_term, &mut process),
                Ok(Term::cons(
                    0.into_process(&mut process),
                    Term::cons(
                        1.into_process(&mut process),
                        Term::cons(2.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_without_bit_count_returns_list_of_bytes() {
            let mut process: Process = Default::default();
            // <<1::1, 0, 1, 2>>
            let binary_term = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 1, 3, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(subbinary_term, &mut process),
                Ok(Term::cons(
                    0.into_process(&mut process),
                    Term::cons(
                        1.into_process(&mut process),
                        Term::cons(2.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_with_bit_count_returns_bad_argument() {
            let mut process: Process = Default::default();
            // <<1::1, 0, 1, 2>>
            let binary_term = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 0, 3, 1, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_list(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod binary_byte_range_to_list {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    atom_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    Term::EMPTY_LIST,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    list_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    small_integer_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    big_integer_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    float_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    local_pid_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    external_pid_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    tuple_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_start_less_than_stop_returns_list_of_bytes() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    heap_binary_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Ok(Term::cons(
                    1.into_process(&mut process),
                    Term::EMPTY_LIST,
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_start_equal_to_stop_returns_list_of_single_byte() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    heap_binary_term,
                    2.into_process(&mut process),
                    2.into_process(&mut process),
                    &mut process
                ),
                Ok(Term::cons(
                    1.into_process(&mut process),
                    Term::EMPTY_LIST,
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_heap_binary_with_start_greater_than_stop_returns_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    heap_binary_term,
                    3.into_process(&mut process),
                    2.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_with_start_less_than_stop_returns_list_of_bytes() {
            let mut process: Process = Default::default();
            // <<1::1, 0, 1, 2>>
            let binary_term = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 1, 3, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    subbinary_term,
                    2.into_process(&mut process),
                    3.into_process(&mut process),
                    &mut process
                ),
                Ok(Term::cons(
                    1.into_process(&mut process),
                    Term::EMPTY_LIST,
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_with_start_equal_to_stop_returns_list_of_single_byte() {
            let mut process: Process = Default::default();
            // <<1::1, 0, 1, 2>>
            let binary_term = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 1, 3, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    subbinary_term,
                    2.into_process(&mut process),
                    2.into_process(&mut process),
                    &mut process
                ),
                Ok(Term::cons(
                    1.into_process(&mut process),
                    Term::EMPTY_LIST,
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_with_start_greater_than_stop_returns_bad_argument() {
            let mut process: Process = Default::default();
            // <<1::1, 0, 1, 2>>
            let binary_term = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 1, 3, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_byte_range_to_list(
                    subbinary_term,
                    3.into_process(&mut process),
                    2.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod binary_to_term {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_returns_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_term(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::binary_to_term(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_term(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::binary_to_term(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_atom_returns_atom() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(:atom)
            let heap_binary_term =
                Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Term::str_to_atom("atom", Existence::DoNotCare, &mut process),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_empty_list_returns_empty_list() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary([])
            let heap_binary_term = Term::slice_to_binary(&[131, 106], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(Term::EMPTY_LIST),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_list_returns_list() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary([:zero, 1])
            let heap_binary_term = Term::slice_to_binary(
                &[
                    131, 108, 0, 0, 0, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1, 106,
                ],
                &mut process,
            );

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(Term::cons(
                    Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
                    Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(0)
            let heap_binary_term = Term::slice_to_binary(&[131, 97, 0], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(0.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_integer_returns_integer() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(-2147483648)
            let heap_binary_term = Term::slice_to_binary(&[131, 98, 128, 0, 0, 0], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok((-2147483648_isize).into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_new_float_returns_float() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(1.0)
            let heap_binary_term =
                Term::slice_to_binary(&[131, 70, 63, 240, 0, 0, 0, 0, 0, 0], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(1.0.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_small_tuple_returns_tuple() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary({:zero, 1})
            let heap_binary_term = Term::slice_to_binary(
                &[131, 104, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1],
                &mut process,
            );

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(Term::slice_to_tuple(
                    &[
                        Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
                        1.into_process(&mut process)
                    ],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_byte_list_returns_list() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary([?0, ?1])
            let heap_binary_term = Term::slice_to_binary(&[131, 107, 0, 2, 48, 49], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(Term::cons(
                    48.into_process(&mut process),
                    Term::cons(
                        49.into_process(&mut process),
                        Term::EMPTY_LIST,
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_binary_returns_binary() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(<<0, 1>>)
            let heap_binary_term =
                Term::slice_to_binary(&[131, 109, 0, 0, 0, 2, 0, 1], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(Term::slice_to_binary(&[0, 1], &mut process)),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_small_big_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(4294967295)
            let heap_binary_term =
                Term::slice_to_binary(&[131, 110, 4, 0, 255, 255, 255, 255], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(4294967295_usize.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_heap_binary_encoding_bit_string_returns_subbinary() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(<<1, 2::3>>)
            let heap_binary_term =
                Term::slice_to_binary(&[131, 77, 0, 0, 0, 2, 3, 1, 64], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(Term::subbinary(
                    Term::slice_to_binary(&[1, 0b010_00000], &mut process),
                    0,
                    0,
                    1,
                    3,
                    &mut process
                )),
                process,
            );
        }

        #[test]
        fn with_heap_binary_encoding_small_atom_utf8_returns_atom() {
            let mut process: Process = Default::default();
            // :erlang.term_to_binary(:"😈")
            let heap_binary_term =
                Term::slice_to_binary(&[131, 119, 4, 240, 159, 152, 136], &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(heap_binary_term, &mut process),
                Ok(Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap()),
                process,
            );
        }

        #[test]
        fn with_subbinary_encoding_atom_returns_atom() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(:atom) :: binary>>
            let original_term = Term::slice_to_binary(
                &[193, 178, 0, 2, 48, 186, 55, 182, 0b1000_0000],
                &mut process,
            );
            let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Term::str_to_atom("atom", Existence::DoNotCare, &mut process),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_empty_list_returns_empty_list() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary([]) :: binary>>
            let original_term = Term::slice_to_binary(&[193, 181, 0b0000_0000], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 2, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(Term::EMPTY_LIST),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_list_returns_list() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary([:zero, 1]) :: binary>>
            let original_term = Term::slice_to_binary(
                &[
                    193, 182, 0, 0, 0, 1, 50, 0, 2, 61, 50, 185, 55, 176, 128, 181, 0,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(original_term, 0, 1, 16, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(Term::cons(
                    Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
                    Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_small_integer_returns_small_integer() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(0) :: binary>>
            let original_term = Term::slice_to_binary(&[193, 176, 128, 0], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 3, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(0.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_integer_returns_integer() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(-2147483648) :: binary>>
            let original_term = Term::slice_to_binary(&[193, 177, 64, 0, 0, 0, 0], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 6, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok((-2147483648_isize).into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_new_float_returns_float() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(1.0) :: binary>>
            let original_term =
                Term::slice_to_binary(&[193, 163, 31, 248, 0, 0, 0, 0, 0, 0, 0], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 10, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(1.0.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_small_tuple_returns_tuple() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary({:zero, 1}) :: binary>>
            let original_term = Term::slice_to_binary(
                &[
                    193,
                    180,
                    1,
                    50,
                    0,
                    2,
                    61,
                    50,
                    185,
                    55,
                    176,
                    128,
                    0b1000_0000,
                ],
                &mut process,
            );
            let subbinary_term = Term::subbinary(original_term, 0, 1, 12, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(Term::slice_to_tuple(
                    &[
                        Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
                        1.into_process(&mut process)
                    ],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_byte_list_returns_list() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary([?0, ?1]) :: binary>>
            let original_term =
                Term::slice_to_binary(&[193, 181, 128, 1, 24, 24, 0b1000_0000], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 6, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(Term::cons(
                    48.into_process(&mut process),
                    Term::cons(
                        49.into_process(&mut process),
                        Term::EMPTY_LIST,
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_binary_returns_binary() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(<<0, 1>>) :: binary>>
            let original_term =
                Term::slice_to_binary(&[193, 182, 128, 0, 0, 1, 0, 0, 0b1000_0000], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(Term::slice_to_binary(&[0, 1], &mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_small_big_integer_returns_big_integer() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(4294967295) :: binary>>
            let original_term = Term::slice_to_binary(
                &[193, 183, 2, 0, 127, 255, 255, 255, 0b1000_0000],
                &mut process,
            );
            let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(4294967295_usize.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_encoding_bit_string_returns_subbinary() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(<<1, 2::3>>) :: binary>>
            let original_term =
                Term::slice_to_binary(&[193, 166, 128, 0, 0, 1, 1, 128, 160, 0], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 9, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(Term::subbinary(
                    Term::slice_to_binary(&[1, 0b010_00000], &mut process),
                    0,
                    0,
                    1,
                    3,
                    &mut process
                )),
                process,
            );
        }

        #[test]
        fn with_subbinary_encoding_small_atom_utf8_returns_atom() {
            let mut process: Process = Default::default();
            // <<1::1, :erlang.term_to_binary(:"😈") :: binary>>
            let original_term =
                Term::slice_to_binary(&[193, 187, 130, 120, 79, 204, 68, 0], &mut process);
            let subbinary_term = Term::subbinary(original_term, 0, 1, 7, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_to_term(subbinary_term, &mut process),
                Ok(Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap()),
                process,
            );
        }
    }

    mod binary_options_to_term {
        use super::*;

        mod with_safe {
            use super::*;

            #[test]
            fn with_binary_encoding_atom_that_does_not_exist_returns_bad_argument() {
                let mut process: Process = Default::default();
                // :erlang.term_to_binary(:atom)
                let binary_term =
                    Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &mut process);
                let options = Term::cons(
                    Term::str_to_atom("safe", Existence::DoNotCare, &mut process).unwrap(),
                    Term::EMPTY_LIST,
                    &mut process,
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, options, &mut process),
                    Err(bad_argument!()),
                    process
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, Term::EMPTY_LIST, &mut process),
                    Ok(Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap()),
                    process
                );
            }

            #[test]
            fn with_binary_encoding_list_containing_atom_that_does_not_exist_returns_bad_argument()
            {
                let mut process: Process = Default::default();
                // :erlang.term_to_binary([:atom])
                let binary_term = Term::slice_to_binary(
                    &[131, 108, 0, 0, 0, 1, 100, 0, 4, 97, 116, 111, 109, 106],
                    &mut process,
                );
                let options = Term::cons(
                    Term::str_to_atom("safe", Existence::DoNotCare, &mut process).unwrap(),
                    Term::EMPTY_LIST,
                    &mut process,
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, options, &mut process),
                    Err(bad_argument!()),
                    process
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, Term::EMPTY_LIST, &mut process),
                    Ok(Term::cons(
                        Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap(),
                        Term::EMPTY_LIST,
                        &mut process
                    )),
                    process
                );
            }

            #[test]
            fn with_binary_encoding_small_tuple_containing_atom_that_does_not_exist_returns_bad_argument(
            ) {
                let mut process: Process = Default::default();
                // :erlang.term_to_binary({:atom})
                let binary_term = Term::slice_to_binary(
                    &[131, 104, 1, 100, 0, 4, 97, 116, 111, 109],
                    &mut process,
                );
                let options = Term::cons(
                    Term::str_to_atom("safe", Existence::DoNotCare, &mut process).unwrap(),
                    Term::EMPTY_LIST,
                    &mut process,
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, options, &mut process),
                    Err(bad_argument!()),
                    process
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, Term::EMPTY_LIST, &mut process),
                    Ok(Term::slice_to_tuple(
                        &[Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap()],
                        &mut process
                    )),
                    process
                );
            }

            #[test]
            fn with_binary_encoding_small_atom_utf8_that_does_not_exist_returns_bad_argument() {
                let mut process: Process = Default::default();
                // :erlang.term_to_binary(:"😈")
                let binary_term =
                    Term::slice_to_binary(&[131, 119, 4, 240, 159, 152, 136], &mut process);
                let options = Term::cons(
                    Term::str_to_atom("safe", Existence::DoNotCare, &mut process).unwrap(),
                    Term::EMPTY_LIST,
                    &mut process,
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, options, &mut process),
                    Err(bad_argument!()),
                    process
                );

                assert_eq_in_process!(
                    erlang::binary_options_to_term(binary_term, Term::EMPTY_LIST, &mut process),
                    Ok(Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap()),
                    process
                );
            }
        }

        #[test]
        fn with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term() {
            let mut process: Process = Default::default();
            // <<131,100,0,5,"hello","world">>
            let binary_term = Term::slice_to_binary(
                &[
                    131, 100, 0, 5, 104, 101, 108, 108, 111, 119, 111, 114, 108, 100,
                ],
                &mut process,
            );
            let options = Term::cons(
                Term::str_to_atom("used", Existence::DoNotCare, &mut process).unwrap(),
                Term::EMPTY_LIST,
                &mut process,
            );

            let term = Term::str_to_atom("hello", Existence::DoNotCare, &mut process).unwrap();
            let result = erlang::binary_options_to_term(binary_term, options, &mut process);

            assert_eq_in_process!(
                result,
                Ok(Term::slice_to_tuple(
                    &[term, 9.into_process(&mut process)],
                    &mut process
                )),
                process
            );

            // Using only `used` portion of binary returns the same result

            let tuple = result.unwrap();
            let used_term = erlang::element(tuple, 1.into_process(&mut process)).unwrap();
            let used: usize = used_term.try_into().unwrap();

            let prefix_term = Term::subbinary(binary_term, 0, 0, used, 0, &mut process);

            assert_eq_in_process!(
                erlang::binary_options_to_term(prefix_term, options, &mut process),
                Ok(tuple),
                process
            );

            // Without used returns only term

            assert_eq_in_process!(
                erlang::binary_options_to_term(binary_term, Term::EMPTY_LIST, &mut process),
                Ok(term),
                process
            );
        }
    }

    mod bit_size {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::bit_size(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::bit_size(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::bit_size(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::bit_size(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::bit_size(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::bit_size(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::bit_size(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::bit_size(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::bit_size(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_eight_times_byte_count() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[1], &mut process);

            assert_eq_in_process!(
                erlang::bit_size(heap_binary_term, &mut process),
                Ok(8.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_is_eight_times_byte_count_plus_bit_count() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0, 1, 0b010], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 0, 2, 3, &mut process);

            assert_eq_in_process!(
                erlang::bit_size(subbinary_term, &mut process),
                Ok(19.into_process(&mut process)),
                process
            );
        }
    }

    mod bitstring_to_list {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::bitstring_to_list(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::bitstring_to_list(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::bitstring_to_list(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::bitstring_to_list(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_returns_list_of_integer() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0], &mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(heap_binary_term, &mut process),
                Ok(Term::cons(
                    0.into_process(&mut process),
                    Term::EMPTY_LIST,
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_without_bit_count_returns_list_of_integer() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0, 1, 0b010], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(subbinary_term, &mut process),
                Ok(Term::cons(
                    1.into_process(&mut process),
                    Term::EMPTY_LIST,
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_subbinary_with_bit_count_returns_list_of_integer_with_bitstring_for_bit_count() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0, 1, 0b010], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 0, 2, 3, &mut process);

            assert_eq_in_process!(
                erlang::bitstring_to_list(subbinary_term, &mut process),
                Ok(Term::cons(
                    0.into_process(&mut process),
                    Term::cons(
                        1.into_process(&mut process),
                        Term::subbinary(
                            Term::slice_to_binary(&[0, 1, 2], &mut process),
                            2,
                            0,
                            0,
                            3,
                            &mut process
                        ),
                        &mut process
                    ),
                    &mut process
                )),
                process
            );
        }
    }

    mod byte_size {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::byte_size(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::byte_size(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::byte_size(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::byte_size(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::byte_size(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::byte_size(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::byte_size(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::byte_size(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::byte_size(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_byte_count() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[1], &mut process);

            assert_eq_in_process!(
                erlang::byte_size(heap_binary_term, &mut process),
                Ok(1.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_without_bit_count_is_byte_count() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0, 1], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &mut process);

            assert_eq_in_process!(
                erlang::byte_size(subbinary_term, &mut process),
                Ok(1.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_subbinary_with_bit_count_is_byte_count_plus_one() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0, 1, 0b0100_0000], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 3, &mut process);

            assert_eq_in_process!(
                erlang::byte_size(subbinary_term, &mut process),
                Ok(2.into_process(&mut process)),
                process
            );
        }
    }

    mod ceil {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::ceil(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::ceil(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::ceil(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_returns_same() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            let result = erlang::ceil(small_integer_term, &mut process);

            assert_eq_in_process!(result, Ok(small_integer_term), process);
            assert_eq!(result.unwrap().tagged, small_integer_term.tagged);
        }

        #[test]
        fn with_big_integer_returns_same() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            let result = erlang::ceil(big_integer_term, &mut process);

            assert_eq_in_process!(result, Ok(big_integer_term), process);
            assert_eq!(result.unwrap().tagged, big_integer_term.tagged);
        }

        #[test]
        fn with_float_without_fraction_returns_integer() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::ceil(float_term, &mut process),
                Ok(1.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_float_with_fraction_rounds_up_to_next_integer() {
            let mut process: Process = Default::default();
            let float_term = (-1.1).into_process(&mut process);

            let result = erlang::ceil(float_term, &mut process);

            assert_eq_in_process!(result, Ok((-1).into_process(&mut process)), process);
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::ceil(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::ceil(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::ceil(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[1], &mut process);

            assert_eq_in_process!(
                erlang::ceil(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0, 1], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &mut process);

            assert_eq_in_process!(
                erlang::ceil(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod convert_time_units {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(atom_term, from_unit_term, to_unit_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(
                    Term::EMPTY_LIST,
                    from_unit_term,
                    to_unit_term,
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(list_term, from_unit_term, to_unit_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        mod with_small_integer {
            use super::*;

            use num_traits::Num;

            #[test]
            fn without_valid_units_returns_bad_argument() {
                let mut process: Process = Default::default();
                let small_integer_term: Term = 0.into_process(&mut process);
                let valid_unit_term =
                    Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
                let invalid_unit_term =
                    Term::str_to_atom("s", Existence::DoNotCare, &mut process).unwrap();

                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        valid_unit_term,
                        invalid_unit_term,
                        &mut process,
                    ),
                    Err(bad_argument!()),
                    process
                );

                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        invalid_unit_term,
                        valid_unit_term,
                        &mut process,
                    ),
                    Err(bad_argument!()),
                    process
                );
            }

            #[test]
            fn with_valid_units_returns_converted_value() {
                let mut process: Process = Default::default();
                let small_integer_term: Term = 1_000_000_000.into_process(&mut process);

                assert_eq!(small_integer_term.tag(), Tag::SmallInteger);

                // (Hertz, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        2_usize.into_process(&mut process),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(2_000_000_000.into_process(&mut process)),
                    process
                );
                // (Hertz, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(500_000_000.into_process(&mut process)),
                    process
                );
                // (Hertz, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Hertz, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("500_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Hertz, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Hertz, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Hertz, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("500_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );

                // (Second, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Second, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000_usize.into_process(&mut process)),
                    process
                );
                // (Second, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Second, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Second, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Second, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Second, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );

                // (Millisecond, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(5_000_000.into_process(&mut process)),
                    process
                );
                // (Millisecond, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000.into_process(&mut process)),
                    process
                );
                // (Millisecond, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Millisecond, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Millisecond, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Millisecond, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Millisecond, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );

                // (Microsecond, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(5_000_000.into_process(&mut process)),
                    process
                );
                // (Microsecond, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000.into_process(&mut process)),
                    process
                );
                // (Microsecond, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Microsecond, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Microsecond, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Microsecond, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Microsecond, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );

                // (Nanosecond, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(5.into_process(&mut process)),
                    process
                );
                // (Nanosecond, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1.into_process(&mut process)),
                    process
                );
                // (Nanosecond, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000.into_process(&mut process)),
                    process
                );
                // (Nanosecond, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000.into_process(&mut process)),
                    process
                );
                // (Nanosecond, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Nanosecond, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000.into_process(&mut process)),
                    process
                );
                // (Nanosecond, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000.into_process(&mut process)),
                    process
                );

                // (Native, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(5_000_000.into_process(&mut process)),
                    process
                );
                // (Native, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000.into_process(&mut process)),
                    process
                );
                // (Native, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Native, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Native, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );

                // (Native, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(5_000_000.into_process(&mut process)),
                    process
                );
                // (Native, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000.into_process(&mut process)),
                    process
                );
                // (Native, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Native, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Native, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        small_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
            }
        }

        mod with_big_integer {
            use super::*;

            use num_traits::Num;

            #[test]
            fn without_valid_units_returns_bad_argument() {
                let mut process: Process = Default::default();
                let big_integer_term: Term =
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process);

                assert_eq!(big_integer_term.tag(), Tag::Boxed);

                let valid_unit_term =
                    Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
                let invalid_unit_term =
                    Term::str_to_atom("s", Existence::DoNotCare, &mut process).unwrap();

                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        valid_unit_term,
                        invalid_unit_term,
                        &mut process,
                    ),
                    Err(bad_argument!()),
                    process
                );

                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        invalid_unit_term,
                        valid_unit_term,
                        &mut process,
                    ),
                    Err(bad_argument!()),
                    process
                );
            }

            #[test]
            fn with_valid_units_returns_converted_value() {
                let mut process: Process = Default::default();
                let big_integer_term: Term =
                    <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process);

                assert_eq!(big_integer_term.tag(), Tag::Boxed);

                // (Hertz, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        2_usize.into_process(&mut process),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("2_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Hertz, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Hertz, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Hertz, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Hertz, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Hertz, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Hertz, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        2_usize.into_process(&mut process),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );

                // (Second, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("5_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Second, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Second, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Second, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Second, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix(
                        "1_000_000_000_000_000_000_000_000_000",
                        10
                    )
                    .unwrap()
                    .into_process(&mut process)),
                    process
                );
                // (Second, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Second, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );

                // (Millisecond, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Millisecond, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Millisecond, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Millisecond, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Millisecond, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Millisecond, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Millisecond, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );

                // (Microsecond, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Microsecond, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Microsecond, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Microsecond, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Microsecond, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Microsecond, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Microsecond, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );

                // (Nanosecond, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("5_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Nanosecond, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(1_000_000_000.into_process(&mut process)),
                    process
                );
                // (Nanosecond, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Nanosecond, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Nanosecond, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Nanosecond, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Nanosecond, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );

                // (Native, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );

                // (Native, Hertz)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        5_usize.into_process(&mut process),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Second)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("second", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                        .unwrap()
                        .into_process(&mut process)),
                    process
                );
                // (Native, Millisecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("millisecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, Microsecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("microsecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, Nanosecond)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("nanosecond", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, Native)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
                // (Native, PerformanceCounter)
                assert_eq_in_process!(
                    erlang::convert_time_unit(
                        big_integer_term,
                        Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap(),
                        Term::str_to_atom("perf_counter", Existence::DoNotCare, &mut process)
                            .unwrap(),
                        &mut process
                    ),
                    Ok(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&mut process)
                    ),
                    process
                );
            }
        }

        #[test]
        fn with_float_returns_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(float_term, from_unit_term, to_unit_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(
                    local_pid_term,
                    from_unit_term,
                    to_unit_term,
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(
                    external_pid_term,
                    from_unit_term,
                    to_unit_term,
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(tuple_term, from_unit_term, to_unit_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[1], &mut process);
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(
                    heap_binary_term,
                    from_unit_term,
                    to_unit_term,
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term = Term::slice_to_binary(&[0, 1], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &mut process);
            let from_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();
            let to_unit_term =
                Term::str_to_atom("native", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::convert_time_unit(
                    subbinary_term,
                    from_unit_term,
                    to_unit_term,
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod delete_element {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::delete_element(atom_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::delete_element(
                    Term::EMPTY_LIST,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::delete_element(list_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::delete_element(
                    small_integer_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::delete_element(
                    big_integer_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::delete_element(float_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::delete_element(local_pid_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::delete_element(
                    external_pid_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_without_small_integer_index_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[
                    0.into_process(&mut process),
                    1.into_process(&mut process),
                    2.into_process(&mut process),
                ],
                &mut process,
            );
            let index = 1usize;
            let invalid_index_term = Term::arity(index);

            assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                erlang::delete_element(tuple_term, invalid_index_term, &mut process),
                Err(bad_argument!()),
                process
            );

            let valid_index_term: Term = index.into_process(&mut process);

            assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                erlang::delete_element(tuple_term, valid_index_term, &mut process),
                Ok(Term::slice_to_tuple(
                    &[0.into_process(&mut process), 2.into_process(&mut process)],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_tuple_without_index_in_range_is_bad_argument() {
            let mut process: Process = Default::default();
            let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::delete_element(
                    empty_tuple_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_in_range_returns_tuple_without_element() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[
                    0.into_process(&mut process),
                    1.into_process(&mut process),
                    2.into_process(&mut process),
                ],
                &mut process,
            );

            assert_eq_in_process!(
                erlang::delete_element(tuple_term, 1.into_process(&mut process), &mut process),
                Ok(Term::slice_to_tuple(
                    &[0.into_process(&mut process), 2.into_process(&mut process)],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::delete_element(
                    heap_binary_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(
                erlang::delete_element(subbinary_term, 0.into_process(&mut process), &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod element {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::element(atom_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::element(Term::EMPTY_LIST, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::element(list_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term: Term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::element(small_integer_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::element(big_integer_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::element(float_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::element(local_pid_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::element(external_pid_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_without_small_integer_index_is_bad_argument() {
            let mut process: Process = Default::default();
            let element_term = 1.into_process(&mut process);
            let tuple_term = Term::slice_to_tuple(&[element_term], &mut process);
            let index = 0usize;
            let invalid_index_term = Term::arity(index);

            assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                erlang::element(tuple_term, invalid_index_term),
                Err(bad_argument!()),
                process
            );

            let valid_index_term: Term = index.into_process(&mut process);

            assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                erlang::element(tuple_term, valid_index_term),
                Ok(element_term),
                process
            );
        }

        #[test]
        fn with_tuple_without_index_in_range_is_bad_argument() {
            let mut process: Process = Default::default();
            let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::element(empty_tuple_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_in_range_is_element() {
            let mut process: Process = Default::default();
            let element_term = 1.into_process(&mut process);
            let tuple_term = Term::slice_to_tuple(&[element_term], &mut process);

            assert_eq_in_process!(
                erlang::element(tuple_term, 0.into_process(&mut process)),
                Ok(element_term),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::element(heap_binary_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(
                erlang::element(subbinary_term, 0.into_process(&mut process)),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod head {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(erlang::head(atom_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let empty_list_term = Term::EMPTY_LIST;

            assert_eq_in_process!(
                erlang::head(empty_list_term),
                Err(bad_argument!()),
                Default::default()
            );
        }

        #[test]
        fn with_list_returns_head() {
            let mut process: Process = Default::default();
            let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);

            assert_eq_in_process!(erlang::head(list_term), Ok(head_term), process);
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::head(small_integer_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::head(big_integer_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(erlang::head(float_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(erlang::head(local_pid_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::head(external_pid_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(erlang::head(tuple_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::head(heap_binary_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(erlang::head(subbinary_term), Err(bad_argument!()), process);
        }
    }

    mod insert_element {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::insert_element(
                    atom_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::insert_element(
                    Term::EMPTY_LIST,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    list_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    small_integer_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    big_integer_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    float_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::insert_element(
                    local_pid_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::insert_element(
                    external_pid_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_without_small_integer_index_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 2.into_process(&mut process)],
                &mut process,
            );
            let index = 1usize;
            let invalid_index_term = Term::arity(index);

            assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                erlang::insert_element(
                    tuple_term,
                    invalid_index_term,
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );

            let valid_index_term: Term = index.into_process(&mut process);

            assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                erlang::insert_element(
                    tuple_term,
                    valid_index_term,
                    1.into_process(&mut process),
                    &mut process
                ),
                Ok(Term::slice_to_tuple(
                    &[
                        0.into_process(&mut process),
                        1.into_process(&mut process),
                        2.into_process(&mut process)
                    ],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_tuple_without_index_in_range_is_bad_argument() {
            let mut process: Process = Default::default();
            let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    empty_tuple_term,
                    1.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_in_range_returns_tuple_with_new_element_at_index() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(
                &[0.into_process(&mut process), 2.into_process(&mut process)],
                &mut process,
            );

            assert_eq_in_process!(
                erlang::insert_element(
                    tuple_term,
                    1.into_process(&mut process),
                    1.into_process(&mut process),
                    &mut process
                ),
                Ok(Term::slice_to_tuple(
                    &[
                        0.into_process(&mut process),
                        1.into_process(&mut process),
                        2.into_process(&mut process)
                    ],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_at_size_return_tuples_with_new_element_at_end() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    tuple_term,
                    1.into_process(&mut process),
                    1.into_process(&mut process),
                    &mut process
                ),
                Ok(Term::slice_to_tuple(
                    &[0.into_process(&mut process), 1.into_process(&mut process)],
                    &mut process
                )),
                process
            )
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    heap_binary_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(
                erlang::insert_element(
                    subbinary_term,
                    0.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod is_atom {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_true() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::is_atom(atom_term, &mut process),
                true.into_process(&mut process),
                process
            );
        }

        #[test]
        fn with_booleans_is_true() {
            let mut process: Process = Default::default();
            let true_term = true.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(erlang::is_atom(true_term, &mut process), true_term, process);
            assert_eq_in_process!(
                erlang::is_atom(false_term, &mut process),
                true_term,
                process
            );
        }

        #[test]
        fn with_nil_is_true() {
            let mut process: Process = Default::default();
            let nil_term = Term::str_to_atom("nil", Existence::DoNotCare, &mut process).unwrap();
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(erlang::is_atom(nil_term, &mut process), true_term, process);
        }

        #[test]
        fn with_empty_list_is_false() {
            let mut process: Process = Default::default();
            let empty_list_term = Term::EMPTY_LIST;
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(empty_list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_list_is_false() {
            let mut process: Process = Default::default();
            let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(small_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_big_integer_is_false() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(big_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_float_is_false() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(float_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_local_pid_is_false() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(local_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_external_pid_is_false() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(external_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(tuple_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_heap_binary_is_false() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(heap_binary_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_subbinary_is_false() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_atom(subbinary_term, &mut process),
                false_term,
                process
            );
        }
    }

    mod is_binary {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_false() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(atom_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_empty_list_is_false() {
            let mut process: Process = Default::default();
            let empty_list_term = Term::EMPTY_LIST;
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(empty_list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_list_is_false() {
            let mut process: Process = Default::default();
            let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(small_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_big_integer_is_false() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(big_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_float_is_false() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(float_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_local_pid_is_false() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(local_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_external_pid_is_false() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(external_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(tuple_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_heap_binary_is_true() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(heap_binary_term, &mut process),
                true_term,
                process
            );
        }

        #[test]
        fn with_subbinary_with_bit_count_is_false() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(subbinary_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_subbinary_without_bit_count_is_true() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_binary(subbinary_term, &mut process),
                true_term,
                process
            );
        }
    }

    mod is_integer {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_false() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(atom_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_empty_list_is_false() {
            let mut process: Process = Default::default();
            let empty_list_term = Term::EMPTY_LIST;
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(empty_list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_list_is_false() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_small_integer_is_true() {
            let mut process: Process = Default::default();
            let zero_term = 0usize.into_process(&mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(zero_term, &mut process),
                true_term,
                process
            );
        }

        #[test]
        fn with_big_integer_is_true() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(big_integer_term, &mut process),
                true_term,
                process
            );
        }

        #[test]
        fn with_float_is_false() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(float_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_local_pid_is_false() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(local_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_external_pid_is_false() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(external_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(tuple_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_heap_binary_is_false() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(heap_binary_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_subbinary_is_false() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_integer(subbinary_term, &mut process),
                false_term,
                process
            );
        }
    }

    mod is_list {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_false() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(atom_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_empty_list_is_true() {
            let mut process: Process = Default::default();
            let empty_list_term = Term::EMPTY_LIST;
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(empty_list_term, &mut process),
                true_term,
                process
            );
        }

        #[test]
        fn with_list_is_true() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(erlang::is_list(list_term, &mut process), true_term, process);
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(small_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_big_integer_is_false() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(big_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_float_is_false() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(float_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_local_pid_is_false() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(local_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_external_pid_is_false() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(external_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(tuple_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_heap_binary_is_false() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(heap_binary_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_subbinary_is_false() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_list(subbinary_term, &mut process),
                false_term,
                process
            );
        }
    }

    mod is_tuple {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_false() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(atom_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_empty_list_is_false() {
            let mut process: Process = Default::default();
            let empty_list_term = Term::EMPTY_LIST;
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(empty_list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_list_is_false() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(list_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(small_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_big_integer_is_false() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(big_integer_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_float_is_false() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(float_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_local_pid_is_true() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(local_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_external_pid_is_true() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(external_pid_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_true() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(tuple_term, &mut process),
                true_term,
                process
            );
        }

        #[test]
        fn with_heap_binary_is_false() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(heap_binary_term, &mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_subbinary_is_false() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                erlang::is_tuple(subbinary_term, &mut process),
                false_term,
                process
            );
        }
    }

    mod list_to_pid {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::list_to_pid(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::list_to_pid(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_encoding_local_pid() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<0", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<0.", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<0.1", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<0.1.", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<0.1.2", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );

            assert_eq_in_process!(
                erlang::list_to_pid(
                    Term::str_to_char_list("<0.1.2>", &mut process),
                    &mut process
                ),
                Term::local_pid(1, 2),
                process
            );

            assert_eq_in_process!(
                erlang::list_to_pid(
                    Term::str_to_char_list("<0.1.2>?", &mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_encoding_external_pid() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<1", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<1.", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<1.2", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<1.2.", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );
            assert_eq_in_process!(
                erlang::list_to_pid(Term::str_to_char_list("<1.2.3", &mut process), &mut process),
                Err(bad_argument!()),
                process
            );

            assert_eq_in_process!(
                erlang::list_to_pid(
                    Term::str_to_char_list("<1.2.3>", &mut process),
                    &mut process
                ),
                Term::external_pid(1, 2, 3, &mut process),
                process
            );

            assert_eq_in_process!(
                erlang::list_to_pid(
                    Term::str_to_char_list("<1.2.3>?", &mut process),
                    &mut process
                ),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::list_to_pid(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::list_to_pid(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::list_to_pid(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::list_to_pid(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::list_to_pid(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::list_to_pid(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_false() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::list_to_pid(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_false() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(
                erlang::list_to_pid(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod length {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::length(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_zero() {
            let mut process: Process = Default::default();
            let zero_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::length(Term::EMPTY_LIST, &mut process),
                Ok(zero_term),
                process
            );
        }

        #[test]
        fn with_improper_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
            let tail_term = Term::str_to_atom("tail", Existence::DoNotCare, &mut process).unwrap();
            let improper_list_term = Term::cons(head_term, tail_term, &mut process);

            assert_eq_in_process!(
                erlang::length(improper_list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_length() {
            let mut process: Process = Default::default();
            let list_term = (0..=2).rfold(Term::EMPTY_LIST, |acc, i| {
                Term::cons(i.into_process(&mut process), acc, &mut process)
            });

            assert_eq_in_process!(
                erlang::length(list_term, &mut process),
                Ok(3.into_process(&mut process)),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::length(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::length(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::length(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::length(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::length(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(
                erlang::length(tuple_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_false() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::length(heap_binary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_false() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(
                erlang::length(subbinary_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }
    }

    mod size {
        use super::*;

        use num_traits::Num;

        use crate::process::IntoProcess;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::size(atom_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process: Process = Default::default();

            assert_eq_in_process!(
                erlang::size(Term::EMPTY_LIST, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process: Process = Default::default();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                erlang::size(list_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(small_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(big_integer_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(float_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(
                erlang::size(local_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::size(external_pid_term, &mut process),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_without_elements_is_zero() {
            let mut process: Process = Default::default();
            let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);
            let zero_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(empty_tuple_term, &mut process),
                Ok(zero_term),
                process
            );
        }

        #[test]
        fn with_tuple_with_elements_is_element_count() {
            let mut process: Process = Default::default();
            let element_vec: Vec<Term> =
                (0..=2usize).map(|i| i.into_process(&mut process)).collect();
            let element_slice: &[Term] = element_vec.as_slice();
            let tuple_term = Term::slice_to_tuple(element_slice, &mut process);
            let arity_term = 3usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(tuple_term, &mut process),
                Ok(arity_term),
                process
            );
        }

        #[test]
        fn with_heap_binary_is_byte_count() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);
            let byte_count_term = 3usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(heap_binary_term, &mut process),
                Ok(byte_count_term),
                process
            );
        }

        #[test]
        fn with_subbinary_with_bit_count_is_byte_count() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
            let byte_count_term = 2usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(subbinary_term, &mut process),
                Ok(byte_count_term),
                process
            );
        }

        #[test]
        fn with_subbinary_without_bit_count_is_byte_count() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);
            let byte_count_term = 2usize.into_process(&mut process);

            assert_eq_in_process!(
                erlang::size(subbinary_term, &mut process),
                Ok(byte_count_term),
                process
            );
        }
    }

    mod tail {
        use super::*;

        use num_traits::Num;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process: Process = Default::default();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq_in_process!(erlang::tail(atom_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let empty_list_term = Term::EMPTY_LIST;

            assert_eq_in_process!(
                erlang::tail(empty_list_term),
                Err(bad_argument!()),
                Default::default()
            );
        }

        #[test]
        fn with_list_returns_tail() {
            let mut process: Process = Default::default();
            let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);

            assert_eq_in_process!(erlang::tail(list_term), Ok(Term::EMPTY_LIST), process);
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let small_integer_term = 0.into_process(&mut process);

            assert_eq_in_process!(
                erlang::tail(small_integer_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_big_integer_is_bad_argument() {
            let mut process: Process = Default::default();
            let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
                .unwrap()
                .into_process(&mut process);

            assert_eq_in_process!(
                erlang::tail(big_integer_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_float_is_bad_argument() {
            let mut process: Process = Default::default();
            let float_term = 1.0.into_process(&mut process);

            assert_eq_in_process!(erlang::tail(float_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_local_pid_is_bad_argument() {
            let process: Process = Default::default();
            let local_pid_term = Term::local_pid(0, 0).unwrap();

            assert_eq_in_process!(erlang::tail(local_pid_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_external_pid_is_bad_argument() {
            let mut process: Process = Default::default();
            let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

            assert_eq_in_process!(
                erlang::tail(external_pid_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process: Process = Default::default();
            let tuple_term = Term::slice_to_tuple(&[], &mut process);

            assert_eq_in_process!(erlang::tail(tuple_term), Err(bad_argument!()), process);
        }

        #[test]
        fn with_heap_binary_is_bad_argument() {
            let mut process: Process = Default::default();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq_in_process!(
                erlang::tail(heap_binary_term),
                Err(bad_argument!()),
                process
            );
        }

        #[test]
        fn with_subbinary_is_bad_argument() {
            let mut process: Process = Default::default();
            let binary_term =
                Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
            let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

            assert_eq_in_process!(erlang::tail(subbinary_term), Err(bad_argument!()), process);
        }
    }

    fn list_term(mut process: &mut Process) -> Term {
        let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
        Term::cons(head_term, Term::EMPTY_LIST, process)
    }
}
