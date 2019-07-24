//! Mirrors [erlang](http://erlang::org/doc/man/erlang::html) module

use core::cmp::Ordering;
use core::convert::TryInto;
use core::num::FpCategory;

use alloc::sync::Arc;

use num_bigint::BigInt;
use num_traits::Zero;

use liblumen_core::locks::MutexGuard;

use liblumen_alloc::erts::exception::runtime::Class;
use liblumen_alloc::erts::exception::{Exception, Result};
use liblumen_alloc::erts::term::binary::{
    AlignedBinary, Bitstring, IterableBitstring, MaybeAlignedMaybeBinary, MaybePartialByte,
};
use liblumen_alloc::erts::term::{
    atom_unchecked, AsTerm, Atom, Boxed, Cons, Encoding, Float, ImproperList, Map, SmallInteger,
    Term, Tuple, TypedTerm,
};
use liblumen_alloc::erts::ProcessControlBlock;
use liblumen_alloc::{badarg, badarith, badkey, badmap, default_heap, error, exit, raise, throw};

use crate::binary::{start_length_to_part_range, PartRange, ToTermOptions};
use crate::code;
use crate::node;
use crate::otp;
use crate::process::Alloc;
use crate::registry::{self, pid_to_self_or_process};
use crate::scheduler::Scheduler;
use crate::send::{self, send, Sent};
use crate::stacktrace;
use crate::time::{
    self,
    monotonic::{self, Milliseconds},
    Unit::*,
};
use crate::timer::start::ReferenceFrame;
use crate::timer::{self, Timeout};
use crate::tuple::ZeroBasedIndex;
use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod tests;

pub fn abs_1(number: Term, process_control_block: &ProcessControlBlock) -> Result {
    let option_abs = match number.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let i: isize = small_integer.into();

            if i < 0 {
                let positive = -i;
                let abs_number = process_control_block.integer(positive)?;

                Some(abs_number)
            } else {
                Some(number)
            }
        }
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();
                let zero_big_int: &BigInt = &Zero::zero();

                let abs_number: Term = if big_int < zero_big_int {
                    let positive_big_int: BigInt = -1 * big_int;

                    process_control_block.integer(positive_big_int)?
                } else {
                    number
                };

                Some(abs_number)
            }
            TypedTerm::Float(float) => {
                let f: f64 = float.into();

                let abs_number = match f.partial_cmp(&0.0).unwrap() {
                    Ordering::Less => {
                        let positive_f = f.abs();

                        process_control_block.float(positive_f).unwrap()
                    }
                    _ => number,
                };

                Some(abs_number)
            }
            _ => None,
        },
        _ => None,
    };

    match option_abs {
        Some(abs) => Ok(abs),
        None => Err(badarg!().into()),
    }
}

/// `+/2` infix operator
pub fn add_2(augend: Term, addend: Term, process_control_block: &ProcessControlBlock) -> Result {
    number_infix_operator!(augend, addend, process_control_block, checked_add, +)
}

/// `and/2` infix operator.
///
/// **NOTE: NOT SHORT-CIRCUITING!**  Use `andalso/2` for short-circuiting, but it doesn't enforce
/// that `right` is boolean.
pub fn and_2(left_boolean: Term, right_boolean: Term) -> Result {
    boolean_infix_operator!(left_boolean, right_boolean, &)
}

/// `andalso/2` infix operator.
///
/// Short-circuiting, but doesn't enforce `right` is boolean.  If you need to enforce `boolean` for
/// both operands, use `and_2`.
pub fn andalso_2(boolean: Term, term: Term) -> Result {
    let boolean_bool: bool = boolean.try_into()?;

    if boolean_bool {
        Ok(term)
    } else {
        // always `false.into()`, but this is faster
        Ok(boolean)
    }
}

pub fn append_element_2(
    tuple: Term,
    element: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let internal: Boxed<Tuple> = tuple.try_into()?;
    let new_tuple = process_control_block.tuple_from_slices(&[&internal[..], &[element]])?;

    Ok(new_tuple)
}

/// `==/2` infix operator.  Unlike `=:=`, converts between floats and integers.
pub fn are_equal_after_conversion_2(left: Term, right: Term) -> Term {
    left.eq(&right).into()
}

/// `=:=/2` infix operator.  Unlike `==`, does not convert between floats and integers.
pub fn are_exactly_equal_2(left: Term, right: Term) -> Term {
    left.exactly_eq(&right).into()
}

/// `=/=/2` infix operator.  Unlike `!=`, does not convert between floats and integers.
pub fn are_exactly_not_equal_2(left: Term, right: Term) -> Term {
    left.exactly_ne(&right).into()
}

/// `/=/2` infix operator.  Unlike `=/=`, converts between floats and integers.
pub fn are_not_equal_after_conversion_2(left: Term, right: Term) -> Term {
    left.ne(&right).into()
}

pub fn atom_to_binary_2(
    atom: Term,
    encoding: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    match atom.to_typed_term().unwrap() {
        TypedTerm::Atom(atom) => {
            let _: Encoding = encoding.try_into()?;
            let binary = process_control_block.binary_from_str(atom.name())?;

            Ok(binary)
        }
        _ => Err(badarg!().into()),
    }
}

pub fn atom_to_list_1(atom: Term, process_control_block: &ProcessControlBlock) -> Result {
    match atom.to_typed_term().unwrap() {
        TypedTerm::Atom(atom) => {
            let chars = atom.name().chars();

            process_control_block
                .list_from_chars(chars)
                .map_err(|error| error.into())
        }
        _ => Err(badarg!().into()),
    }
}

// `band/2` infix operator.
pub fn band_2(
    left_integer: Term,
    right_integer: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    bitwise_infix_operator!(left_integer, right_integer, process_control_block, &)
}

pub fn binary_part_2(
    binary: Term,
    start_length: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let option_result = match start_length.to_typed_term().unwrap() {
        TypedTerm::Boxed(unboxed_start_length) => {
            match unboxed_start_length.to_typed_term().unwrap() {
                TypedTerm::Tuple(tuple) => {
                    if tuple.len() == 2 {
                        Some(binary_part_3(
                            binary,
                            tuple[0],
                            tuple[1],
                            process_control_block,
                        ))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    };

    match option_result {
        Some(result) => result,
        None => Err(badarg!().into()),
    }
}

pub fn binary_part_3(
    binary: Term,
    start: Term,
    length: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let start_usize: usize = start.try_into()?;
    let length_isize: isize = length.try_into()?;

    match binary.to_typed_term().unwrap() {
        TypedTerm::Boxed(unboxed_binary) => match unboxed_binary.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => {
                let available_byte_count = heap_binary.full_byte_len();
                let PartRange {
                    byte_offset,
                    byte_len,
                } = start_length_to_part_range(start_usize, length_isize, available_byte_count)?;

                if (byte_offset == 0) && (byte_len == available_byte_count) {
                    Ok(binary)
                } else {
                    process_control_block
                        .subbinary_from_original(binary, byte_offset, 0, byte_len, 0)
                        .map_err(|error| error.into())
                }
            }
            TypedTerm::ProcBin(process_binary) => {
                let available_byte_count = process_binary.full_byte_len();
                let PartRange {
                    byte_offset,
                    byte_len,
                } = start_length_to_part_range(start_usize, length_isize, available_byte_count)?;

                if (byte_offset == 0) && (byte_len == available_byte_count) {
                    Ok(binary)
                } else {
                    process_control_block
                        .subbinary_from_original(binary, byte_offset, 0, byte_len, 0)
                        .map_err(|error| error.into())
                }
            }
            TypedTerm::SubBinary(subbinary) => {
                let PartRange {
                    byte_offset,
                    byte_len,
                } = start_length_to_part_range(
                    start_usize,
                    length_isize,
                    subbinary.full_byte_len(),
                )?;

                // new subbinary is entire subbinary
                if (subbinary.is_binary())
                    && (byte_offset == 0)
                    && (byte_len == subbinary.full_byte_len())
                {
                    Ok(binary)
                } else {
                    process_control_block
                        .subbinary_from_original(
                            subbinary.original(),
                            subbinary.byte_offset() + byte_offset,
                            subbinary.bit_offset(),
                            byte_len,
                            0,
                        )
                        .map_err(|error| error.into())
                }
            }
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}

pub fn binary_to_atom_2(binary: Term, encoding: Term) -> Result {
    let _: Encoding = encoding.try_into()?;

    match binary.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => {
                Atom::try_from_latin1_bytes(heap_binary.as_bytes()).map_err(|error| error.into())
            }
            TypedTerm::ProcBin(process_binary) => {
                Atom::try_from_latin1_bytes(process_binary.as_bytes()).map_err(|error| error.into())
            }
            TypedTerm::SubBinary(subbinary) => {
                if subbinary.is_binary() {
                    if subbinary.is_aligned() {
                        let bytes = unsafe { subbinary.as_bytes() };

                        Atom::try_from_latin1_bytes(bytes)
                    } else {
                        let byte_vec: Vec<u8> = subbinary.full_byte_iter().collect();

                        Atom::try_from_latin1_bytes(&byte_vec)
                    }
                    .map_err(|error| error.into())
                } else {
                    Err(badarg!().into())
                }
            }
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
    .map(|atom| unsafe { atom.as_term() })
}

pub fn binary_to_existing_atom_2(binary: Term, encoding: Term) -> Result {
    let _: Encoding = encoding.try_into()?;

    match binary.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => {
                Atom::try_from_latin1_bytes_existing(heap_binary.as_bytes())
                    .map_err(|error| error.into())
            }
            TypedTerm::ProcBin(process_binary) => {
                Atom::try_from_latin1_bytes_existing(process_binary.as_bytes())
                    .map_err(|error| error.into())
            }
            TypedTerm::SubBinary(subbinary) => {
                if subbinary.is_binary() {
                    if subbinary.is_aligned() {
                        let bytes = unsafe { subbinary.as_bytes() };

                        Atom::try_from_latin1_bytes_existing(bytes)
                    } else {
                        let byte_vec: Vec<u8> = subbinary.full_byte_iter().collect();

                        Atom::try_from_latin1_bytes_existing(&byte_vec)
                    }
                    .map_err(|error| error.into())
                } else {
                    Err(badarg!().into())
                }
            }
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
    .map(|atom| unsafe { atom.as_term() })
}

pub fn binary_to_float_1<'process>(
    binary: Term,
    process_control_block: &'process ProcessControlBlock,
) -> Result {
    let mut heap: MutexGuard<'process, _> = process_control_block.acquire_heap();
    let s: &str = heap.str_from_binary(binary)?;

    match s.parse::<f64>() {
        Ok(inner) => {
            match inner.classify() {
                FpCategory::Normal | FpCategory::Subnormal =>
                // unlike Rust, Erlang requires float strings to have a decimal point
                {
                    if (inner.fract() == 0.0) & !s.chars().any(|b| b == '.') {
                        Err(badarg!().into())
                    } else {
                        heap.float(inner).map_err(|error| error.into())
                    }
                }
                // Erlang has no support for Nan, +inf or -inf
                FpCategory::Nan | FpCategory::Infinite => Err(badarg!().into()),
                FpCategory::Zero => {
                    // Erlang does not track the difference without +0 and -0.
                    let zero = inner.abs();

                    heap.float(zero).map_err(|error| error.into())
                }
            }
        }
        Err(_) => Err(badarg!().into()),
    }
}

pub fn binary_to_integer_1<'process>(
    binary: Term,
    process_control_block: &'process ProcessControlBlock,
) -> Result {
    let mut heap = process_control_block.acquire_heap();
    let s: &str = heap.str_from_binary(binary)?;
    let bytes = s.as_bytes();

    match BigInt::parse_bytes(bytes, 10) {
        Some(big_int) => {
            let term = heap.integer(big_int)?;

            Ok(term)
        }
        None => Err(badarg!().into()),
    }
}

pub fn binary_to_integer_2<'process>(
    binary: Term,
    base: Term,
    process_control_block: &'process ProcessControlBlock,
) -> Result {
    let mut heap = process_control_block.acquire_heap();
    let s: &str = heap.str_from_binary(binary)?;
    let radix: usize = base.try_into()?;

    if 2 <= radix && radix <= 36 {
        let bytes = s.as_bytes();

        match BigInt::parse_bytes(bytes, radix as u32) {
            Some(big_int) => {
                let term = heap.integer(big_int)?;

                Ok(term)
            }
            None => Err(badarg!()),
        }
    } else {
        Err(badarg!())
    }
    .map_err(|error| error.into())
}

pub fn binary_to_list_1(binary: Term, process_control_block: &ProcessControlBlock) -> Result {
    let bytes = process_control_block.bytes_from_binary(binary)?;
    let byte_terms = bytes.iter().map(|byte| (*byte).into());

    process_control_block
        .list_from_iter(byte_terms)
        .map_err(|error| error.into())
}

/// The one-based indexing for binaries used by this function is deprecated. New code is to use
/// [crate::otp::binary::bin_to_list] instead. All functions in module [crate::otp::binary]
/// consistently use zero-based indexing.
pub fn binary_to_list_3(
    binary: Term,
    start: Term,
    stop: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let one_based_start_usize: usize = start.try_into()?;

    if 1 <= one_based_start_usize {
        let one_based_stop_usize: usize = stop.try_into()?;

        if one_based_start_usize <= one_based_stop_usize {
            let zero_based_start_usize = one_based_start_usize - 1;
            let zero_based_stop_usize = one_based_stop_usize - 1;

            let length_usize = zero_based_stop_usize - zero_based_start_usize + 1;

            otp::binary::bin_to_list(
                binary,
                process_control_block.integer(zero_based_start_usize)?,
                process_control_block.integer(length_usize)?,
                process_control_block,
            )
        } else {
            Err(badarg!().into())
        }
    } else {
        Err(badarg!().into())
    }
}

pub fn binary_to_term_1(binary: Term, process_control_block: &ProcessControlBlock) -> Result {
    binary_to_term_2(binary, Term::NIL, process_control_block)
}

pub fn binary_to_term_2(
    binary: Term,
    options: Term,
    _process_control_block: &ProcessControlBlock,
) -> Result {
    let _to_term_options: ToTermOptions = options.try_into()?;

    match binary.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(_heap_binary) => unimplemented!(),
            TypedTerm::ProcBin(_process_binary) => unimplemented!(),
            TypedTerm::SubBinary(_subbinary) => unimplemented!(),
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}

pub fn bit_size_1(bitstring: Term, process_control_block: &ProcessControlBlock) -> Result {
    let option_total_bit_len = match bitstring.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => Some(heap_binary.total_bit_len()),
            TypedTerm::ProcBin(process_binary) => Some(process_binary.total_bit_len()),
            TypedTerm::SubBinary(subbinary) => Some(subbinary.total_bit_len()),
            _ => None,
        },
        _ => None,
    };

    match option_total_bit_len {
        Some(total_bit_len) => Ok(process_control_block.integer(total_bit_len)?),
        None => Err(badarg!().into()),
    }
}

/// Returns a list of integers corresponding to the bytes of `bitstring`. If the number of bits in
/// `bitstring` is not divisible by `8`, the last element of the list is a `bitstring` containing
/// the remaining `1`-`7` bits.
pub fn bitstring_to_list_1<'process>(
    bitstring: Term,
    process_control_block: &'process ProcessControlBlock,
) -> Result {
    match bitstring.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => {
                let byte_term_iter = heap_binary.as_bytes().iter().map(|byte| (*byte).into());
                let last = Term::NIL;

                process_control_block
                    .improper_list_from_iter(byte_term_iter, last)
                    .map_err(|error| error.into())
            }
            TypedTerm::ProcBin(process_binary) => {
                let byte_term_iter = process_binary.as_bytes().iter().map(|byte| (*byte).into());
                let last = Term::NIL;

                process_control_block
                    .improper_list_from_iter(byte_term_iter, last)
                    .map_err(|error| error.into())
            }
            TypedTerm::SubBinary(subbinary) => {
                let last = if subbinary.is_binary() {
                    Term::NIL
                } else {
                    process_control_block.subbinary_from_original(
                        subbinary.original(),
                        subbinary.byte_offset() + subbinary.full_byte_len(),
                        subbinary.bit_offset(),
                        0,
                        subbinary.partial_byte_bit_len(),
                    )?
                };

                let byte_term_iter = subbinary.full_byte_iter().map(|byte| byte.into());

                process_control_block
                    .improper_list_from_iter(byte_term_iter, last)
                    .map_err(|error| error.into())
            }
            _ => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}

// `bnot/1` prefix operator.
pub fn bnot_1(integer: Term, process_control_block: &ProcessControlBlock) -> Result {
    match integer.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let integer_isize: isize = small_integer.into();
            let output = !integer_isize;
            let output_term = process_control_block.integer(output)?;

            Ok(output_term)
        }
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();
                let output_big_int = !big_int;
                let output_term = process_control_block.integer(output_big_int)?;

                Ok(output_term)
            }
            _ => Err(badarith!().into()),
        },
        _ => Err(badarith!().into()),
    }
}

/// `bor/2` infix operator.
pub fn bor_2(
    left_integer: Term,
    right_integer: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    bitwise_infix_operator!(left_integer, right_integer, process_control_block, |)
}

pub const MAX_SHIFT: usize = std::mem::size_of::<isize>() * 8 - 1;

/// `bsl/2` infix operator.
pub fn bsl_2(integer: Term, shift: Term, process_control_block: &ProcessControlBlock) -> Result {
    bitshift_infix_operator!(integer, shift, process_control_block, <<, >>)
}

/// `bsr/2` infix operator.
pub fn bsr_2(integer: Term, shift: Term, process_control_block: &ProcessControlBlock) -> Result {
    bitshift_infix_operator!(integer, shift, process_control_block, >>, <<)
}

/// `bxor/2` infix operator.
pub fn bxor_2(
    left_integer: Term,
    right_integer: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    bitwise_infix_operator!(left_integer, right_integer, process_control_block, ^)
}

pub fn byte_size_1(bitstring: Term, process_control_block: &ProcessControlBlock) -> Result {
    let option_total_byte_len = match bitstring.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => Some(heap_binary.total_byte_len()),
            TypedTerm::ProcBin(process_binary) => Some(process_binary.total_byte_len()),
            TypedTerm::SubBinary(subbinary) => Some(subbinary.total_byte_len()),
            _ => None,
        },
        _ => None,
    };

    match option_total_byte_len {
        Some(total_byte_len) => Ok(process_control_block.integer(total_byte_len)?),
        None => Err(badarg!().into()),
    }
}

pub fn cancel_timer_1(
    timer_reference: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    cancel_timer(timer_reference, Default::default(), process_control_block)
}

pub fn cancel_timer_2(
    timer_reference: Term,
    options: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let cancel_timer_options: timer::cancel::Options = options.try_into()?;

    cancel_timer(timer_reference, cancel_timer_options, process_control_block)
}

pub fn ceil_1(number: Term, process_control_block: &ProcessControlBlock) -> Result {
    let option_ceil = match number.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(_) => Some(number),
        TypedTerm::Boxed(boxed) => {
            match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(_) => Some(number),
                TypedTerm::Float(float) => {
                    let inner: f64 = float.into();
                    let ceil_inner = inner.ceil();

                    // skip creating a BigInt if float can fit in small integer.
                    let ceil_term = if (SmallInteger::MIN_VALUE as f64).max(Float::INTEGRAL_MIN)
                        <= ceil_inner
                        && ceil_inner <= (SmallInteger::MAX_VALUE as f64).min(Float::INTEGRAL_MAX)
                    {
                        process_control_block.integer(ceil_inner as isize)?
                    } else {
                        let ceil_string = ceil_inner.to_string();
                        let ceil_bytes = ceil_string.as_bytes();
                        let big_int = BigInt::parse_bytes(ceil_bytes, 10).unwrap();

                        process_control_block.integer(big_int)?
                    };

                    Some(ceil_term)
                }
                _ => None,
            }
        }
        _ => None,
    };

    match option_ceil {
        Some(ceil) => Ok(ceil),
        None => Err(badarg!().into()),
    }
}

/// `++/2`
pub fn concatenate_2(
    list: Term,
    term: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    match list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok(term),
        TypedTerm::List(cons) => match cons
            .into_iter()
            .collect::<std::result::Result<Vec<Term>, _>>()
        {
            Ok(vec) => process_control_block
                .improper_list_from_slice(&vec, term)
                .map_err(|error| error.into()),
            Err(ImproperList { .. }) => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}

pub fn convert_time_unit_3(
    time: Term,
    from_unit: Term,
    to_unit: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let time_big_int: BigInt = time.try_into()?;
    let from_unit_unit: crate::time::Unit = from_unit.try_into()?;
    let to_unit_unit: crate::time::Unit = to_unit.try_into()?;
    let converted_big_int = time::convert(time_big_int, from_unit_unit, to_unit_unit);
    let converted_term = process_control_block.integer(converted_big_int)?;

    Ok(converted_term)
}

pub fn delete_element_2(
    index: Term,
    tuple: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let initial_inner_tuple: Boxed<Tuple> = tuple.try_into()?;
    let ZeroBasedIndex(index_zero_based): ZeroBasedIndex = index.try_into()?;
    let initial_len = initial_inner_tuple.len();

    if index_zero_based < initial_len {
        let smaller_len = initial_len - 1;
        let smaller_element_iterator =
            initial_inner_tuple
                .iter()
                .enumerate()
                .filter_map(|(old_index, old_term)| {
                    if old_index == index_zero_based {
                        None
                    } else {
                        Some(old_term)
                    }
                });
        let smaller_tuple =
            process_control_block.tuple_from_iter(smaller_element_iterator, smaller_len)?;

        Ok(smaller_tuple)
    } else {
        Err(badarg!().into())
    }
}

/// `div/2` infix operator.  Integer division.
pub fn div_2(dividend: Term, divisor: Term, process_control_block: &ProcessControlBlock) -> Result {
    integer_infix_operator!(dividend, divisor, process_control_block, /)
}

/// `//2` infix operator.  Unlike `+/2`, `-/2` and `*/2` always promotes to `float` returns the
/// `float`.
pub fn divide_2(
    dividend: Term,
    divisor: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let dividend_f64: f64 = dividend.try_into().map_err(|_| badarith!())?;
    let divisor_f64: f64 = divisor.try_into().map_err(|_| badarith!())?;

    if divisor_f64 == 0.0 {
        Err(badarith!().into())
    } else {
        let quotient_f64 = dividend_f64 / divisor_f64;
        let quotient_term = process_control_block.float(quotient_f64)?;

        Ok(quotient_term)
    }
}

pub fn element_2(index: Term, tuple: Term) -> Result {
    let inner_tuple: Boxed<Tuple> = tuple.try_into()?;
    let index_usize: usize = index.try_into()?;

    inner_tuple
        .get_element_internal(index_usize)
        .map_err(|error| error.into())
}

/// `orelse/2` infix operator.
///
/// Short-circuiting, but doesn't enforce `right` is boolean.  If you need to enforce `boolean` for
/// both operands, use `or_2`.
pub fn orelse_2(boolean: Term, term: Term) -> Result {
    let boolean_bool: bool = boolean.try_into()?;

    if boolean_bool {
        // always `true.into()`, but this is faster
        Ok(boolean)
    } else {
        Ok(term)
    }
}

pub fn error_1(reason: Term) -> Result {
    Err(error!(reason).into())
}

pub fn error_2(reason: Term, arguments: Term) -> Result {
    Err(error!(reason, Some(arguments)).into())
}

pub fn exit_1(reason: Term) -> Result {
    Err(exit!(reason).into())
}

pub fn hd_1(list: Term) -> Result {
    let cons: Boxed<Cons> = list.try_into()?;

    Ok(cons.head)
}

pub fn insert_element_3(
    index: Term,
    tuple: Term,
    element: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let initial_inner_tuple: Boxed<Tuple> = tuple.try_into()?;
    let ZeroBasedIndex(index_zero_based): ZeroBasedIndex = index.try_into()?;

    let length = initial_inner_tuple.len();

    // can be equal to arity when insertion is at the end
    if index_zero_based <= length {
        if index_zero_based == 0 {
            process_control_block.tuple_from_slices(&[&[element], &initial_inner_tuple[..]])
        } else if index_zero_based < length {
            process_control_block.tuple_from_slices(&[
                &initial_inner_tuple[..index_zero_based],
                &[element],
                &initial_inner_tuple[index_zero_based..],
            ])
        } else {
            process_control_block.tuple_from_slices(&[&initial_inner_tuple[..], &[element]])
        }
        .map_err(|error| error.into())
    } else {
        Err(badarg!().into())
    }
}

/// Distribution is not supported at this time.  Always returns `false`.
pub fn is_alive_0() -> Term {
    false.into()
}

pub fn is_atom_1(term: Term) -> Term {
    term.is_atom().into()
}

pub fn is_binary_1(term: Term) -> Term {
    term.is_binary().into()
}

pub fn is_bitstring_1(term: Term) -> Term {
    term.is_bitstring().into()
}

pub fn is_boolean_1(term: Term) -> Term {
    term.is_boolean().into()
}

/// `=</2` infix operator.  Floats and integers are converted.
///
/// **NOTE: `=</2` is not a typo.  Unlike `>=/2`, which has the `=` second, Erlang put the `=` first
/// for `=</2`, instead of the more common `<=`.
pub fn is_equal_or_less_than_2(left: Term, right: Term) -> Term {
    left.le(&right).into()
}

pub fn is_float_1(term: Term) -> Term {
    term.is_float().into()
}

/// `>/2` infix operator.  Floats and integers are converted.
pub fn is_greater_than_2(left: Term, right: Term) -> Term {
    left.gt(&right).into()
}

/// `>=/2` infix operator.  Floats and integers are converted.
pub fn is_greater_than_or_equal_2(left: Term, right: Term) -> Term {
    left.ge(&right).into()
}

pub fn is_integer_1(term: Term) -> Term {
    term.is_integer().into()
}

/// `</2` infix operator.  Floats and integers are converted.
pub fn is_less_than_2(left: Term, right: Term) -> Term {
    left.lt(&right).into()
}

pub fn is_list_1(term: Term) -> Term {
    term.is_list().into()
}

pub fn is_map_1(term: Term) -> Term {
    term.is_map().into()
}

pub fn is_map_key_2(key: Term, map: Term, process_control_block: &ProcessControlBlock) -> Result {
    let result: core::result::Result<Boxed<Map>, _> = map.try_into();

    match result {
        Ok(map_header) => Ok(map_header.is_key(key).into()),
        Err(_) => Err(badmap!(&mut process_control_block.acquire_heap(), map)),
    }
}

pub fn is_number_1(term: Term) -> Term {
    term.is_number().into()
}

pub fn is_pid_1(term: Term) -> Term {
    term.is_pid().into()
}

pub fn is_record_2(term: Term, record_tag: Term) -> Result {
    is_record(term, record_tag, None)
}

pub fn is_record_3(term: Term, record_tag: Term, size: Term) -> Result {
    is_record(term, record_tag, Some(size))
}

pub fn is_reference_1(term: Term) -> Term {
    match term.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => boxed.to_typed_term().unwrap().is_reference(),
        _ => false,
    }
    .into()
}

pub fn is_tuple_1(term: Term) -> Term {
    term.is_tuple().into()
}

pub fn length_1(list: Term, process_control_block: &ProcessControlBlock) -> Result {
    match list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok(0.into()),
        TypedTerm::List(cons) => match cons.count() {
            Some(count) => Ok(process_control_block.integer(count)?),
            None => Err(badarg!().into()),
        },
        _ => Err(badarg!().into()),
    }
}

pub fn list_to_atom_1(string: Term) -> Result {
    list_to_string(string).and_then(|s| match Atom::try_from_str(s) {
        Ok(atom) => unsafe { Ok(atom.as_term()) },
        Err(_) => Err(badarg!().into()),
    })
}

pub fn list_to_existing_atom_1(string: Term) -> Result {
    list_to_string(string).and_then(|s| match Atom::try_from_str_existing(s) {
        Ok(atom) => unsafe { Ok(atom.as_term()) },
        Err(_) => Err(badarg!().into()),
    })
}

pub fn list_to_binary_1(iolist: Term, process_control_block: &ProcessControlBlock) -> Result {
    match iolist.to_typed_term().unwrap() {
        TypedTerm::Nil | TypedTerm::List(_) => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut stack: Vec<Term> = vec![iolist];

            while let Some(top) = stack.pop() {
                match top.to_typed_term().unwrap() {
                    TypedTerm::SmallInteger(small_integer) => {
                        let top_byte = small_integer.try_into()?;

                        byte_vec.push(top_byte);
                    }
                    TypedTerm::Nil => (),
                    TypedTerm::List(boxed_cons) => {
                        // @type iolist :: maybe_improper_list(byte() | binary() | iolist(),
                        // binary() | []) means that `byte()` isn't allowed
                        // for `tail`s unlike `head`.

                        let tail = boxed_cons.tail;

                        if tail.is_smallint() {
                            return Err(badarg!().into());
                        } else {
                            stack.push(tail);
                        }

                        stack.push(boxed_cons.head);
                    }
                    TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                        TypedTerm::HeapBinary(heap_binary) => {
                            byte_vec.extend_from_slice(heap_binary.as_bytes());
                        }
                        TypedTerm::SubBinary(subbinary) => {
                            if subbinary.is_binary() {
                                if subbinary.is_aligned() {
                                    byte_vec.extend(unsafe { subbinary.as_bytes() });
                                } else {
                                    byte_vec.extend(subbinary.full_byte_iter());
                                }
                            } else {
                                return Err(badarg!().into());
                            }
                        }
                        _ => return Err(badarg!().into()),
                    },
                    _ => return Err(badarg!().into()),
                }
            }

            Ok(process_control_block
                .binary_from_bytes(byte_vec.as_slice())
                .unwrap())
        }
        _ => Err(badarg!().into()),
    }
}

pub fn list_to_bitstring_1(iolist: Term, process_control_block: &ProcessControlBlock) -> Result {
    match iolist.to_typed_term().unwrap() {
        TypedTerm::Nil | TypedTerm::List(_) => {
            let mut byte_vec: Vec<u8> = Vec::new();
            let mut bit_offset = 0;
            let mut tail_byte = 0;
            let mut stack: Vec<Term> = vec![iolist];

            while let Some(top) = stack.pop() {
                match top.to_typed_term().unwrap() {
                    TypedTerm::SmallInteger(small_integer) => {
                        let top_byte = small_integer.try_into()?;

                        if bit_offset == 0 {
                            byte_vec.push(top_byte);
                        } else {
                            tail_byte |= top_byte >> bit_offset;
                            byte_vec.push(tail_byte);

                            tail_byte = top_byte << (8 - bit_offset);
                        }
                    }
                    TypedTerm::Nil => (),
                    TypedTerm::List(boxed_cons) => {
                        // @type bitstring_list ::
                        //   maybe_improper_list(byte() | bitstring() | bitstring_list(),
                        //                       bitstring() | [])
                        // means that `byte()` isn't allowed for `tail`s unlike `head`.

                        let tail = boxed_cons.tail;

                        if tail.is_smallint() {
                            return Err(badarg!().into());
                        } else {
                            stack.push(tail);
                        }

                        stack.push(boxed_cons.head);
                    }
                    TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                        TypedTerm::HeapBinary(heap_binary) => {
                            if bit_offset == 0 {
                                byte_vec.extend_from_slice(heap_binary.as_bytes());
                            } else {
                                for byte in heap_binary.as_bytes() {
                                    tail_byte |= byte >> bit_offset;
                                    byte_vec.push(tail_byte);

                                    tail_byte = byte << (8 - bit_offset);
                                }
                            }
                        }
                        TypedTerm::SubBinary(subbinary) => {
                            if bit_offset == 0 {
                                if subbinary.is_aligned() {
                                    byte_vec.extend(unsafe { subbinary.as_bytes() });
                                } else {
                                    byte_vec.extend(subbinary.full_byte_iter());
                                }
                            } else {
                                for byte in subbinary.full_byte_iter() {
                                    tail_byte |= byte >> bit_offset;
                                    byte_vec.push(tail_byte);

                                    tail_byte = byte << (8 - bit_offset);
                                }
                            }

                            if !subbinary.is_binary() {
                                for bit in subbinary.partial_byte_bit_iter() {
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
                        _ => return Err(badarg!().into()),
                    },
                    _ => return Err(badarg!().into()),
                }
            }

            if bit_offset == 0 {
                Ok(process_control_block
                    .binary_from_bytes(byte_vec.as_slice())
                    .unwrap())
            } else {
                let byte_count = byte_vec.len();
                byte_vec.push(tail_byte);
                let original = process_control_block
                    .binary_from_bytes(byte_vec.as_slice())
                    .unwrap();

                Ok(process_control_block
                    .subbinary_from_original(original, byte_count, 0, bit_offset as usize, 0)
                    .unwrap())
            }
        }
        _ => Err(badarg!().into()),
    }
}

pub fn list_to_pid_1(string: Term, process_control_block: &ProcessControlBlock) -> Result {
    let cons: Boxed<Cons> = string.try_into()?;

    let prefix_tail = skip_char(cons, '<')?;
    let prefix_tail_cons: Boxed<Cons> = prefix_tail.try_into()?;

    let (node_id, node_tail) = next_decimal(prefix_tail_cons)?;
    let node_tail_cons: Boxed<Cons> = node_tail.try_into()?;

    let first_separator_tail = skip_char(node_tail_cons, '.')?;
    let first_separator_tail_cons: Boxed<Cons> = first_separator_tail.try_into()?;

    let (number, number_tail) = next_decimal(first_separator_tail_cons)?;
    let number_tail_cons: Boxed<Cons> = number_tail.try_into()?;

    let second_separator_tail = skip_char(number_tail_cons, '.')?;
    let second_separator_tail_cons: Boxed<Cons> = second_separator_tail.try_into()?;

    let (serial, serial_tail) = next_decimal(second_separator_tail_cons)?;
    let serial_tail_cons: Boxed<Cons> = serial_tail.try_into()?;

    let suffix_tail = skip_char(serial_tail_cons, '>')?;

    if suffix_tail.is_nil() {
        process_control_block
            .pid_with_node_id(node_id, number, serial)
            .map_err(|error| error.into())
    } else {
        Err(badarg!().into())
    }
}

pub fn list_to_tuple_1(list: Term, process_control_block: &ProcessControlBlock) -> Result {
    match list.to_typed_term().unwrap() {
        TypedTerm::Nil => process_control_block
            .tuple_from_slices(&[])
            .map_err(|error| error.into()),
        TypedTerm::List(cons) => {
            let vec: Vec<Term> = cons.into_iter().collect::<std::result::Result<_, _>>()?;

            process_control_block
                .tuple_from_slice(&vec)
                .map_err(|error| error.into())
        }
        _ => Err(badarg!().into()),
    }
}

pub fn make_ref_0(process_control_block: &ProcessControlBlock) -> Result {
    process_control_block
        .next_reference()
        .map_err(|error| error.into())
}

pub fn map_get_2(key: Term, map: Term, process_control_block: &ProcessControlBlock) -> Result {
    let result: core::result::Result<Boxed<Map>, _> = map.try_into();

    match result {
        Ok(map_header) => match map_header.get(key) {
            Some(value) => Ok(value),
            None => Err(badkey!(&mut process_control_block.acquire_heap(), key)),
        },
        Err(_) => Err(badmap!(&mut process_control_block.acquire_heap(), map)),
    }
}

pub fn map_size_1(map: Term, process_control_block: &ProcessControlBlock) -> Result {
    let result: core::result::Result<Boxed<Map>, _> = map.try_into();

    match result {
        Ok(map_header) => {
            let len = map_header.len();
            let len_term = process_control_block.integer(len)?;

            Ok(len_term)
        }
        Err(_) => Err(badmap!(&mut process_control_block.acquire_heap(), map)),
    }
}

/// `max/2`
///
/// Returns the largest of `Term1` and `Term2`. If the terms are equal, `Term1` is returned.
pub fn max_2(term1: Term, term2: Term) -> Term {
    // Flip the order because for Rust `max` returns the second argument when equal, but Erlang
    // returns the first
    term2.max(term1)
}

/// `min/2`
///
/// Returns the smallest of `Term1` and `Term2`. If the terms are equal, `Term1` is returned.
pub fn min_2(term1: Term, term2: Term) -> Term {
    term1.min(term2)
}

pub fn monotonic_time_0(process_control_block: &ProcessControlBlock) -> Result {
    let big_int = monotonic::time(Native);

    Ok(process_control_block.integer(big_int)?)
}

pub fn monotonic_time_1(unit: Term, process_control_block: &ProcessControlBlock) -> Result {
    let unit_unit: crate::time::Unit = unit.try_into()?;
    let big_int = monotonic::time(unit_unit);
    let term = process_control_block.integer(big_int)?;

    Ok(term)
}

/// `*/2` infix operator
pub fn multiply_2(
    multiplier: Term,
    multiplicand: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    number_infix_operator!(multiplier, multiplicand, process_control_block, checked_mul, *)
}

/// `-/1` prefix operator.
pub fn negate_1(number: Term, process_control_block: &ProcessControlBlock) -> Result {
    let option_negated = match number.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let number_isize: isize = small_integer.into();
            let negated_isize = -number_isize;
            let negated_number: Term = process_control_block.integer(negated_isize)?;

            Some(negated_number)
        }
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => {
                let big_int: &BigInt = big_integer.as_ref().into();
                let negated_big_int = -big_int;
                let negated_number = process_control_block.integer(negated_big_int)?;

                Some(negated_number)
            }
            TypedTerm::Float(float) => {
                let number_f64: f64 = float.into();
                let negated_f64: f64 = -number_f64;
                let negated_number = process_control_block.float(negated_f64)?;

                Some(negated_number)
            }
            _ => None,
        },
        _ => None,
    };

    match option_negated {
        Some(negated) => Ok(negated),
        None => Err(badarith!().into()),
    }
}

pub fn node_0() -> Term {
    atom_unchecked(node::DEAD)
}

/// `not/1` prefix operator.
pub fn not_1(boolean: Term) -> Result {
    let boolean_bool: bool = boolean.try_into()?;
    let output = !boolean_bool;

    Ok(output.into())
}

/// `+/1` prefix operator.
pub fn number_or_badarith_1(term: Term) -> Result {
    if term.is_number() {
        Ok(term)
    } else {
        Err(badarith!().into())
    }
}

/// `or/2` infix operator.
///
/// **NOTE: NOT SHORT-CIRCUITING!**
pub fn or_2(left_boolean: Term, right_boolean: Term) -> Result {
    boolean_infix_operator!(left_boolean, right_boolean, |)
}

pub fn raise_3(class: Term, reason: Term, stacktrace: Term) -> Result {
    let class_class: Class = class.try_into()?;

    let runtime_exception = if stacktrace::is(stacktrace) {
        raise!(class_class, reason, Some(stacktrace)).into()
    } else {
        badarg!()
    };

    Err(runtime_exception.into())
}

pub fn read_timer_1(timer_reference: Term, process_control_block: &ProcessControlBlock) -> Result {
    read_timer(timer_reference, Default::default(), process_control_block)
}

pub fn read_timer_2(
    timer_reference: Term,
    options: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let read_timer_options: timer::read::Options = options.try_into()?;

    read_timer(timer_reference, read_timer_options, process_control_block)
}

pub fn register_2(
    name: Term,
    pid_or_port: Term,
    arc_process_control_block: Arc<ProcessControlBlock>,
) -> Result {
    let atom: Atom = name.try_into()?;

    let option_registered: Option<Term> = match atom.name() {
        "undefined" => None,
        _ => match pid_or_port.to_typed_term().unwrap() {
            TypedTerm::Pid(pid) => pid_to_self_or_process(pid, &arc_process_control_block)
                .and_then(|pid_arc_process_control_block| {
                    if registry::put_atom_to_process(atom, pid_arc_process_control_block) {
                        Some(true.into())
                    } else {
                        None
                    }
                }),
            _ => None,
        },
    };

    match option_registered {
        Some(registered) => Ok(registered),
        None => Err(badarg!().into()),
    }
}

pub fn registered_0(process_control_block: &ProcessControlBlock) -> Result {
    registry::names(process_control_block)
}

/// `rem/2` infix operator.  Integer remainder.
pub fn rem_2(dividend: Term, divisor: Term, process_control_block: &ProcessControlBlock) -> Result {
    integer_infix_operator!(dividend, divisor, process_control_block, %)
}

pub fn self_0(process_control_block: &ProcessControlBlock) -> Term {
    unsafe { process_control_block.pid().as_term() }
}

pub fn send_2(
    destination: Term,
    message: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    send(
        destination,
        message,
        Default::default(),
        process_control_block,
    )
    .map(|sent| match sent {
        Sent::Sent => message,
        _ => unreachable!(),
    })
}

// `send(destination, message, [nosuspend])` is used in `gen.erl`, which is used by `gen_server.erl`
// See https://github.com/erlang/otp/blob/8f6d45ddc8b2b12376c252a30b267a822cad171a/lib/stdlib/src/gen.erl#L167
pub fn send_3(
    destination: Term,
    message: Term,
    options: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let send_options: send::Options = options.try_into()?;

    send(destination, message, send_options, process_control_block)
        .map(|sent| match sent {
            Sent::Sent => "ok",
            Sent::ConnectRequired => "noconnect",
            Sent::SuspendRequired => "nosuspend",
        })
        .map(|s| atom_unchecked(s))
}

pub fn send_after_3(
    time: Term,
    destination: Term,
    message: Term,
    arc_process_control_block: Arc<ProcessControlBlock>,
) -> Result {
    start_timer(
        time,
        destination,
        Timeout::Message,
        message,
        Default::default(),
        arc_process_control_block,
    )
}

pub fn send_after_4(
    time: Term,
    destination: Term,
    message: Term,
    options: Term,
    arc_process_control_block: Arc<ProcessControlBlock>,
) -> Result {
    let timer_start_options: timer::start::Options = options.try_into()?;

    start_timer(
        time,
        destination,
        Timeout::Message,
        message,
        timer_start_options,
        arc_process_control_block,
    )
}

pub fn setelement_3(
    index: Term,
    tuple: Term,
    value: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let initial_inner_tuple: Boxed<Tuple> = tuple.try_into()?;
    let ZeroBasedIndex(index_zero_based): ZeroBasedIndex = index.try_into()?;

    let length = initial_inner_tuple.len();

    if index_zero_based < length {
        if index_zero_based == 0 {
            if 1 < length {
                process_control_block.tuple_from_slices(&[&[value], &initial_inner_tuple[1..]])
            } else {
                process_control_block.tuple_from_slice(&[value])
            }
        } else if index_zero_based < (length - 1) {
            process_control_block.tuple_from_slices(&[
                &initial_inner_tuple[..index_zero_based],
                &[value],
                &initial_inner_tuple[(index_zero_based + 1)..],
            ])
        } else {
            process_control_block
                .tuple_from_slices(&[&initial_inner_tuple[..index_zero_based], &[value]])
        }
        .map_err(|error| error.into())
    } else {
        Err(badarg!().into())
    }
}

pub fn size_1(binary_or_tuple: Term, process_control_block: &ProcessControlBlock) -> Result {
    let option_size = match binary_or_tuple.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::Tuple(tuple) => Some(tuple.len()),
            TypedTerm::HeapBinary(heap_binary) => Some(heap_binary.full_byte_len()),
            TypedTerm::ProcBin(process_binary) => Some(process_binary.full_byte_len()),
            TypedTerm::SubBinary(subbinary) => Some(subbinary.full_byte_len()),
            _ => None,
        },
        _ => None,
    };

    match option_size {
        Some(size) => Ok(process_control_block.integer(size)?),
        None => Err(badarg!().into()),
    }
}

pub fn spawn_3(
    module: Term,
    function: Term,
    arguments: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let module_atom: Atom = module.try_into()?;
    let function_atom: Atom = function.try_into()?;

    let option_pid = match arguments.to_typed_term().unwrap() {
        TypedTerm::Nil => {
            let (heap, heap_size) = default_heap()?;
            let arc_process = Scheduler::spawn(
                process_control_block,
                module_atom,
                function_atom,
                arguments,
                code::apply_fn(),
                heap,
                heap_size,
            )?;

            Some(arc_process.pid())
        }
        TypedTerm::List(cons) => {
            if cons.is_proper() {
                let (heap, heap_size) = default_heap()?;
                let arc_process = Scheduler::spawn(
                    process_control_block,
                    module_atom,
                    function_atom,
                    arguments,
                    code::apply_fn(),
                    heap,
                    heap_size,
                )?;

                Some(arc_process.pid())
            } else {
                None
            }
        }
        _ => None,
    };

    match option_pid {
        Some(pid) => Ok(unsafe { pid.as_term() }),
        None => Err(badarg!().into()),
    }
}

pub fn split_binary_2(
    binary: Term,
    position: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    let index: usize = position.try_into()?;

    match binary.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => {
            match boxed.to_typed_term().unwrap() {
                unboxed_typed_term @ TypedTerm::HeapBinary(_)
                | unboxed_typed_term @ TypedTerm::ProcBin(_) => {
                    if index == 0 {
                        let mut heap = process_control_block.acquire_heap();

                        let empty_prefix = heap.subbinary_from_original(binary, index, 0, 0, 0)?;

                        // Don't make a subbinary of the suffix since it is the same as the
                        // `binary`.
                        heap.tuple_from_slice(&[empty_prefix, binary])
                            .map_err(|error| error.into())
                    } else {
                        let full_byte_length = match unboxed_typed_term {
                            TypedTerm::HeapBinary(heap_binary) => heap_binary.full_byte_len(),
                            TypedTerm::ProcBin(process_binary) => process_binary.full_byte_len(),
                            _ => unreachable!(),
                        };

                        if index < full_byte_length {
                            let mut heap = process_control_block.acquire_heap();
                            let prefix = heap.subbinary_from_original(binary, 0, 0, index, 0)?;
                            let suffix = heap.subbinary_from_original(
                                binary,
                                index,
                                0,
                                full_byte_length - index,
                                0,
                            )?;

                            heap.tuple_from_slice(&[prefix, suffix])
                                .map_err(|error| error.into())
                        } else if index == full_byte_length {
                            let mut heap = process_control_block.acquire_heap();
                            let empty_suffix =
                                heap.subbinary_from_original(binary, index, 0, 0, 0)?;

                            // Don't make a subbinary of the prefix since it is the same as the
                            // `binary`.
                            heap.tuple_from_slice(&[binary, empty_suffix])
                                .map_err(|error| error.into())
                        } else {
                            Err(badarg!().into())
                        }
                    }
                }
                TypedTerm::SubBinary(subbinary) => {
                    if index == 0 {
                        let mut heap = process_control_block.acquire_heap();
                        let empty_prefix = heap.subbinary_from_original(
                            subbinary.original(),
                            subbinary.byte_offset() + index,
                            subbinary.bit_offset(),
                            0,
                            0,
                        )?;

                        // Don't make a subbinary of the suffix since it is the same as the
                        // `binary`.
                        heap.tuple_from_slice(&[empty_prefix, binary])
                            .map_err(|error| error.into())
                    } else {
                        // total_byte_length includes +1 byte if bits
                        let total_byte_length = subbinary.total_byte_len();

                        if index < total_byte_length {
                            let mut heap = process_control_block.acquire_heap();
                            let original = subbinary.original();
                            let byte_offset = subbinary.byte_offset();
                            let bit_offset = subbinary.bit_offset();

                            let prefix = heap.subbinary_from_original(
                                original,
                                byte_offset,
                                bit_offset,
                                index,
                                0,
                            )?;
                            let suffix = heap.subbinary_from_original(
                                original,
                                byte_offset + index,
                                bit_offset,
                                // full_byte_count does not include bits
                                subbinary.full_byte_len() - index,
                                subbinary.partial_byte_bit_len(),
                            )?;

                            heap.tuple_from_slice(&[prefix, suffix])
                                .map_err(|error| error.into())
                        } else if (index == total_byte_length)
                            & (subbinary.partial_byte_bit_len() == 0)
                        {
                            let mut heap = process_control_block.acquire_heap();
                            let empty_suffix = heap.subbinary_from_original(
                                subbinary.original(),
                                subbinary.byte_offset() + index,
                                subbinary.bit_offset(),
                                0,
                                0,
                            )?;

                            heap.tuple_from_slice(&[binary, empty_suffix])
                                .map_err(|error| error.into())
                        } else {
                            Err(badarg!().into())
                        }
                    }
                }
                _ => Err(badarg!().into()),
            }
        }
        _ => Err(badarg!().into()),
    }
}

pub fn start_timer_3(
    time: Term,
    destination: Term,
    message: Term,
    arc_process_control_block: Arc<ProcessControlBlock>,
) -> Result {
    start_timer(
        time,
        destination,
        Timeout::TimeoutTuple,
        message,
        Default::default(),
        arc_process_control_block,
    )
}

pub fn start_timer_4(
    time: Term,
    destination: Term,
    message: Term,
    options: Term,
    arc_process_control_block: Arc<ProcessControlBlock>,
) -> Result {
    let timer_start_options: timer::start::Options = options.try_into()?;

    start_timer(
        time,
        destination,
        Timeout::TimeoutTuple,
        message,
        timer_start_options,
        arc_process_control_block,
    )
}

/// `-/2` infix operator
pub fn subtract_2(
    minuend: Term,
    subtrahend: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    number_infix_operator!(minuend, subtrahend, process_control_block, checked_sub, -)
}

pub fn subtract_list_2(
    minuend: Term,
    subtrahend: Term,
    process_control_block: &ProcessControlBlock,
) -> Result {
    match (
        minuend.to_typed_term().unwrap(),
        subtrahend.to_typed_term().unwrap(),
    ) {
        (TypedTerm::Nil, TypedTerm::Nil) => Ok(minuend),
        (TypedTerm::Nil, TypedTerm::List(subtrahend_cons)) => {
            if subtrahend_cons.is_proper() {
                Ok(minuend)
            } else {
                Err(badarg!().into())
            }
        }
        (TypedTerm::List(minuend_cons), TypedTerm::Nil) => {
            if minuend_cons.is_proper() {
                Ok(minuend)
            } else {
                Err(badarg!().into())
            }
        }
        (TypedTerm::List(minuend_cons), TypedTerm::List(subtrahend_cons)) => {
            match minuend_cons
                .into_iter()
                .collect::<std::result::Result<Vec<Term>, _>>()
            {
                Ok(mut minuend_vec) => {
                    for result in subtrahend_cons.into_iter() {
                        match result {
                            Ok(subtrahend_element) => minuend_vec.remove_item(&subtrahend_element),
                            Err(ImproperList { .. }) => return Err(badarg!().into()),
                        };
                    }

                    process_control_block
                        .list_from_slice(&minuend_vec)
                        .map_err(|error| error.into())
                }
                Err(ImproperList { .. }) => Err(badarg!().into()),
            }
        }
        _ => Err(badarg!().into()),
    }
}

pub fn throw_1(reason: Term) -> Result {
    Err(throw!(reason).into())
}

pub fn tl_1(list: Term) -> Result {
    let cons: Boxed<Cons> = list.try_into()?;

    Ok(cons.tail)
}

pub fn tuple_size_1(tuple: Term, process_control_block: &ProcessControlBlock) -> Result {
    let tuple: Boxed<Tuple> = tuple.try_into()?;
    let size = process_control_block.integer(tuple.len())?;

    Ok(size)
}

pub fn tuple_to_list_1(tuple: Term, process_control_block: &ProcessControlBlock) -> Result {
    let tuple: Boxed<Tuple> = tuple.try_into()?;
    let mut heap = process_control_block.acquire_heap();
    let mut acc = Term::NIL;

    for element in tuple.iter().rev() {
        acc = heap.cons(element, acc)?;
    }

    Ok(acc)
}

pub fn unregister_1(name: Term) -> Result {
    let atom: Atom = name.try_into()?;

    if registry::unregister(&atom) {
        Ok(true.into())
    } else {
        Err(badarg!().into())
    }
}

pub fn whereis_1(name: Term) -> Result {
    let atom: Atom = name.try_into()?;

    let option = registry::atom_to_process(&atom)
        .map(|arc_process_control_block| arc_process_control_block.pid());

    let term = match option {
        Some(pid) => unsafe { pid.as_term() },
        None => atom_unchecked("undefined"),
    };

    Ok(term)
}

/// `xor/2` infix operator.
///
/// **NOTE: NOT SHORT-CIRCUITING!**
pub fn xor_2(left_boolean: Term, right_boolean: Term) -> Result {
    boolean_infix_operator!(left_boolean, right_boolean, ^)
}

// Private

fn cancel_timer(
    timer_reference: Term,
    options: timer::cancel::Options,
    process_control_block: &ProcessControlBlock,
) -> Result {
    match timer_reference.to_typed_term().unwrap() {
        TypedTerm::Boxed(unboxed_timer_reference) => {
            match unboxed_timer_reference.to_typed_term().unwrap() {
                TypedTerm::Reference(ref reference) => {
                    let canceled = timer::cancel(reference);

                    let term = if options.info {
                        let mut heap = process_control_block.acquire_heap();
                        let canceled_term = match canceled {
                            Some(milliseconds_remaining) => heap.integer(milliseconds_remaining)?,
                            None => false.into(),
                        };

                        if options.r#async {
                            let cancel_timer_message = heap.tuple_from_slice(&[
                                atom_unchecked("cancel_timer"),
                                timer_reference,
                                canceled_term,
                            ])?;
                            process_control_block.send_from_self(cancel_timer_message)?;

                            atom_unchecked("ok")
                        } else {
                            canceled_term
                        }
                    } else {
                        atom_unchecked("ok")
                    };

                    Ok(term)
                }
                _ => Err(badarg!().into()),
            }
        }
        _ => Err(badarg!().into()),
    }
}

fn is_record(term: Term, record_tag: Term, size: Option<Term>) -> Result {
    match term.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::Tuple(tuple) => {
                match record_tag.to_typed_term().unwrap() {
                    TypedTerm::Atom(_) => {
                        let len = tuple.len();

                        let tagged = if 0 < len {
                            let element = tuple[0];

                            match size {
                                Some(size_term) => {
                                    let size_usize: usize = size_term.try_into()?;

                                    (element == record_tag) & (len == size_usize)
                                }
                                None => element == record_tag,
                            }
                        } else {
                            // even if the `record_tag` cannot be checked, the `size` is still type
                            // checked
                            if let Some(size_term) = size {
                                let _: usize = size_term.try_into()?;
                            }

                            false
                        };

                        Ok(tagged.into())
                    }
                    _ => Err(badarg!().into()),
                }
            }
            _ => Ok(false.into()),
        },
        _ => Ok(false.into()),
    }
}

fn list_to_string(list: Term) -> std::result::Result<String, Exception> {
    match list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok("".to_owned()),
        TypedTerm::List(cons) => cons
            .into_iter()
            .map(|result| match result {
                Ok(term) => {
                    let string: String = term.try_into()?;

                    Ok(string)
                }
                Err(ImproperList { .. }) => Err(badarg!().into()),
            })
            .collect::<std::result::Result<String, Exception>>(),
        _ => Err(badarg!().into()),
    }
}

fn next_decimal(cons: Boxed<Cons>) -> std::result::Result<(usize, Term), Exception> {
    next_decimal_digit(cons)
        .and_then(|(first_digit, first_tail)| rest_decimal_digits(first_digit, first_tail))
}

fn next_decimal_digit(cons: Boxed<Cons>) -> std::result::Result<(u8, Term), Exception> {
    let head_char: char = cons.head.try_into()?;

    match head_char.to_digit(10) {
        Some(digit) => Ok((digit as u8, cons.tail)),
        None => Err(badarg!().into()),
    }
}

fn read_timer(
    timer_reference: Term,
    options: timer::read::Options,
    process_control_block: &ProcessControlBlock,
) -> Result {
    match timer_reference.to_typed_term().unwrap() {
        TypedTerm::Boxed(unboxed_timer_reference) => {
            match unboxed_timer_reference.to_typed_term().unwrap() {
                TypedTerm::Reference(ref local_reference) => {
                    let read = timer::read(local_reference);
                    let mut heap = process_control_block.acquire_heap();

                    let read_term = match read {
                        Some(milliseconds_remaining) => heap.integer(milliseconds_remaining)?,
                        None => false.into(),
                    };

                    let term = if options.r#async {
                        let read_timer_message = heap.tuple_from_slice(&[
                            atom_unchecked("read_timer"),
                            timer_reference,
                            read_term,
                        ])?;
                        process_control_block.send_from_self(read_timer_message)?;

                        atom_unchecked("ok")
                    } else {
                        read_term
                    };

                    Ok(term)
                }
                _ => Err(badarg!().into()),
            }
        }
        _ => Err(badarg!().into()),
    }
}

fn rest_decimal_digits(
    first_digit: u8,
    first_tail: Term,
) -> std::result::Result<(usize, Term), Exception> {
    match first_tail.try_into() {
        Ok(first_tail_cons) => {
            let mut acc_decimal: usize = first_digit as usize;
            let mut acc_tail = first_tail;
            let mut acc_cons: Boxed<Cons> = first_tail_cons;

            while let Ok((digit, tail)) = next_decimal_digit(acc_cons) {
                acc_decimal = 10 * acc_decimal + (digit as usize);
                acc_tail = tail;

                match tail.try_into() {
                    Ok(tail_cons) => acc_cons = tail_cons,
                    Err(_) => {
                        break;
                    }
                }
            }

            Ok((acc_decimal, acc_tail))
        }
        Err(_) => Ok((first_digit as usize, first_tail)),
    }
}

fn skip_char(cons: Boxed<Cons>, skip: char) -> Result {
    let c: char = cons.head.try_into()?;

    if c == skip {
        Ok(cons.tail)
    } else {
        Err(badarg!().into())
    }
}

fn start_timer(
    time: Term,
    destination: Term,
    timeout: Timeout,
    message: Term,
    options: timer::start::Options,
    arc_process_control_block: Arc<ProcessControlBlock>,
) -> Result {
    if time.is_integer() {
        let reference_frame_milliseconds: Milliseconds = time.try_into()?;

        let absolute_milliseconds = match options.reference_frame {
            ReferenceFrame::Relative => {
                monotonic::time_in_milliseconds() + reference_frame_milliseconds
            }
            ReferenceFrame::Absolute => reference_frame_milliseconds,
        };

        match destination.to_typed_term().unwrap() {
            // Registered names are looked up at time of send
            TypedTerm::Atom(destination_atom) => timer::start(
                absolute_milliseconds,
                timer::Destination::Name(destination_atom),
                timeout,
                message,
                &arc_process_control_block,
            )
            .map_err(|error| error.into()),
            // PIDs are looked up at time of create.  If they don't exist, they still return a
            // LocalReference.
            TypedTerm::Pid(destination_pid) => {
                match pid_to_self_or_process(destination_pid, &arc_process_control_block) {
                    Some(pid_arc_process_control_block) => timer::start(
                        absolute_milliseconds,
                        timer::Destination::Process(Arc::downgrade(&pid_arc_process_control_block)),
                        timeout,
                        message,
                        &arc_process_control_block,
                    )
                    .map_err(|error| error.into()),
                    None => make_ref_0(&arc_process_control_block),
                }
            }
            _ => Err(badarg!().into()),
        }
    } else {
        Err(badarg!().into())
    }
}
