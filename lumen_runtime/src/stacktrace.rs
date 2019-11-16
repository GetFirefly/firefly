use core::convert::TryInto;
use core::result::Result;

use num_bigint::BigInt;

use liblumen_alloc::erts::term::prelude::*;

pub fn is(term: Term) -> bool {
    match term.decode().unwrap() {
        TypedTerm::Nil => true,
        TypedTerm::List(cons) => cons.into_iter().all(|result| match result {
            Ok(term) => term_is_item(term),
            Err(_) => false,
        }),
        _ => false,
    }
}

fn term_is_location(term: Term) -> bool {
    match term.decode().unwrap() {
        TypedTerm::Nil => true,
        TypedTerm::List(cons) => cons.into_iter().all(|result| match result {
            Ok(term) => term_is_location_keyword_pair(term),
            Err(_) => false,
        }),
        _ => false,
    }
}

fn term_is_location_keyword_pair(term: Term) -> bool {
    let result: Result<Boxed<Tuple>, _> = term.try_into();

    match result {
        Ok(tuple) => tuple_is_location_keyword_pair(tuple),
        Err(_) => false,
    }
}

fn tuple_is_location_keyword_pair(tuple: Boxed<Tuple>) -> bool {
    (tuple.len() == 2) && {
        let atom_result: Result<Atom, _> = tuple[0].try_into();

        match atom_result {
            Ok(atom) => match atom.name() {
                "file" => is_file(tuple[1]),
                "line" => is_line(tuple[1]),
                _ => false,
            },
            Err(_) => false,
        }
    }
}

fn is_file(term: Term) -> bool {
    is_charlist(term)
}

fn is_charlist(term: Term) -> bool {
    match term.try_into() {
        Ok(list) => match list {
            List::Empty => true,
            List::NonEmpty(cons) => cons_is_charlist(cons),
        },
        Err(_) => false,
    }
}

fn cons_is_charlist(cons: Boxed<Cons>) -> bool {
    cons.into_iter().all(|result| match result {
        Ok(term) => {
            let result: Result<char, _> = term.try_into();
            result.is_ok()
        }
        Err(_) => false,
    })
}

fn is_line(term: Term) -> bool {
    match term.decode().unwrap() {
        TypedTerm::SmallInteger(small_integer) => 0_isize < small_integer.into(),
        TypedTerm::BigInteger(big_integer) => {
            let big_int: &BigInt = big_integer.as_ref().into();
            let zero_big_int: &BigInt = &1.into();

            zero_big_int < big_int
        }
        _ => false,
    }
}

fn term_is_item(term: Term) -> bool {
    match term.try_into() {
        Ok(tuple) => tuple_is_item(tuple),
        Err(_) => false,
    }
}

fn tuple_is_item(tuple: Boxed<Tuple>) -> bool {
    match tuple.len() {
        // {function, args}
        // https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1107-L1114
        2 => tuple[0].is_boxed_function(),
        // https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1115-L1128
        3 => {
            let first_element = tuple[0];

            match first_element.decode().unwrap() {
                // {M, F, arity | args}
                TypedTerm::Atom(_) => tuple[1].is_atom() && is_arity_or_arguments(tuple[2]),
                // {function, args, location}
                TypedTerm::Closure(_) => term_is_location(tuple[2]),
                _ => false,
            }
        }
        // https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1129-L1134
        4 => {
            // {M, F, arity | args, location}
            tuple[0].is_atom()
                & tuple[1].is_atom()
                & is_arity_or_arguments(tuple[2])
                & term_is_location(tuple[3])
        }
        _ => false,
    }
}

fn is_arity_or_arguments(term: Term) -> bool {
    match term.decode().unwrap() {
        // args
        TypedTerm::Nil | TypedTerm::List(_) => true,
        // arity
        TypedTerm::SmallInteger(small_integer) => {
            let arity: isize = small_integer.into();

            0 <= arity
        }
        // arity
        TypedTerm::BigInteger(big_integer) => {
            let big_int = big_integer.as_ref().into();
            let zero_big_int: &BigInt = &0.into();

            zero_big_int <= big_int
        }
        _ => false,
    }
}
