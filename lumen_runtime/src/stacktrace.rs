use num_bigint::BigInt;

use crate::integer::big;
use crate::list::Cons;
use crate::term::{Tag::*, Term};
use crate::tuple::Tuple;

pub fn is(term: Term) -> bool {
    match term.tag() {
        EmptyList => true,
        List => {
            let cons: &Cons = unsafe { term.as_ref_cons_unchecked() };

            cons.into_iter().all(|result| match result {
                Ok(term) => term_is_item(term),
                Err(_) => false,
            })
        }
        _ => false,
    }
}

fn term_is_location(term: Term) -> bool {
    match term.tag() {
        EmptyList => true,
        List => {
            let cons: &Cons = unsafe { term.as_ref_cons_unchecked() };

            cons.into_iter().all(|result| match result {
                Ok(term) => term_is_location_keyword_pair(term),
                Err(_) => false,
            })
        }
        _ => false,
    }
}

fn term_is_location_keyword_pair(term: Term) -> bool {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = term.unbox_reference();

                    tuple_is_location_keyword_pair(tuple)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn tuple_is_location_keyword_pair(tuple: &Tuple) -> bool {
    (tuple.len() == 2) && {
        let first_element = tuple[0];

        match first_element.tag() {
            Atom => match unsafe { first_element.atom_to_string() }.as_ref().as_ref() {
                "file" => is_file(tuple[1]),
                "line" => is_line(tuple[1]),
                _ => false,
            },
            _ => false,
        }
    }
}

fn is_file(term: Term) -> bool {
    term.is_char_list()
}

fn is_line(term: Term) -> bool {
    match term.tag() {
        SmallInteger => 0 < unsafe { term.small_integer_to_isize() },
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                BigInteger => {
                    let big_integer: &big::Integer = term.unbox_reference();
                    let big_int = &big_integer.inner;
                    let zero_big_int: &BigInt = &0.into();

                    zero_big_int < big_int
                }
                _ => false,
            }
        }

        _ => false,
    }
}

fn term_is_item(term: Term) -> bool {
    match term.tag() {
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                Arity => {
                    let tuple: &Tuple = term.unbox_reference();

                    tuple_is_item(tuple)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn tuple_is_item(tuple: &Tuple) -> bool {
    match tuple.len() {
        // {function, args}
        // https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1107-L1114
        2 => tuple[0].is_function(),
        // https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1115-L1128
        3 => {
            let first_element = tuple[0];

            match first_element.tag() {
                // {M, F, arity | args}
                Atom => tuple[1].is_atom() && is_arity_or_arguments(tuple[2]),
                // {function, args, location}
                Boxed => {
                    let unboxed: &Term = first_element.unbox_reference();

                    (unboxed.tag() == Function) && term_is_location(tuple[2])
                }
                _ => false,
            }
        }
        // https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1129-L1134
        4 => {
            // {M, F, arity | args, location}
            tuple[0].is_atom()
                && tuple[1].is_atom()
                && is_arity_or_arguments(tuple[2])
                && term_is_location(tuple[3])
        }
        _ => false,
    }
}

fn is_arity_or_arguments(term: Term) -> bool {
    match term.tag() {
        // args
        EmptyList | List => true,
        // arity
        SmallInteger => {
            let arity = unsafe { term.small_integer_to_isize() };

            0 <= arity
        }
        // arity
        Boxed => {
            let unboxed: &Term = term.unbox_reference();

            match unboxed.tag() {
                BigInteger => {
                    let big_integer: &big::Integer = term.unbox_reference();
                    let big_int = &big_integer.inner;
                    let zero_big_int: &BigInt = &0.into();

                    zero_big_int <= big_int
                }
                _ => false,
            }
        }
        _ => false,
    }
}
