use proptest::strategy::Just;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::scheduler;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::binary_to_term_1;
use crate::erlang::term_to_binary_1::result;
use crate::test::strategy;
use crate::test::with_process;

#[test]
#[ignore]
fn roundtrips_through_binary_to_term() {
    run!(
        |arc_process| (Just(arc_process.clone()), strategy::term(arc_process)),
        |(arc_process, term)| {
            let result_binary = result(&arc_process, term);

            prop_assert!(result_binary.is_ok());

            let binary = result_binary.unwrap();

            prop_assert!(binary.is_binary());
            prop_assert_eq!(binary_to_term_1::result(&arc_process, binary), Ok(term));

            Ok(())
        },
    );
}

// NEW_FLOAT_EXT (70)
#[test]
fn with_negative_float_returns_new_float_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.float(std::f64::MIN).unwrap()),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    NEW_FLOAT_EXT,
                    255,
                    239,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255
                ])
                .unwrap())
        );
    });
}

// NEW_FLOAT_EXT (70)
#[test]
fn with_zero_float_returns_new_float_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.float(0.0).unwrap()),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    NEW_FLOAT_EXT,
                    0b0000_0000,
                    0b0000_0000,
                    0b0000_0000,
                    0b0000_0000,
                    0b0000_0000,
                    0b0000_0000,
                    0b0000_0000,
                    0b0000_0000
                ])
                .unwrap())
        );
    });
}

// NEW_FLOAT_EXT (70)
#[test]
fn with_positive_float_returns_new_float_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.float(std::f64::MAX).unwrap()),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    NEW_FLOAT_EXT,
                    127,
                    239,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255
                ])
                .unwrap())
        );
    });
}

// BIT_BINARY_EXT (77)
#[test]
fn with_subbinary_without_binary_with_aligned_returns_bit_binary_ext() {
    with_process(|process| {
        let binary = process
            .binary_from_bytes(&[0b1010_1010, 0b1010_1010])
            .unwrap();
        let subbinary = process.subbinary_from_original(binary, 0, 0, 1, 1).unwrap();

        let expected = process
            .binary_from_bytes(&[131, 77, 0, 0, 0, 2, 1, 0b1010_1010, 0b1000_0000])
            .unwrap();

        assert_eq!(result(process, subbinary), Ok(expected));
    });
}

// BIT_BINARY_EXT (77)
#[test]
fn with_subbinary_without_binary_without_aligned_returns_bit_binary_ext() {
    with_process(|process| {
        let binary = process
            .binary_from_bytes(&[0b1010_1010, 0b1010_1010])
            .unwrap();
        let subbinary = process.subbinary_from_original(binary, 0, 1, 1, 1).unwrap();

        assert_eq!(
            result(process, subbinary),
            Ok(process
                .binary_from_bytes(&[131, 77, 0, 0, 0, 2, 1, 0b10_10101, 0b0000_0000])
                .unwrap())
        );
    });
}

// NEWER_REFERENCE_EXT (90)
#[test]
fn with_reference_returns_new_reference_ext() {
    with_process(|process| {
        let scheduler_id: scheduler::ID = 1.into();
        let reference = Reference::new(scheduler_id, 2).encode().unwrap();

        assert_eq!(
            result(process, reference),
            Ok(process
                .binary_from_bytes(&[
                    131, 90, 0, 3, 100, 0, 13, 110, 111, 110, 111, 100, 101, 64, 110, 111, 104,
                    111, 115, 116, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2
                ])
                .unwrap())
        );
    });
}

// SMALL_INTEGER_EXT (97)
#[test]
fn with_unsigned_byte_small_integer_returns_small_integer_ext() {
    with_process(|process| {
        let small_integer_u8 = 0b1010_1010_u8;

        assert_eq!(
            result(process, process.integer(small_integer_u8).unwrap()),
            Ok(process
                .binary_from_bytes(&[VERSION_NUMBER, SMALL_INTEGER_EXT, small_integer_u8])
                .unwrap())
        );
    });
}

// INTEGER_EXT (98)
#[test]
fn with_negative_i32_small_integer_returns_integer_ext() {
    with_process(|process| {
        let small_integer_i32 = std::i32::MIN;

        assert_eq!(
            result(process, process.integer(small_integer_i32).unwrap()),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    INTEGER_EXT,
                    0b1000_0000,
                    0b0000_0000,
                    0b0000_0000,
                    0b0000_0000
                ])
                .unwrap())
        );
    });
}

// INTEGER_EXT (98)
#[test]
fn with_positive_i32_small_integer_returns_integer_ext() {
    with_process(|process| {
        let small_integer_i32 = std::i32::MAX;

        assert_eq!(
            result(process, process.integer(small_integer_i32).unwrap()),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    INTEGER_EXT,
                    0b0111_1111,
                    0b1111_1111,
                    0b1111_1111,
                    0b1111_1111
                ])
                .unwrap())
        );
    });
}

// ATOM_EXT (100)
#[test]
fn with_empty_atom_returns_atom_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, Atom::str_to_term("")),
            Ok(process
                .binary_from_bytes(&[VERSION_NUMBER, ATOM_EXT, 0, 0])
                .unwrap())
        );
    });
}

// ATOM_EXT (100)
#[test]
fn with_non_empty_atom_returns_atom_ext() {
    with_process(|process| {
        let mut byte_vec = Vec::new();
        byte_vec.push(VERSION_NUMBER);
        byte_vec.append(&mut non_empty_atom_byte_vec());

        assert_eq!(
            result(process, non_empty_atom_term()),
            Ok(process.binary_from_bytes(&byte_vec).unwrap())
        );
    });
}

// PID_EXT (103)
#[test]
fn with_pid_returns_pid_ext() {
    with_process(|process| {
        let pid = Pid::new(1, 2).unwrap().encode().unwrap();

        assert_eq!(
            result(process, pid),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    PID_EXT,
                    100,
                    0,
                    13,
                    110,
                    111,
                    110,
                    111,
                    100,
                    101,
                    64,
                    110,
                    111,
                    104,
                    111,
                    115,
                    116,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    2,
                    0
                ])
                .unwrap())
        )
    });
}

// SMALL_TUPLE_EXT (104)
#[test]
fn with_empty_tuple_returns_small_tuple_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.tuple_from_slice(&[]).unwrap()),
            Ok(process
                .binary_from_bytes(&[VERSION_NUMBER, SMALL_TUPLE_EXT, 0])
                .unwrap())
        );
    });
}

// SMALL_TUPLE_EXT (104)
#[test]
fn with_non_empty_tuple_returns_small_tuple_ext() {
    with_process(|process| {
        let mut byte_vec = Vec::new();
        byte_vec.push(VERSION_NUMBER);
        byte_vec.append(&mut non_empty_tuple_byte_vec());

        assert_eq!(
            result(process, non_empty_tuple_term(process)),
            Ok(process.binary_from_bytes(&byte_vec).unwrap())
        );
    });
}

// NIL_EXT (106)
#[test]
fn with_empty_list_returns_nil_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, Term::NIL),
            Ok(process
                .binary_from_bytes(&[VERSION_NUMBER, NIL_EXT])
                .unwrap())
        );
    });
}

// STRING_EXT (106)
#[test]
fn with_ascii_charlist_returns_string_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.charlist_from_str("string").unwrap()),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    STRING_EXT,
                    0,
                    6,
                    115,
                    116,
                    114,
                    105,
                    110,
                    103
                ])
                .unwrap())
        );
    })
}

// LIST_EXT (108)
#[test]
fn with_improper_list_returns_list_ext() {
    with_process(|process| {
        assert_eq!(
            result(
                process,
                process
                    .improper_list_from_slice(&[Atom::str_to_term("hd")], Atom::str_to_term("tl"))
                    .unwrap()
            ),
            Ok(process
                .binary_from_bytes(&[
                    131, 108, 0, 0, 0, 1, 100, 0, 2, 104, 100, 100, 0, 2, 116, 108
                ])
                .unwrap())
        )
    });
}

// LIST_EXT (108)
#[test]
fn with_proper_list_returns_list_ext() {
    with_process(|process| {
        assert_eq!(
            result(
                process,
                process
                    .list_from_slice(&[process.integer(256).unwrap()])
                    .unwrap()
            ),
            Ok(process
                .binary_from_bytes(&[131, 108, 0, 0, 0, 1, 98, 0, 0, 1, 0, 106])
                .unwrap())
        )
    });
}

// LIST_EXT (108)
#[test]
fn with_nested_list_returns_list_ext() {
    with_process(|process| {
        assert_eq!(
            result(
                process,
                process
                    .list_from_slice(&[
                        process
                            .list_from_slice(&[
                                process.integer(1100).unwrap(),
                                process.integer(1200).unwrap()
                            ])
                            .unwrap(),
                        process
                            .list_from_slice(&[
                                process.integer(2100).unwrap(),
                                process.integer(2200).unwrap()
                            ])
                            .unwrap()
                    ])
                    .unwrap()
            ),
            Ok(process
                .binary_from_bytes(&[
                    131, 108, 0, 0, 0, 2, 108, 0, 0, 0, 2, 98, 0, 0, 4, 76, 98, 0, 0, 4, 176, 106,
                    108, 0, 0, 0, 2, 98, 0, 0, 8, 52, 98, 0, 0, 8, 152, 106, 106
                ])
                .unwrap())
        )
    });
}

// BINARY_EXT (109)
#[test]
fn with_empty_heap_binary_returns_binary_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.binary_from_bytes(&[]).unwrap()),
            Ok(process
                .binary_from_bytes(&[VERSION_NUMBER, BINARY_EXT, 0, 0, 0, 0])
                .unwrap())
        );
    });
}

// BINARY_EXT (109)
#[test]
fn with_non_empty_heap_binary_returns_binary_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.binary_from_bytes(&[1, 2, 3]).unwrap()),
            Ok(process
                .binary_from_bytes(&[VERSION_NUMBER, BINARY_EXT, 0, 0, 0, 3, 1, 2, 3])
                .unwrap())
        )
    })
}

// BINARY_EXT (109)
#[test]
fn with_proc_bin_returns_binary_ext() {
    with_process(|process| {
        assert_eq!(
            result(
                process,
                process
                    .binary_from_bytes(&[
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63, 64
                    ])
                    .unwrap()
            ),
            Ok(process
                .binary_from_bytes(&[
                    VERSION_NUMBER,
                    BINARY_EXT,
                    0,
                    0,
                    0,
                    65,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64
                ])
                .unwrap())
        );
    });
}

// BINARY_EXT (109)
#[test]
fn with_subbinary_with_binary_with_aligned_returns_binary_ext() {
    with_process(|process| {
        let binary = process
            .binary_from_bytes(&[0b1010_1010, 0b1010_1010])
            .unwrap();
        let subbinary = process.subbinary_from_original(binary, 0, 0, 1, 0).unwrap();

        assert_eq!(
            result(process, subbinary),
            Ok(process
                .binary_from_bytes(&[131, 109, 0, 0, 0, 1, 0b1010_1010])
                .unwrap())
        );
    });
}

// BINARY_EXT (109)
#[test]
fn with_subbinary_with_binary_without_aligned_returns_binary_ext() {
    with_process(|process| {
        let binary = process
            .binary_from_bytes(&[0b1010_1010, 0b1010_1010])
            .unwrap();
        let subbinary = process.subbinary_from_original(binary, 0, 1, 1, 0).unwrap();

        assert_eq!(
            result(process, subbinary),
            Ok(process
                .binary_from_bytes(&[131, 109, 0, 0, 0, 1, 0b101_0101])
                .unwrap())
        );
    });
}

// SMALL_BIG_EXT (110)
#[test]
fn with_small_integer_returns_small_big_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.integer(9_999_999_999_i64).unwrap()),
            Ok(process
                .binary_from_bytes(&[131, 110, 5, 0, 255, 227, 11, 84, 2])
                .unwrap())
        )
    })
}

// SMALL_BIG_EXT (110)
#[test]
fn with_big_integer_returns_small_big_ext() {
    with_process(|process| {
        let big_integer = process.integer(9_223_372_036_854_775_807_i64).unwrap();

        assert!(big_integer.is_boxed_bigint());

        assert_eq!(
            result(process, big_integer),
            Ok(process
                .binary_from_bytes(&[131, 110, 8, 0, 255, 255, 255, 255, 255, 255, 255, 127])
                .unwrap())
        )
    })
}

// MAP_EXT (116)
#[test]
fn empty_map_returns_map_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, process.map_from_slice(&[]).unwrap()),
            Ok(process.binary_from_bytes(&[131, 116, 0, 0, 0, 0]).unwrap())
        )
    })
}

// MAP_EXT (116)
#[test]
fn non_empty_map_returns_map_ext() {
    with_process(|process| {
        assert_eq!(
            result(
                process,
                process
                    .map_from_slice(&[(
                        Atom::str_to_term("k"),
                        process
                            .map_from_slice(&[(Atom::str_to_term("v_k"), Atom::str_to_term("v_v"))])
                            .unwrap()
                    )])
                    .unwrap()
            ),
            Ok(process
                .binary_from_bytes(&[
                    131, 116, 0, 0, 0, 1, 100, 0, 1, 107, 116, 0, 0, 0, 1, 100, 0, 3, 118, 95, 107,
                    100, 0, 3, 118, 95, 118
                ])
                .unwrap())
        );
    });
}

// SMALL_ATOM_UTF8_EXT (119)
#[test]
fn with_small_utf8_atom_returns_small_atom_utf8_ext() {
    with_process(|process| {
        assert_eq!(
            result(process, Atom::str_to_term("😈")),
            Ok(process
                .binary_from_bytes(&[131, 119, 4, 240, 159, 152, 136])
                .unwrap())
        )
    });
}

const VERSION_NUMBER: u8 = 131;

const NEW_FLOAT_EXT: u8 = 70;
const SMALL_INTEGER_EXT: u8 = 97;
const INTEGER_EXT: u8 = 98;
const ATOM_EXT: u8 = 100;
const PID_EXT: u8 = 103;
const SMALL_TUPLE_EXT: u8 = 104;
const NIL_EXT: u8 = 106;
const STRING_EXT: u8 = 107;
const BINARY_EXT: u8 = 109;

fn non_empty_atom_term() -> Term {
    Atom::str_to_term("atom")
}

fn non_empty_atom_byte_vec() -> Vec<u8> {
    vec![ATOM_EXT, 0, 4, 97, 116, 111, 109]
}

fn non_empty_tuple_term(process: &Process) -> Term {
    process.tuple_from_slice(&[non_empty_atom_term()]).unwrap()
}

fn non_empty_tuple_byte_vec() -> Vec<u8> {
    let arity = 1;

    let mut byte_vec = vec![SMALL_TUPLE_EXT, arity];
    byte_vec.append(&mut non_empty_atom_byte_vec());

    byte_vec
}
