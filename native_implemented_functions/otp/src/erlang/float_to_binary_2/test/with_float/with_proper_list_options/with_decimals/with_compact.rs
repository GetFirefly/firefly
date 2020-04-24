use super::*;

use liblumen_alloc::erts::term::prelude::Atom;

#[test]
fn trailing_zeros_are_truncated() {
    with_process_arc(|arc_process| {
        let float = arc_process.float(12345.6789).unwrap();

        assert_eq!(
            result(&arc_process, float, options(&arc_process, 0)),
            Ok(arc_process.binary_from_str("12346").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 1)),
            Ok(arc_process.binary_from_str("12345.7").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 2)),
            Ok(arc_process.binary_from_str("12345.68").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 3)),
            Ok(arc_process.binary_from_str("12345.679").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 4)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 5)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 6)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 7)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 8)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 9)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 10)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 11)),
            Ok(arc_process.binary_from_str("12345.6789").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 12)),
            Ok(arc_process.binary_from_str("12345.678900000001").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 13)),
            Ok(arc_process.binary_from_str("12345.6789000000008").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 14)),
            Ok(arc_process.binary_from_str("12345.67890000000079").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 15)),
            Ok(arc_process
                .binary_from_str("12345.678900000000795")
                .unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 16)),
            Ok(arc_process
                .binary_from_str("12345.6789000000007945")
                .unwrap())
        );
        // BEAM and Rust differ after this many digits
    });
}

#[test]
fn with_no_fractional_part_still_has_zero_after_decimal_point() {
    with_process_arc(|arc_process| {
        let float = arc_process.float(1.0).unwrap();

        assert_eq!(
            result(&arc_process, float, options(&arc_process, 2)),
            Ok(arc_process.binary_from_str("1.0").unwrap())
        );
    });
}

fn options(process: &Process, digits: u8) -> Term {
    process
        .list_from_slice(&[option(process, digits), Atom::str_to_term("compact")])
        .unwrap()
}
