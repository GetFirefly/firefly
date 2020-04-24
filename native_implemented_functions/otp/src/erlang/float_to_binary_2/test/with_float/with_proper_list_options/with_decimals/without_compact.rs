use super::*;

#[test]
fn trailing_zeros_are_not_truncated() {
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
            Ok(arc_process.binary_from_str("12345.67890").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 6)),
            Ok(arc_process.binary_from_str("12345.678900").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 7)),
            Ok(arc_process.binary_from_str("12345.6789000").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 8)),
            Ok(arc_process.binary_from_str("12345.67890000").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 9)),
            Ok(arc_process.binary_from_str("12345.678900000").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 10)),
            Ok(arc_process.binary_from_str("12345.6789000000").unwrap())
        );
        assert_eq!(
            result(&arc_process, float, options(&arc_process, 11)),
            Ok(arc_process.binary_from_str("12345.67890000000").unwrap())
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

fn options(process: &Process, digits: u8) -> Term {
    process.list_from_slice(&[option(process, digits)]).unwrap()
}
