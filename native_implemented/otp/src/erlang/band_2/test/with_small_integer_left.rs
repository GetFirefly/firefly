use super::*;

// No using `proptest` because I'd rather cover truth table manually
#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|process| {
        // all combinations of `0` and `1` bit.
        let left = process.integer(0b1100).unwrap();
        let right = process.integer(0b1010).unwrap();

        assert_eq!(
            result(&process, left, right),
            Ok(process.integer(0b1000).unwrap())
        );
    })
}

#[test]
fn with_integer_right_returns_bitwise_and() {
    super::with_integer_right_returns_bitwise_and(file!(), strategy::term::integer::small);
}
