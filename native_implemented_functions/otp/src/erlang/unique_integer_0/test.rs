use crate::erlang::unique_integer_0::result;
use crate::test::with_process;

use liblumen_alloc::erts::term::prelude::Encoded;

#[test]
fn returns_non_monotonic_negative_and_positive_integer() {
    with_process(|process| {
        let result_first_unique_integer = result(process);

        assert!(result_first_unique_integer.is_ok());

        let first_unique_integer = result_first_unique_integer.unwrap();
        let zero = process.integer(0).unwrap();

        assert!(first_unique_integer.is_integer());
        assert!(first_unique_integer <= zero);

        let result_second_unique_integer = result(process);

        assert!(result_second_unique_integer.is_ok());

        let second_unique_integer = result_second_unique_integer.unwrap();

        assert!(second_unique_integer.is_integer());
        assert!(second_unique_integer <= zero);

        assert_ne!(first_unique_integer, second_unique_integer);
    });
}
