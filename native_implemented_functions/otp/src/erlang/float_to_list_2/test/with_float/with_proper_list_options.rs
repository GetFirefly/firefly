mod with_decimals;
mod with_scientific;

use super::*;

#[test]
fn without_valid_option_errors_badarg() {
    crate::test::float_to_string::without_valid_option_errors_badarg(file!(), result);
}
