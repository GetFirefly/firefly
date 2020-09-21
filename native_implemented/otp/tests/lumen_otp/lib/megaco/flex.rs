//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src/flex

use super::*;

test_compiles_lumen_otp!(megaco_flex_scanner);

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("flex")
}
