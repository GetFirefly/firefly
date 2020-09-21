//! https://github.com/lumen/otp/tree/lumen/lib/odbc/src

use super::*;

test_compiles_lumen_otp!(odbc);
test_compiles_lumen_otp!(odbc_app imports "lib/stdlib/src/supervisor");
test_compiles_lumen_otp!(odbc_debug);
test_compiles_lumen_otp!(odbc_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/kernel/include");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("odbc/src")
}
