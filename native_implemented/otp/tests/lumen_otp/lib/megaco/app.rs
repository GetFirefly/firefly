//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src/app

use super::*;

test_compiles_lumen_otp!(megaco);

fn includes() -> Vec<&'static str> {
    let includes = super::includes();
    // includes.push("lib/inets/src/http_client");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("app")
}
