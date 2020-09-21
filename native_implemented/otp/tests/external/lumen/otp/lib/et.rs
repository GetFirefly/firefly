//! https://github.com/lumen/otp/tree/lumen/lib/et/src

use super::*;

test_compiles_lumen_otp!(et);
test_compiles_lumen_otp!(et_collector);
test_compiles_lumen_otp!(et_selector);
test_compiles_lumen_otp!(et_viewer);
test_compiles_lumen_otp!(et_wx_contents_viewer);
test_compiles_lumen_otp!(et_wx_viewer);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/et/src");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("et/src")
}
