//! https://github.com/lumen/otp/tree/lumen/lib/diameter/src/compiler

use super::*;

test_compiles_lumen_otp!(diameter_codegen);
test_compiles_lumen_otp!(diameter_dict_scanner);
test_compiles_lumen_otp!(diameter_dict_util);
test_compiles_lumen_otp!(diameter_exprecs);
test_compiles_lumen_otp!(diameter_make);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/diameter/src/compiler");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("compiler")
}
