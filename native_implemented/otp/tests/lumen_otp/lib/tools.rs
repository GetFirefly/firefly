//! https://github.com/lumen/otp/tree/lumen/lib/tools/src

use super::*;

test_compiles_lumen_otp!(cover);
test_compiles_lumen_otp!(cprof);
test_compiles_lumen_otp!(eprof);
test_compiles_lumen_otp!(fprof);
test_compiles_lumen_otp!(instrument);
test_compiles_lumen_otp!(lcnt);
test_compiles_lumen_otp!(make);
test_compiles_lumen_otp!(tags);
test_compiles_lumen_otp!(xref);
test_compiles_lumen_otp!(xref_base);
test_compiles_lumen_otp!(xref_compiler);
test_compiles_lumen_otp!(xref_reader);
test_compiles_lumen_otp!(xref_scanner);
test_compiles_lumen_otp!(xref_utils);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/tools/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("tools/src")
}
