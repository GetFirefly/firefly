//! https://github.com/lumen/otp/tree/lumen/lib/syntax_tools/src

use super::*;

test_compiles_lumen_otp!(epp_dodger);
test_compiles_lumen_otp!(erl_comment_scan);
test_compiles_lumen_otp!(erl_prettypr);
test_compiles_lumen_otp!(erl_recomment);
test_compiles_lumen_otp!(erl_syntax);
test_compiles_lumen_otp!(erl_syntax_lib);
test_compiles_lumen_otp!(erl_tidy);
test_compiles_lumen_otp!(igor);
test_compiles_lumen_otp!(merl);
test_compiles_lumen_otp!(merl_tests);
test_compiles_lumen_otp!(merl_transform);
test_compiles_lumen_otp!(prettypr);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/syntax_tools/include"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("syntax_tools/src")
}
