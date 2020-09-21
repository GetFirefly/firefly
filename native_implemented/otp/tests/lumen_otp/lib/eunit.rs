//! https://github.com/lumen/otp/tree/lumen/lib/eunit/src

use super::*;

test_compiles_lumen_otp!(eunit);
test_compiles_lumen_otp!(eunit_autoexport);
test_compiles_lumen_otp!(eunit_data);
test_compiles_lumen_otp!(eunit_lib);
test_compiles_lumen_otp!(eunit_listener);
test_compiles_lumen_otp!(eunit_proc);
test_compiles_lumen_otp!(eunit_serial);
test_compiles_lumen_otp!(eunit_server);
test_compiles_lumen_otp!(eunit_striptests);
test_compiles_lumen_otp!(eunit_surefire);
test_compiles_lumen_otp!(eunit_test);
test_compiles_lumen_otp!(eunit_tests);
test_compiles_lumen_otp!(eunit_tty);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/eunit/include", "lib/eunit/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("eunit/src")
}
