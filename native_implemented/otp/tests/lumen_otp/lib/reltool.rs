//! https://github.com/lumen/otp/tree/lumen/lib/reltool/src

use super::*;

test_compiles_lumen_otp!(reltool);
test_compiles_lumen_otp!(reltool_app_win);
test_compiles_lumen_otp!(reltool_fgraph);
test_compiles_lumen_otp!(reltool_fgraph_win);
test_compiles_lumen_otp!(reltool_mod_win);
test_compiles_lumen_otp!(reltool_server);
test_compiles_lumen_otp!(reltool_sys_win);
test_compiles_lumen_otp!(reltool_target);
test_compiles_lumen_otp!(reltool_utils);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/reltool/src");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("reltool/src")
}
