//! https://github.com/lumen/otp/tree/lumen/lib/snmp/src/compile

use super::*;

test_compiles_lumen_otp!(snmpc);
test_compiles_lumen_otp!(snmpc_lib);
test_compiles_lumen_otp!(snmpc_mib_to_hrl);
test_compiles_lumen_otp!(snmpc_misc);
test_compiles_lumen_otp!(snmpc_tok);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/snmp/include", "lib/snmp/src/compile"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("compile")
}
