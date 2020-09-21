//! https://github.com/lumen/otp/tree/lumen/lib/snmp/src/app

use super::*;

test_compiles_lumen_otp!(snmp);
test_compiles_lumen_otp!(snmp_app imports "lib/kernel/src/application", "lib/snmp/src/app/snmp_app_sup", "lib/kernel/src/error_logger", "lib/stdlib/src/lists");
test_compiles_lumen_otp!(snmp_app_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/kernel/src", "lib/snmp/src/misc"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("app")
}
