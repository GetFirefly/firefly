//! https://github.com/lumen/otp/tree/lumen/lib/snmp/src/misc

use super::*;

test_compiles_lumen_otp!(snmp_conf);
test_compiles_lumen_otp!(snmp_config);
test_compiles_lumen_otp!(snmp_log);
test_compiles_lumen_otp!(snmp_mini_mib);
test_compiles_lumen_otp!(snmp_misc);
test_compiles_lumen_otp!(snmp_note_store);
test_compiles_lumen_otp!(snmp_pdus);
test_compiles_lumen_otp!(snmp_usm);
test_compiles_lumen_otp!(snmp_verbosity);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/snmp/include",
        "lib/snmp/src/compile",
        "lib/snmp/src/misc",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("misc")
}

fn setup() {
    mibs::setup()
}
