//! https://github.com/lumen/otp/tree/lumen/lib/snmp/src/manager

use super::*;

test_compiles_lumen_otp!(snmpm);
test_compiles_lumen_otp!(snmpm_conf);
test_compiles_lumen_otp!(snmpm_config);
test_compiles_lumen_otp!(snmpm_misc_sup);
test_compiles_lumen_otp!(snmpm_mpd);
test_compiles_lumen_otp!(snmpm_net_if);
test_compiles_lumen_otp!(snmpm_net_if_filter);
test_compiles_lumen_otp!(snmpm_net_if_mt);
test_compiles_lumen_otp!(snmpm_network_interface imports "lib/snmp/src/misc/snmp_misc");
test_compiles_lumen_otp!(snmpm_network_interface_filter imports "lib/snmp/src/misc/snmp_misc");
test_compiles_lumen_otp!(snmpm_server);
test_compiles_lumen_otp!(snmpm_server_sup);
test_compiles_lumen_otp!(snmpm_supervisor);
test_compiles_lumen_otp!(snmpm_user);
test_compiles_lumen_otp!(snmpm_user_default imports "lib/kernel/src/error_logger");
test_compiles_lumen_otp!(snmpm_user_old);
test_compiles_lumen_otp!(snmpm_usm);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/kernel/src/",
        "lib/snmp/include",
        "lib/snmp/src/compile",
        "lib/snmp/src/manager",
        "lib/snmp/src/misc",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("manager")
}

fn setup() {
    mibs::setup()
}
