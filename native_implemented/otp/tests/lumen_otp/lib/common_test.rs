//! https://github.com/lumen/otp/tree/lumen/lib/common_test/src

use super::*;

test_compiles_lumen_otp!(ct);
test_compiles_lumen_otp!(ct_config);
test_compiles_lumen_otp!(ct_config_plain);
test_compiles_lumen_otp!(ct_config_xml);
test_compiles_lumen_otp!(ct_conn_log_h);
test_compiles_lumen_otp!(ct_cover);
test_compiles_lumen_otp!(ct_default_gl);
test_compiles_lumen_otp!(ct_event imports "lib/common_test/src/ct_master");
test_compiles_lumen_otp!(ct_framework);
test_compiles_lumen_otp!(ct_ftp);
test_compiles_lumen_otp!(ct_gen_conn);
test_compiles_lumen_otp!(ct_groups);
test_compiles_lumen_otp!(ct_hooks);
test_compiles_lumen_otp!(ct_hooks_lock);
test_compiles_lumen_otp!(ct_logs);
test_compiles_lumen_otp!(ct_make);
test_compiles_lumen_otp!(ct_master);
test_compiles_lumen_otp!(ct_master_event);
test_compiles_lumen_otp!(ct_master_logs);
test_compiles_lumen_otp!(ct_master_status);
test_compiles_lumen_otp!(ct_netconfc);
test_compiles_lumen_otp!(ct_property_test);
test_compiles_lumen_otp!(ct_release_test);
test_compiles_lumen_otp!(ct_repeat);
test_compiles_lumen_otp!(ct_rpc imports "lib/common_test/src/ct", "lib/kernel/src/rpc", "lib/stdlib/src/lists");
test_compiles_lumen_otp!(ct_run);
test_compiles_lumen_otp!(ct_slave);
test_compiles_lumen_otp!(ct_snmp);
test_compiles_lumen_otp!(ct_ssh);
test_compiles_lumen_otp!(ct_telnet);
test_compiles_lumen_otp!(ct_telnet_client);
test_compiles_lumen_otp!(ct_testspec);
test_compiles_lumen_otp!(ct_util);
test_compiles_lumen_otp!(cth_conn_log);
test_compiles_lumen_otp!(cth_log_redirect);
test_compiles_lumen_otp!(cth_surefire);
test_compiles_lumen_otp!(erl2html2);
test_compiles_lumen_otp!(test_server);
test_compiles_lumen_otp!(test_server_ctrl);
test_compiles_lumen_otp!(test_server_gl);
test_compiles_lumen_otp!(test_server_io);
test_compiles_lumen_otp!(test_server_node);
test_compiles_lumen_otp!(test_server_sup);
test_compiles_lumen_otp!(unix_telnet);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/common_test/include",
        "lib/common_test/src",
        "lib/kernel/include",
        "lib/snmp/include",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("common_test/src")
}
