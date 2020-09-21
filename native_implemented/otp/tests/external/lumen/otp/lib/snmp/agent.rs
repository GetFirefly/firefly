//! https://github.com/lumen/otp/tree/lumen/lib/snmp/src/agent

use super::*;

test_compiles_lumen_otp!(snmp_community_mib);
test_compiles_lumen_otp!(snmp_framework_mib);
test_compiles_lumen_otp!(snmp_generic);
test_compiles_lumen_otp!(snmp_generic_mnesia);
test_compiles_lumen_otp!(snmp_index);
test_compiles_lumen_otp!(snmp_notification_mib);
test_compiles_lumen_otp!(snmp_shadow_table);
test_compiles_lumen_otp!(snmp_standard_mib);
test_compiles_lumen_otp!(snmp_target_mib);
test_compiles_lumen_otp!(snmp_user_based_sm_mib);
test_compiles_lumen_otp!(snmp_view_based_acm_mib);
test_compiles_lumen_otp!(snmpa);
test_compiles_lumen_otp!(snmpa_acm);
test_compiles_lumen_otp!(snmpa_agent);
test_compiles_lumen_otp!(snmpa_agent_sup);
test_compiles_lumen_otp!(snmpa_app);
test_compiles_lumen_otp!(snmpa_authentication_service);
test_compiles_lumen_otp!(snmpa_conf);
test_compiles_lumen_otp!(snmpa_discovery_handler imports "lib/snmp/src/misc/snmp_misc");
test_compiles_lumen_otp!(snmpa_discovery_handler_default);
test_compiles_lumen_otp!(snmpa_error);
test_compiles_lumen_otp!(snmpa_error_io);
test_compiles_lumen_otp!(snmpa_error_logger);
test_compiles_lumen_otp!(snmpa_error_report);
test_compiles_lumen_otp!(snmpa_general_db);
test_compiles_lumen_otp!(snmpa_get);
test_compiles_lumen_otp!(snmpa_get_lib);
test_compiles_lumen_otp!(snmpa_get_mechanism);
test_compiles_lumen_otp!(snmpa_local_db);
test_compiles_lumen_otp!(snmpa_mib);
test_compiles_lumen_otp!(snmpa_mib_data);
test_compiles_lumen_otp!(snmpa_mib_data_ttln);
test_compiles_lumen_otp!(snmpa_mib_data_tttn);
test_compiles_lumen_otp!(snmpa_mib_lib);
test_compiles_lumen_otp!(snmpa_mib_storage);
test_compiles_lumen_otp!(snmpa_mib_storage_dets);
test_compiles_lumen_otp!(snmpa_mib_storage_ets);
test_compiles_lumen_otp!(snmpa_mib_storage_mnesia);
test_compiles_lumen_otp!(snmpa_misc_sup);
test_compiles_lumen_otp!(snmpa_mpd);
test_compiles_lumen_otp!(snmpa_net_if);
test_compiles_lumen_otp!(snmpa_net_if_filter);
test_compiles_lumen_otp!(snmpa_network_interface);
test_compiles_lumen_otp!(snmpa_network_interface_filter imports "lib/snmp/src/misc/snmp_misc");
test_compiles_lumen_otp!(snmpa_notification_delivery_info_receiver imports "lib/snmp/src/misc/snmp_misc");
test_compiles_lumen_otp!(snmpa_notification_filter);
test_compiles_lumen_otp!(snmpa_set);
test_compiles_lumen_otp!(snmpa_set_lib);
test_compiles_lumen_otp!(snmpa_set_mechanism);
test_compiles_lumen_otp!(snmpa_supervisor);
test_compiles_lumen_otp!(snmpa_svbl);
test_compiles_lumen_otp!(snmpa_symbolic_store);
test_compiles_lumen_otp!(snmpa_target_cache);
test_compiles_lumen_otp!(snmpa_trap);
test_compiles_lumen_otp!(snmpa_usm);
test_compiles_lumen_otp!(snmpa_vacm);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/snmp/include",
        "lib/snmp/src/agent",
        "lib/snmp/src/compile",
        "lib/snmp/src/misc",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("agent")
}

fn setup() {
    mibs::setup()
}
