//! https://github.com/lumen/otp/tree/lumen/lib/diameter/src/base

use super::*;

test_compiles_lumen_otp!(diameter);
test_compiles_lumen_otp!(diameter_app imports "lib/diameter/src/base/diameter_sup");
test_compiles_lumen_otp!(diameter_callback imports "lib/diameter/src/base/diameter_lib");
test_compiles_lumen_otp!(diameter_capx);
test_compiles_lumen_otp!(diameter_codec);
test_compiles_lumen_otp!(diameter_config);
test_compiles_lumen_otp!(diameter_config_sup);
test_compiles_lumen_otp!(diameter_dist);
test_compiles_lumen_otp!(diameter_gen);
test_compiles_lumen_otp!(diameter_lib);
test_compiles_lumen_otp!(diameter_misc_sup);
test_compiles_lumen_otp!(diameter_peer);
test_compiles_lumen_otp!(diameter_peer_fsm);
test_compiles_lumen_otp!(diameter_peer_fsm_sup);
test_compiles_lumen_otp!(diameter_reg);
test_compiles_lumen_otp!(diameter_service);
test_compiles_lumen_otp!(diameter_service_sup);
test_compiles_lumen_otp!(diameter_session);
test_compiles_lumen_otp!(diameter_stats);
test_compiles_lumen_otp!(diameter_sup);
test_compiles_lumen_otp!(diameter_sync);
test_compiles_lumen_otp!(diameter_traffic);
test_compiles_lumen_otp!(diameter_types);
test_compiles_lumen_otp!(diameter_watchdog);
test_compiles_lumen_otp!(diameter_watchdog_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/diameter/src/base/");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("base")
}
