//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src/engine

use super::*;

test_compiles_lumen_otp!(megaco_config);
test_compiles_lumen_otp!(megaco_config_misc);
test_compiles_lumen_otp!(megaco_digit_map);
test_compiles_lumen_otp!(megaco_edist_compress);
test_compiles_lumen_otp!(megaco_encoder);
test_compiles_lumen_otp!(megaco_erl_dist_encoder);
test_compiles_lumen_otp!(megaco_erl_dist_encoder_mc);
test_compiles_lumen_otp!(megaco_filter);
test_compiles_lumen_otp!(megaco_messenger);
test_compiles_lumen_otp!(megaco_messenger_misc);
test_compiles_lumen_otp!(megaco_misc_sup);
test_compiles_lumen_otp!(megaco_monitor);
test_compiles_lumen_otp!(megaco_sdp);
test_compiles_lumen_otp!(megaco_stats);
test_compiles_lumen_otp!(megaco_sup);
test_compiles_lumen_otp!(megaco_timer);
test_compiles_lumen_otp!(megaco_trans_sender);
test_compiles_lumen_otp!(megaco_trans_sup);
test_compiles_lumen_otp!(megaco_transport);
test_compiles_lumen_otp!(megaco_user);
test_compiles_lumen_otp!(megaco_user_default);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/megaco/src/engine");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("engine")
}
