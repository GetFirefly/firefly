//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src/binary

use super::*;

test_compiles_lumen_otp!(megaco_ber_encoder imports "lib/megaco/src/binary/megaco_binary_encoder_lib");
test_compiles_lumen_otp!(megaco_binary_encoder imports "lib/megaco/src/binary/megaco_binary_encoder_lib");
test_compiles_lumen_otp!(megaco_binary_encoder_lib);
test_compiles_lumen_otp!(megaco_binary_name_resolver_v1);
test_compiles_lumen_otp!(megaco_binary_name_resolver_v2);
test_compiles_lumen_otp!(megaco_binary_name_resolver_v3);
test_compiles_lumen_otp!(megaco_binary_term_id);
test_compiles_lumen_otp!(megaco_binary_term_id_gen);
test_compiles_lumen_otp!(megaco_binary_transformer_v1);
test_compiles_lumen_otp!(megaco_binary_transformer_v2);
test_compiles_lumen_otp!(megaco_binary_transformer_v3);
test_compiles_lumen_otp!(megaco_per_encoder imports "lib/megaco/src/binary/megaco_binary_encoder_lib");

fn includes() -> Vec<&'static str> {
    let includes = super::includes();
    // includes.push("lib/inets/src/http_client");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("binary")
}
