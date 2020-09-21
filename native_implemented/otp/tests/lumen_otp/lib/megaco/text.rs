//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src/text

use super::*;

test_compiles_lumen_otp!(megaco_compact_text_encoder);
test_compiles_lumen_otp!(megaco_compact_text_encoder_v1);
test_compiles_lumen_otp!(megaco_compact_text_encoder_v2);
test_compiles_lumen_otp!(megaco_compact_text_encoder_v3);
test_compiles_lumen_otp!(megaco_pretty_text_encoder);
test_compiles_lumen_otp!(megaco_pretty_text_encoder_v1);
test_compiles_lumen_otp!(megaco_pretty_text_encoder_v2);
test_compiles_lumen_otp!(megaco_pretty_text_encoder_v3);
test_compiles_lumen_otp!(megaco_text_mini_decoder imports "lib/megaco/src/text/megaco_text_scanner");
test_compiles_lumen_otp!(megaco_text_scanner);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/megaco/src/text");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("text")
}
