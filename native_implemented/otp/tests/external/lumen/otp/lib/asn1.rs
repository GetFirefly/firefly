//! https://github.com/lumen/otp/tree/lumen/lib/asn1/src

use super::*;

test_compiles_lumen_otp!(asn1_app);
test_compiles_lumen_otp!(asn1_db);
test_compiles_lumen_otp!(asn1ct);
// FIXME takes > 60 seconds
test_compiles_lumen_otp!(asn1ct_check);
test_compiles_lumen_otp!(asn1ct_constructed_ber_bin_v2);
test_compiles_lumen_otp!(asn1ct_constructed_per);
test_compiles_lumen_otp!(asn1ct_func);
test_compiles_lumen_otp!(asn1ct_gen);
test_compiles_lumen_otp!(asn1ct_gen_ber_bin_v2);
test_compiles_lumen_otp!(asn1ct_gen_check);
test_compiles_lumen_otp!(asn1ct_gen_jer);
test_compiles_lumen_otp!(asn1ct_gen_per);
test_compiles_lumen_otp!(asn1ct_imm);
test_compiles_lumen_otp!(asn1ct_name);
test_compiles_lumen_otp!(asn1ct_parser2);
test_compiles_lumen_otp!(asn1ct_pretty_format);
test_compiles_lumen_otp!(asn1ct_table);
test_compiles_lumen_otp!(asn1ct_tok);
test_compiles_lumen_otp!(asn1ct_value);
test_compiles_lumen_otp!(asn1rt);
test_compiles_lumen_otp!(asn1rt_nif);
test_compiles_lumen_otp!(asn1rtt_ber);
test_compiles_lumen_otp!(asn1rtt_check);
test_compiles_lumen_otp!(asn1rtt_ext);
test_compiles_lumen_otp!(asn1rtt_jer);
test_compiles_lumen_otp!(asn1rtt_per);
test_compiles_lumen_otp!(asn1rtt_per_common);
test_compiles_lumen_otp!(asn1rtt_real_common);
test_compiles_lumen_otp!(asn1rtt_uper);
test_compiles_lumen_otp!(prepare_templates);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/asn1/src");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("asn1/src")
}
