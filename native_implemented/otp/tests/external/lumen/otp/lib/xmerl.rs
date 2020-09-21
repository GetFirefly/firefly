//! https://github.com/lumen/otp/tree/lumen/lib/xmerl/src

use super::*;

test_compiles_lumen_otp!(xmerl);
test_compiles_lumen_otp!(xmerl_b64Bin_scan);
test_compiles_lumen_otp!(xmerl_eventp);
test_compiles_lumen_otp!(xmerl_html);
test_compiles_lumen_otp!(xmerl_lib);
test_compiles_lumen_otp!(xmerl_otpsgml imports "lib/xmerl/src/xmerl_lib");
test_compiles_lumen_otp!(xmerl_regexp);
test_compiles_lumen_otp!(xmerl_sax_old_dom);
test_compiles_lumen_otp!(xmerl_sax_parser);
test_compiles_lumen_otp!(xmerl_sax_simple_dom);
test_compiles_lumen_otp!(xmerl_scan);
test_compiles_lumen_otp!(xmerl_sgml imports "lib/xmerl/src/xmerl_lib");
test_compiles_lumen_otp!(xmerl_simple);
test_compiles_lumen_otp!(xmerl_text);
test_compiles_lumen_otp!(xmerl_ucs);
test_compiles_lumen_otp!(xmerl_uri);
test_compiles_lumen_otp!(xmerl_validate);
test_compiles_lumen_otp!(xmerl_xlate);
test_compiles_lumen_otp!(xmerl_xml);
test_compiles_lumen_otp!(xmerl_xpath);
test_compiles_lumen_otp!(xmerl_xpath_lib);
test_compiles_lumen_otp!(xmerl_xpath_pred);
test_compiles_lumen_otp!(xmerl_xpath_scan);
test_compiles_lumen_otp!(xmerl_xs);
test_compiles_lumen_otp!(xmerl_xsd);
test_compiles_lumen_otp!(xmerl_xsd_type);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/xmerl/include", "lib/xmerl/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("xmerl/src")
}
