//! https://github.com/lumen/otp/tree/lumen/lib/erl_docgen/src

use super::*;

test_compiles_lumen_otp!(docgen_edoc_xml_cb);
test_compiles_lumen_otp!(docgen_otp_specs);
test_compiles_lumen_otp!(docgen_xmerl_xml_cb);
test_compiles_lumen_otp!(docgen_xml_check);
test_compiles_lumen_otp!(docgen_xml_to_chunk);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/xmerl/include");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("erl_docgen/src")
}
