//! https://github.com/lumen/otp/tree/lumen/lib/edoc/src

use super::*;

test_compiles_lumen_otp!(edoc);
test_compiles_lumen_otp!(edoc_data);
test_compiles_lumen_otp!(edoc_doclet);
test_compiles_lumen_otp!(edoc_extract);
test_compiles_lumen_otp!(edoc_layout);
test_compiles_lumen_otp!(edoc_lib);
test_compiles_lumen_otp!(edoc_macros);
test_compiles_lumen_otp!(edoc_refs imports "lib/edoc/src/edoc_lib");
test_compiles_lumen_otp!(edoc_report);
test_compiles_lumen_otp!(edoc_run);
test_compiles_lumen_otp!(edoc_scanner);
test_compiles_lumen_otp!(edoc_specs);
test_compiles_lumen_otp!(edoc_tags);
test_compiles_lumen_otp!(edoc_types);
test_compiles_lumen_otp!(edoc_wiki);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/edoc/include", "lib/edoc/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("edoc/src")
}
