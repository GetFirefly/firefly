//! https://github.com/lumen/otp/tree/lumen/lib/dialyzer/src

use super::*;

test_compiles_lumen_otp!(dialyzer);
test_compiles_lumen_otp!(dialyzer_analysis_callgraph);
test_compiles_lumen_otp!(dialyzer_behaviours);
test_compiles_lumen_otp!(dialyzer_callgraph);
test_compiles_lumen_otp!(dialyzer_cl);
test_compiles_lumen_otp!(dialyzer_cl_parse);
test_compiles_lumen_otp!(dialyzer_clean_core);
test_compiles_lumen_otp!(dialyzer_codeserver);
test_compiles_lumen_otp!(dialyzer_contracts);
test_compiles_lumen_otp!(dialyzer_coordinator);
test_compiles_lumen_otp!(dialyzer_dataflow);
test_compiles_lumen_otp!(dialyzer_dep);
test_compiles_lumen_otp!(dialyzer_explanation);
test_compiles_lumen_otp!(dialyzer_gui_wx);
test_compiles_lumen_otp!(dialyzer_options);
test_compiles_lumen_otp!(dialyzer_plt);
test_compiles_lumen_otp!(dialyzer_race_data_server imports "lib/stdlib/src/dict", "lib/stdlib/src/ordsets");
test_compiles_lumen_otp!(dialyzer_races);
test_compiles_lumen_otp!(dialyzer_succ_typings);
test_compiles_lumen_otp!(dialyzer_timing imports "lib/stdlib/src/io_lib", "lib/stdlib/src/io_lib");
test_compiles_lumen_otp!(dialyzer_typesig);
test_compiles_lumen_otp!(dialyzer_utils);
test_compiles_lumen_otp!(dialyzer_worker);
test_compiles_lumen_otp!(typer);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/dialyzer/src");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("dialyzer/src")
}
