//! https://github.com/lumen/otp/tree/lumen/lib/runtime_tools/src

use super::*;

test_compiles_lumen_otp!(appmon_info);
test_compiles_lumen_otp!(dbg);
test_compiles_lumen_otp!(dyntrace);
test_compiles_lumen_otp!(erts_alloc_config);
test_compiles_lumen_otp!(msacc);
test_compiles_lumen_otp!(observer_backend);
test_compiles_lumen_otp!(runtime_tools imports "lib/stdlib/src/supervisor");
test_compiles_lumen_otp!(runtime_tools_sup);
test_compiles_lumen_otp!(scheduler);
test_compiles_lumen_otp!(system_information);
test_compiles_lumen_otp!(ttb_autostart imports "lib/kernel/src/application", "lib/kernel/src/file", "lib/runtime_tools/src/observer_backend", "lib/stdlib/src/gen_server");

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/kernel/include",
        "lib/kernel/src",
        "lib/runtime_tools/include",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("runtime_tools/src")
}
