//! https://github.com/lumen/otp/tree/lumen/lib/inets/src/inets_app

use super::*;

test_compiles_lumen_otp!(inets);
test_compiles_lumen_otp!(inets_app imports "lib/inets/src/inets_app/inets_sup");
test_compiles_lumen_otp!(inets_lib imports "lib/kernel/src/os", "lib/stdlib/src/calendar", "lib/stdlib/src/io_lib", "lib/stdlib/src/lists", "lib/stdlib/src/timer");
test_compiles_lumen_otp!(inets_service);
test_compiles_lumen_otp!(inets_sup);
test_compiles_lumen_otp!(inets_trace);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/kernel/include");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("inets_app")
}
