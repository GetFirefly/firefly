//! https://github.com/lumen/otp/tree/lumen/lib/os_mon/src

use super::*;

test_compiles_lumen_otp!(cpu_sup);
test_compiles_lumen_otp!(disksup);
test_compiles_lumen_otp!(memsup);
test_compiles_lumen_otp!(nteventlog);
test_compiles_lumen_otp!(os_mon);
test_compiles_lumen_otp!(os_mon_mib);
test_compiles_lumen_otp!(os_mon_sysinfo);
test_compiles_lumen_otp!(os_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/os_mon/include");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("os_mon/src")
}
