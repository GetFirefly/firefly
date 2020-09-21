//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src/udp

use super::*;

test_compiles_lumen_otp!(megaco_udp);
test_compiles_lumen_otp!(megaco_udp_server);
test_compiles_lumen_otp!(megaco_udp_sup);

fn includes() -> Vec<&'static str> {
    let includes = super::includes();
    // includes.push("lib/inets/src/http_client");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("udp")
}
