//! https://github.com/lumen/otp/tree/lumen/lib/inets/src/http_client

use super::*;

test_compiles_lumen_otp!(httpc);
test_compiles_lumen_otp!(httpc_cookie);
test_compiles_lumen_otp!(httpc_handler);
test_compiles_lumen_otp!(httpc_handler_sup);
test_compiles_lumen_otp!(httpc_manager);
test_compiles_lumen_otp!(httpc_profile_sup);
test_compiles_lumen_otp!(httpc_request);
test_compiles_lumen_otp!(httpc_response);
test_compiles_lumen_otp!(httpc_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/inets/src/http_client");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("http_client")
}
