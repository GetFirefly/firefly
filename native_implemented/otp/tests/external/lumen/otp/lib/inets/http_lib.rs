//! https://github.com/lumen/otp/tree/lumen/lib/inets/src/http_lib

use super::*;

test_compiles_lumen_otp!(http_chunk);
test_compiles_lumen_otp!(http_request);
test_compiles_lumen_otp!(http_response);
test_compiles_lumen_otp!(http_transport);
test_compiles_lumen_otp!(http_uri);
test_compiles_lumen_otp!(http_util);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/inets/src/http_lib");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("http_lib")
}
