//! https://github.com/lumen/otp/tree/lumen/lib/diameter/src/transport

use super::*;

test_compiles_lumen_otp!(diameter_etcp);
test_compiles_lumen_otp!(diameter_etcp_sup);
test_compiles_lumen_otp!(diameter_sctp);
test_compiles_lumen_otp!(diameter_sctp_sup);
test_compiles_lumen_otp!(diameter_tcp);
test_compiles_lumen_otp!(diameter_tcp_sup);
test_compiles_lumen_otp!(diameter_transport imports "lib/diameter/src/base/diameter_lib");
test_compiles_lumen_otp!(diameter_transport_sup);

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("transport")
}
