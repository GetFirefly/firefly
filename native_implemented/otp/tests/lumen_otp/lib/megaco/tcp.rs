//! https://github.com/lumen/otp/tree/lumen/lib/megaco/src/tcp

use super::*;

test_compiles_lumen_otp!(megaco_tcp);
test_compiles_lumen_otp!(megaco_tcp_accept imports "lib/kernel/src/gen_tcp", "lib/megaco/src/tcp/megaco_tcp", "lib/stdlib/src/proc_lib");
test_compiles_lumen_otp!(megaco_tcp_accept_sup);
test_compiles_lumen_otp!(megaco_tcp_connection);
test_compiles_lumen_otp!(megaco_tcp_connection_sup);
test_compiles_lumen_otp!(megaco_tcp_sup imports "lib/stdlib/src/lists", "lib/stdlib/src/supervisor");

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/kernel/include/", "lib/kernel/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("tcp")
}
