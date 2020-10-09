//! https://github.com/lumen/otp/tree/lumen/erts/preloaded/src

use super::*;

test_compiles_lumen_otp!(atomics);
test_compiles_lumen_otp!(counters imports "erts/preloaded/src/atomics");
test_compiles_lumen_otp!(erl_init imports "erts/preloaded/src/erl_tracer", "erts/preloaded/src/prim_buffer", "erts/preloaded/src/prim_file", "erts/preloaded/src/zlib");
test_compiles_lumen_otp!(erl_prim_loader);
test_compiles_lumen_otp!(erl_tracer);
test_compiles_lumen_otp!(erlang);
test_compiles_lumen_otp!(erts_code_purger);
test_compiles_lumen_otp!(erts_dirty_process_signal_handler);
test_compiles_lumen_otp!(erts_internal);
test_compiles_lumen_otp!(erts_literal_area_collector);
test_compiles_lumen_otp!(init);
test_compiles_lumen_otp!(persistent_term);
test_compiles_lumen_otp!(prim_buffer);
test_compiles_lumen_otp!(prim_eval);
test_compiles_lumen_otp!(prim_file);
test_compiles_lumen_otp!(prim_inet);
test_compiles_lumen_otp!(prim_net imports "erts/preloaded/src/prim_socket");
test_compiles_lumen_otp!(prim_socket);
test_compiles_lumen_otp!(prim_zip);
test_compiles_lumen_otp!(socket_registry);
test_compiles_lumen_otp!(zlib);

fn includes() -> Vec<&'static str> {
    vec![
        "erts/preloaded/src",
        "lib",
        "lib/kernel/include",
        "lib/kernel/src",
    ]
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("preloaded/src")
}
