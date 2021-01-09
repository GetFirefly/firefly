//! https://github.com/lumen/otp/tree/lumen/lib/mnesia/src

use super::*;

test_compiles_lumen_otp!(mnesia);
test_compiles_lumen_otp!(mnesia_app imports "lib/mnesia/src/mnesia_sup");
test_compiles_lumen_otp!(mnesia_backend_type);
test_compiles_lumen_otp!(mnesia_backup);
test_compiles_lumen_otp!(mnesia_bup);
test_compiles_lumen_otp!(mnesia_checkpoint);
test_compiles_lumen_otp!(mnesia_checkpoint_sup);
test_compiles_lumen_otp!(mnesia_controller);
test_compiles_lumen_otp!(mnesia_dumper);
test_compiles_lumen_otp!(mnesia_event);
test_compiles_lumen_otp!(mnesia_ext_sup);
test_compiles_lumen_otp!(mnesia_frag);
test_compiles_lumen_otp!(mnesia_frag_hash);
test_compiles_lumen_otp!(mnesia_index);
test_compiles_lumen_otp!(mnesia_kernel_sup imports "lib/stdlib/src/supervisor", "lib/stdlib/src/timer");
test_compiles_lumen_otp!(mnesia_late_loader);
test_compiles_lumen_otp!(mnesia_lib);
test_compiles_lumen_otp!(mnesia_loader);
test_compiles_lumen_otp!(mnesia_locker);
test_compiles_lumen_otp!(mnesia_log);
test_compiles_lumen_otp!(mnesia_monitor);
test_compiles_lumen_otp!(mnesia_recover);
test_compiles_lumen_otp!(mnesia_registry);
test_compiles_lumen_otp!(mnesia_rpc);
test_compiles_lumen_otp!(mnesia_schema);
test_compiles_lumen_otp!(mnesia_snmp_hook);
test_compiles_lumen_otp!(mnesia_sp);
test_compiles_lumen_otp!(mnesia_subscr);
test_compiles_lumen_otp!(mnesia_sup);
test_compiles_lumen_otp!(mnesia_text);
test_compiles_lumen_otp!(mnesia_tm);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/mnesia/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("mnesia/src")
}
