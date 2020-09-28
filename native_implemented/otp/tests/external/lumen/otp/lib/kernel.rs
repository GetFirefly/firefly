//! https://github.com/lumen/otp/tree/lumen/lib/kernel/src

use std::process::Command;
use std::time::Duration;

use super::*;

test_compiles_lumen_otp!(application);
test_compiles_lumen_otp!(application_controller);
test_compiles_lumen_otp!(application_master);
test_compiles_lumen_otp!(application_starter);
test_compiles_lumen_otp!(auth);
test_compiles_lumen_otp!(code);
test_compiles_lumen_otp!(code_server);
test_compiles_lumen_otp!(disk_log);
test_compiles_lumen_otp!(disk_log_1);
test_compiles_lumen_otp!(disk_log_server);
test_compiles_lumen_otp!(disk_log_sup);
test_compiles_lumen_otp!(dist_ac);
test_compiles_lumen_otp!(dist_util);
test_compiles_lumen_otp!(erl_boot_server);
test_compiles_lumen_otp!(erl_compile_server);
test_compiles_lumen_otp!(erl_ddll);
test_compiles_lumen_otp!(erl_distribution);
test_compiles_lumen_otp!(erl_epmd);
test_compiles_lumen_otp!(erl_reply imports "lib/kernel/src/error_logger", "lib/kernel/src/gen_tcp", "lib/stdlib/src/string");
test_compiles_lumen_otp!(erl_signal_handler);
test_compiles_lumen_otp!(erpc);
test_compiles_lumen_otp!(error_handler);
test_compiles_lumen_otp!(error_logger);
test_compiles_lumen_otp!(erts_debug);
test_compiles_lumen_otp!(file);
test_compiles_lumen_otp!(file_io_server);
test_compiles_lumen_otp!(file_server);
test_compiles_lumen_otp!(gen_sctp);
test_compiles_lumen_otp!(gen_tcp);
test_compiles_lumen_otp!(gen_tcp_socket);
test_compiles_lumen_otp!(gen_udp);
test_compiles_lumen_otp!(global);
test_compiles_lumen_otp!(global_group);
test_compiles_lumen_otp!(global_search);
test_compiles_lumen_otp!(group);
test_compiles_lumen_otp!(group_history);
test_compiles_lumen_otp!(heart);
test_compiles_lumen_otp!(inet);
test_compiles_lumen_otp!(inet6_sctp);
test_compiles_lumen_otp!(inet6_tcp);
test_compiles_lumen_otp!(inet6_tcp_dist imports "lib/kernel/src/inet_tcp", "lib/kernel/src/inet_tcp_dist");
test_compiles_lumen_otp!(inet6_udp);
test_compiles_lumen_otp!(inet_config);
test_compiles_lumen_otp!(inet_db);
test_compiles_lumen_otp!(inet_dns);
test_compiles_lumen_otp!(inet_gethost_native);
test_compiles_lumen_otp!(inet_hosts);
test_compiles_lumen_otp!(inet_parse);
test_compiles_lumen_otp!(inet_res);
test_compiles_lumen_otp!(inet_sctp);
test_compiles_lumen_otp!(inet_tcp);
test_compiles_lumen_otp!(inet_tcp_dist);
test_compiles_lumen_otp!(inet_udp);
test_compiles_lumen_otp!(kernel);
test_compiles_lumen_otp!(kernel_config);
test_compiles_lumen_otp!(kernel_refc);
test_compiles_lumen_otp!(local_tcp);
test_compiles_lumen_otp!(local_udp);
test_compiles_lumen_otp!(logger);
test_compiles_lumen_otp!(logger_backend);
test_compiles_lumen_otp!(logger_config);
test_compiles_lumen_otp!(logger_disk_log_h);
test_compiles_lumen_otp!(logger_filters);
test_compiles_lumen_otp!(logger_formatter);
test_compiles_lumen_otp!(logger_h_common);
test_compiles_lumen_otp!(logger_handler_watcher);
test_compiles_lumen_otp!(logger_olp);
test_compiles_lumen_otp!(logger_proxy);
test_compiles_lumen_otp!(logger_server);
test_compiles_lumen_otp!(logger_simple_h);
test_compiles_lumen_otp!(logger_std_h);
test_compiles_lumen_otp!(logger_sup);
test_compiles_lumen_otp!(net);
test_compiles_lumen_otp!(net_adm);
test_compiles_lumen_otp!(net_kernel);
test_compiles_lumen_otp!(os);
test_compiles_lumen_otp!(pg);
test_compiles_lumen_otp!(pg2);
test_compiles_lumen_otp!(ram_file);
test_compiles_lumen_otp!(raw_file_io);
test_compiles_lumen_otp!(raw_file_io_compressed);
test_compiles_lumen_otp!(raw_file_io_deflate);
test_compiles_lumen_otp!(raw_file_io_delayed);
test_compiles_lumen_otp!(raw_file_io_inflate);
test_compiles_lumen_otp!(raw_file_io_list);
test_compiles_lumen_otp!(raw_file_io_raw imports "erts/preloaded/src/prim_file");
test_compiles_lumen_otp!(rpc);
test_compiles_lumen_otp!(seq_trace);
test_compiles_lumen_otp!(socket);
test_compiles_lumen_otp!(standard_error);
test_compiles_lumen_otp!(user);
test_compiles_lumen_otp!(user_drv);
test_compiles_lumen_otp!(user_sup imports "erts/preloaded/src/init", "lib/kernel/src/error_logger", "lib/kernel/src/rpc", "lib/stdlib/src/supervisor");
test_compiles_lumen_otp!(wrap_log_reader);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/kernel/include/", "lib/kernel/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("kernel/src")
}

fn setup() {
    let working_directory = lumen_otp_directory().join("lib/kernel/src");

    let mut command = Command::new("make");
    command
        .current_dir(&working_directory)
        .arg("inet_dns_record_adts.hrl");

    if let Err((command, output)) = crate::test::timeout(
        "make inet_dns_record_adts.hrl",
        working_directory.clone(),
        command,
        Duration::from_secs(10),
    ) {
        crate::test::command_failed(
            "make inet_dns_record_adts.hrl",
            working_directory,
            command,
            output,
        )
    }
}
