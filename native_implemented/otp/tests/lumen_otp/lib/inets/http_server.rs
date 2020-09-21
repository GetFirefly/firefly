//! https://github.com/lumen/otp/tree/lumen/lib/inets/src/http_server

use super::*;

test_compiles_lumen_otp!(httpd);
test_compiles_lumen_otp!(httpd_acceptor);
test_compiles_lumen_otp!(httpd_acceptor_sup);
test_compiles_lumen_otp!(httpd_cgi);
test_compiles_lumen_otp!(httpd_conf);
test_compiles_lumen_otp!(httpd_connection_sup);
test_compiles_lumen_otp!(httpd_custom);
test_compiles_lumen_otp!(httpd_custom_api);
test_compiles_lumen_otp!(httpd_esi);
test_compiles_lumen_otp!(httpd_example);
test_compiles_lumen_otp!(httpd_file);
test_compiles_lumen_otp!(httpd_instance_sup imports "lib/inets/src/http_server/httpd_conf", "lib/inets/src/http_server/httpd_util", "lib/kernel/src/error_logger", "lib/stdlib/src/proplists", "lib/stdlib/src/supervisor");
test_compiles_lumen_otp!(httpd_log);
test_compiles_lumen_otp!(httpd_logger);
test_compiles_lumen_otp!(httpd_manager);
test_compiles_lumen_otp!(httpd_misc_sup);
test_compiles_lumen_otp!(httpd_request);
test_compiles_lumen_otp!(httpd_request_handler);
test_compiles_lumen_otp!(httpd_response);
test_compiles_lumen_otp!(httpd_script_env);
test_compiles_lumen_otp!(httpd_socket);
test_compiles_lumen_otp!(httpd_sup);
test_compiles_lumen_otp!(httpd_util);
test_compiles_lumen_otp!(mod_actions);
test_compiles_lumen_otp!(mod_alias);
test_compiles_lumen_otp!(mod_auth);
test_compiles_lumen_otp!(mod_auth_dets);
test_compiles_lumen_otp!(mod_auth_mnesia);
test_compiles_lumen_otp!(mod_auth_plain);
test_compiles_lumen_otp!(mod_auth_server);
test_compiles_lumen_otp!(mod_cgi);
test_compiles_lumen_otp!(mod_dir);
test_compiles_lumen_otp!(mod_disk_log);
test_compiles_lumen_otp!(mod_esi);
test_compiles_lumen_otp!(mod_get);
test_compiles_lumen_otp!(mod_head imports "lib/inets/src/http_server/httpd_util", "lib/inets/src/http_server/mod_alias", "lib/kernel/src/file", "lib/stdlib/src/io_lib", "lib/stdlib/src/lists", "lib/stdlib/src/proplists");
test_compiles_lumen_otp!(mod_log);
test_compiles_lumen_otp!(mod_range);
test_compiles_lumen_otp!(mod_responsecontrol);
test_compiles_lumen_otp!(mod_security);
test_compiles_lumen_otp!(mod_security_server);
test_compiles_lumen_otp!(mod_trace imports "lib/inets/src/http_server/httpd_socket", "lib/stdlib/src/lists", "lib/stdlib/src/proplists");

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/inets/src/http_lib",
        "lib/inets/src/http_server",
        "lib/inets/src/inets_app",
        "lib/kernel/include",
        "lib/kernel/src",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("http_server")
}
