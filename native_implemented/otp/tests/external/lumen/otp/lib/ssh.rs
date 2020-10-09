//! https://github.com/lumen/otp/tree/lumen/lib/ssh/src

use super::*;

test_compiles_lumen_otp!(ssh);
test_compiles_lumen_otp!(ssh_acceptor);
test_compiles_lumen_otp!(ssh_acceptor_sup);
test_compiles_lumen_otp!(ssh_agent);
test_compiles_lumen_otp!(ssh_app imports "lib/stdlib/src/supervisor");
test_compiles_lumen_otp!(ssh_auth);
test_compiles_lumen_otp!(ssh_bits imports "lib/crypto/src/crypto", "lib/stdlib/src/lists");
test_compiles_lumen_otp!(ssh_channel imports "lib/ssh/src/ssh_client_channel");
test_compiles_lumen_otp!(ssh_channel_sup);
test_compiles_lumen_otp!(ssh_cli);
test_compiles_lumen_otp!(ssh_client_channel);
test_compiles_lumen_otp!(ssh_client_key_api);
test_compiles_lumen_otp!(ssh_connection);
test_compiles_lumen_otp!(ssh_connection_handler);
test_compiles_lumen_otp!(ssh_connection_sup imports "lib/stdlib/src/supervisor");
test_compiles_lumen_otp!(ssh_daemon_channel imports "lib/ssh/ssh_server_channel");
test_compiles_lumen_otp!(ssh_dbg);
test_compiles_lumen_otp!(ssh_file);
test_compiles_lumen_otp!(ssh_info);
test_compiles_lumen_otp!(ssh_io);
test_compiles_lumen_otp!(ssh_message);
test_compiles_lumen_otp!(ssh_no_io);
test_compiles_lumen_otp!(ssh_options);
test_compiles_lumen_otp!(ssh_server_channel imports "lib/ssh/ssh_client_channel");
test_compiles_lumen_otp!(ssh_server_key_api);
test_compiles_lumen_otp!(ssh_sftp);
test_compiles_lumen_otp!(ssh_sftpd);
test_compiles_lumen_otp!(ssh_sftpd_file imports "lib/kernel/src/file", "lib/stdlib/src/filelib");
test_compiles_lumen_otp!(ssh_sftpd_file_api);
test_compiles_lumen_otp!(ssh_shell);
test_compiles_lumen_otp!(ssh_subsystem_sup);
test_compiles_lumen_otp!(ssh_sup);
test_compiles_lumen_otp!(ssh_system_sup);
test_compiles_lumen_otp!(ssh_tcpip_forward_acceptor);
test_compiles_lumen_otp!(ssh_tcpip_forward_acceptor_sup imports "lib/stdlib/src/supervisor");
test_compiles_lumen_otp!(ssh_tcpip_forward_client);
test_compiles_lumen_otp!(ssh_tcpip_forward_srv);
test_compiles_lumen_otp!(ssh_transport);
test_compiles_lumen_otp!(ssh_xfer);
test_compiles_lumen_otp!(sshc_sup);
test_compiles_lumen_otp!(sshd_sup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/kernel/include",
        "lib/kernel/src",
        "lib/public_key/include",
        "lib/ssh/src",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("ssh/src")
}

fn setup() {
    public_key::setup();
}
