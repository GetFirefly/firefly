//! https://github.com/lumen/otp/tree/lumen/lib/ssl/src

use super::*;

test_compiles_lumen_otp!(dtls_connection);
test_compiles_lumen_otp!(dtls_connection_sup);
test_compiles_lumen_otp!(dtls_handshake);
test_compiles_lumen_otp!(dtls_listener_sup);
test_compiles_lumen_otp!(dtls_packet_demux);
test_compiles_lumen_otp!(dtls_record);
test_compiles_lumen_otp!(dtls_socket);
test_compiles_lumen_otp!(dtls_sup);
test_compiles_lumen_otp!(dtls_v1 imports "lib/crypto/src/crypto", "lib/ssl/src/ssl_cipher", "lib/ssl/src/ssl_cipher_format", "lib/ssl/src/tls_v1", "lib/stdlib/src/lists", "lib/stdlib/src/rand");
test_compiles_lumen_otp!(inet6_tls_dist imports "lib/ssl/src/inet_tls_dist");
test_compiles_lumen_otp!(inet_tls_dist);
test_compiles_lumen_otp!(ssl);
test_compiles_lumen_otp!(ssl_admin_sup);
test_compiles_lumen_otp!(ssl_alert);
test_compiles_lumen_otp!(ssl_app);
test_compiles_lumen_otp!(ssl_certificate);
test_compiles_lumen_otp!(ssl_cipher);
test_compiles_lumen_otp!(ssl_cipher_format);
test_compiles_lumen_otp!(ssl_config);
test_compiles_lumen_otp!(ssl_connection);
test_compiles_lumen_otp!(ssl_connection_sup);
test_compiles_lumen_otp!(ssl_crl);
test_compiles_lumen_otp!(ssl_crl_cache);
test_compiles_lumen_otp!(ssl_crl_cache_api);
test_compiles_lumen_otp!(ssl_crl_hash_dir);
test_compiles_lumen_otp!(ssl_dh_groups);
test_compiles_lumen_otp!(ssl_dist_admin_sup);
test_compiles_lumen_otp!(ssl_dist_connection_sup);
test_compiles_lumen_otp!(ssl_dist_sup);
test_compiles_lumen_otp!(ssl_handshake);
test_compiles_lumen_otp!(ssl_listen_tracker_sup);
test_compiles_lumen_otp!(ssl_logger);
test_compiles_lumen_otp!(ssl_manager);
test_compiles_lumen_otp!(ssl_pem_cache);
test_compiles_lumen_otp!(ssl_pkix_db);
test_compiles_lumen_otp!(ssl_record);
test_compiles_lumen_otp!(ssl_session);
test_compiles_lumen_otp!(ssl_session_cache imports "lib/stdlib/src/ets");
test_compiles_lumen_otp!(ssl_session_cache_api);
test_compiles_lumen_otp!(ssl_srp_primes);
test_compiles_lumen_otp!(ssl_sup);
test_compiles_lumen_otp!(tls_bloom_filter);
test_compiles_lumen_otp!(tls_client_ticket_store);
test_compiles_lumen_otp!(tls_connection);
test_compiles_lumen_otp!(tls_connection_1_3);
test_compiles_lumen_otp!(tls_connection_sup);
test_compiles_lumen_otp!(tls_handshake);
test_compiles_lumen_otp!(tls_handshake_1_3);
test_compiles_lumen_otp!(tls_record);
test_compiles_lumen_otp!(tls_record_1_3);
test_compiles_lumen_otp!(tls_sender);
test_compiles_lumen_otp!(tls_server_session_ticket);
test_compiles_lumen_otp!(tls_server_session_ticket_sup);
test_compiles_lumen_otp!(tls_server_sup);
test_compiles_lumen_otp!(tls_socket);
test_compiles_lumen_otp!(tls_sup);
test_compiles_lumen_otp!(tls_v1);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/public_key/include", "lib/ssl/src"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("ssl/src")
}

fn setup() {
    public_key::setup();
}
