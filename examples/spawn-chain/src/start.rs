pub fn export_code() {
    liblumen_otp::erlang::apply_3::export();

    crate::elixir::chain::console_1::export();
    crate::elixir::chain::counter_2::export();
    crate::elixir::chain::create_processes_2::export();
    crate::elixir::chain::dom_1::export();
    crate::elixir::chain::on_submit_1::export();
}

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
