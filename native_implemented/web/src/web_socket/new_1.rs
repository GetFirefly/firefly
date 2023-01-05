//! ```elixir
//! case Lumen.Web.WebSocket.new(url) do
//!   {:ok, web_socket} -> ...
//!   {:error, reason} -> ...
//! end
//! ```

use web_sys::WebSocket;

use liblumen_alloc::erts::exception;
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::runtime::binary_to_string::binary_to_string;
use crate::{error_tuple, ok_tuple};

#[native_implemented::function(Elixir.Lumen.Web.WebSocket:new/1)]
pub fn result(process: &Process, url: Term) -> exception::Result<Term> {
    let url_string = binary_to_string(url)?;

    let result_tuple = match WebSocket::new(&url_string) {
        Ok(web_socket) => ok_tuple(process, web_socket),
        Err(js_value) => error_tuple(process, js_value),
    };

    Ok(result_tuple)
}
