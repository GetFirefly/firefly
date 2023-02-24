use firefly_rt::function::ErlangResult;
use firefly_rt::process::ProcessLock;
use firefly_rt::term::*;

#[export_name = "file:native_name_encoding/0"]
pub extern "C-unwind" fn native_name_encoding(_process: &mut ProcessLock) -> ErlangResult {
    ErlangResult::Ok(atoms::Utf8.into())
}
