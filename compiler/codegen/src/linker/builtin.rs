use std::ffi::CString;
use std::io;
use std::os;
use std::os::raw::{c_char, c_int};

use firefly_util::fs;

extern "C" {
    #[cfg(windows)]
    pub fn LLVMFireflyLink(
        argc: c_int,
        argv: *const *const c_char,
        stdout: os::windows::io::RawHandle,
        stderr: os::windows::io::RawHandle,
    ) -> bool;

    #[cfg(not(windows))]
    pub fn LLVMFireflyLink(
        argc: c_int,
        argv: *const *const c_char,
        stdout: os::unix::io::RawFd,
        stderr: os::unix::io::RawFd,
    ) -> bool;
}

/// Invoke the statically linked `lld` linker with the given arguments.
///
/// NOTE: Assumes that the first value of the argument vector contains the
/// program name which informs lld which flavor of linker is being run.
pub fn link(argv: &[CString]) -> Result<(), ()> {
    // Acquire exclusive access to stdout/stderr for the linker
    let stdout = io::stdout();
    let stderr = io::stderr();
    let stdout_lock = stdout.lock();
    let stderr_lock = stderr.lock();
    let stdout_fd = fs::get_file_descriptor(&stdout_lock);
    let stderr_fd = fs::get_file_descriptor(&stderr_lock);

    let argc = argv.len();
    let mut c_argv = Vec::with_capacity(argc);
    for arg in argv {
        c_argv.push(arg.as_ptr());
    }
    let is_ok = unsafe { LLVMFireflyLink(argc as c_int, c_argv.as_ptr(), stdout_fd, stderr_fd) };

    if is_ok {
        Ok(())
    } else {
        Err(())
    }
}
