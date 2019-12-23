use std::ffi::CString;

extern "C" {
    pub fn LLVMLumenLink(argc: libc::c_int, argv: *const *const libc::c_char) -> libc::c_int;
}

/// Invoke the statically linked `lld` linker with the given arguments.
///
/// NOTE: Assumes that the first value of the argument vector contains the
/// program name which informs lld which flavor of linker is being run.
pub fn link(argv: &[CString]) -> Result<(), ()> {
    let argc = argv.len();
    let mut c_argv = Vec::with_capacity(argc);
    for arg in argv {
        c_argv.push(arg.as_ptr());
    }
    let result = unsafe { LLVMLumenLink(argc as libc::c_int, c_argv.as_ptr()) };
    if result != 0 {
        Ok(())
    } else {
        Err(())
    }
}
