#![allow(unused_macros)]
///! Support macros for the runtime, these should primarily be used to ease common
///! routines for interacting via FFI with the running program

/// Helper macro for converting String/str to a raw C pointer (* const i8)
macro_rules! c_str {
    ($s:expr) => {{
        use std::ffi::CString;
        CString::new($s).expect("invalid cast to C string").as_ptr() as *const i8
    }};
}

/// Helper macro for converting from a C string pointer (* libc::c_char) or (* const i8) to &str
#[allow(unused_unsafe)]
macro_rules! c_str_to_str {
    ($s:expr) => {{
        use std::ffi::CStr;
        #[allow(unused_unsafe)]
        unsafe {
            CStr::from_ptr($s)
                .to_str()
                .expect("invalid C string pointer")
        }
    }};
}

#[cfg(test)]
macro_rules! run {
    ($arc_process_fun:expr, $test:expr$(,)?) => {
        crate::test::run(file!(), $arc_process_fun, $test)
    };
}
