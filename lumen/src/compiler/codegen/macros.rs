/// Helper macro for converting String/str to a C string pointer
macro_rules! c_str {
    ($s:expr) => {{
        use std::ffi::CString;
        CString::new($s).expect("invalid cast to C string").as_ptr() as *const i8
    }};
}

/// Helper macro for converting from a C string pointer to &str
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
