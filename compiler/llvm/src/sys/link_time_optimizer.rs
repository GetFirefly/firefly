//! Link-time-optimization

/// Dummy type for pointers to the LTO object
#[allow(non_camel_case_types)]
pub type llvm_lto_t = *mut ::libc::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum llvm_lto_status_t {
    LLVM_LTO_UNKNOWN = 0,
    LLVM_LTO_OPT_SUCCESS = 1,
    LLVM_LTO_READ_SUCCESS = 2,
    LLVM_LTO_READ_FAILURE = 3,
    LLVM_LTO_WRITE_FAILURE = 4,
    LLVM_LTO_NO_TARGET = 5,
    LLVM_LTO_NO_WORK = 6,
    LLVM_LTO_MODULE_MERGE_FAILURE = 7,
    LLVM_LTO_ASM_FAILURE = 8,
    LLVM_LTO_NULL_OBJECT = 9,
}

extern "C" {
    pub fn llvm_create_optimizer() -> llvm_lto_t;
    pub fn llvm_destroy_optimizer(lto: llvm_lto_t);
    pub fn llvm_read_object_file(
        lto: llvm_lto_t,
        input_filename: *const ::libc::c_char,
    ) -> llvm_lto_status_t;
    pub fn llvm_optimize_modules(
        lto: llvm_lto_t,
        output_filename: *const ::libc::c_char,
    ) -> llvm_lto_status_t;
}
