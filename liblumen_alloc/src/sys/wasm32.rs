mod malloc;

pub use malloc::{
    alloc,
    alloc_zeroed,
    realloc,
    free,
};


#[inline]
pub(crate) fn pagesize() -> usize {
    // Page size is always constant in WebAssembly, per the spec
    64 * 1024
}


#[inline]
pub fn get_num_cpus() -> usize {
    // No multi-threading for now
    1
}
