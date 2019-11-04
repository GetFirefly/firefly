#[inline]
pub(crate) fn pagesize() -> usize {
    // Page size is always constant in WebAssembly, per the spec
    64 * 1024
}

#[inline]
pub fn get_num_cpus() -> usize {
    // In the unknown environment, we have no resources for determining
    // the number of available CPUs online, so if atomics are enabled, we
    // set a reasonable default of 2, otherwise 1
    if cfg!(target_feature = "atomics") {
        2
    } else {
        1
    }
}
