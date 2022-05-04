/// This struct represents the serialized form of an atom table entry
///
/// Constant atoms found during compilation are serialized into a static
/// atom table which seeds the runtime atom table during initialization,
/// in order to connect the values of those constant atoms to the runtime
/// table entries, we serialize both the string value and the id, so that
/// constant usages can be replaced with a constant term value, and the
/// atom table will reflect the same data (i.e. the id matches the string)
#[repr(C)]
pub struct ConstantAtom {
    // The id of the atom, which will be used as the id in term form
    pub id: usize,
    // The string value of the atom.
    //
    // We use i8 here, which is equivalent to libc::c_char, but libc
    // is not universally available in this crate, so we use the former
    pub value: *const i8,
}
