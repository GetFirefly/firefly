use firefly_rt::term::AtomData;

extern "C-unwind" {
    /// This function is defined in `firefly_alloc::erts::term::atom`
    #[link_name = "__firefly_initialize_atom_table"]
    pub fn init(start: *const AtomData, end: *const AtomData) -> bool;
}

#[cfg(target_os = "macos")]
extern "C" {
    #[link_name = "\x01section$start$__DATA$__atoms"]
    static ATOMS_START: AtomData;

    #[link_name = "\x01section$end$__DATA$__atoms"]
    static ATOMS_END: AtomData;
}

#[cfg(all(unix, not(target_os = "macos")))]
extern "C" {
    #[link_name = "__start___atoms"]
    static ATOMS_START: AtomData;

    #[link_name = "__stop___atoms"]
    static ATOMS_END: AtomData;
}

pub(super) fn start() -> *const AtomData {
    unsafe { &ATOMS_START }
}

pub(super) fn end() -> *const AtomData {
    unsafe { &ATOMS_END }
}
