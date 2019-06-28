#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkerFlavor {
    Invalid = 0usize,
    Gnu = 1usize,
    Darwin = 2usize,
    Windows = 3usize,
    Wasm = 4usize,
}
impl LinkerFlavor {
    pub fn get() -> Self {
        if cfg!(target_arch = "wasm32") || cfg!(target_arch = "wasm64") {
            LinkerFlavor::Wasm
        } else if cfg!(target_family = "windows") {
            LinkerFlavor::Windows
        } else if cfg!(target_env = "ios") || cfg!(target_vendor = "apple") {
            LinkerFlavor::Darwin
        } else {
            LinkerFlavor::Gnu
        }
    }
}
impl Default for LinkerFlavor {
    fn default() -> Self {
        LinkerFlavor::get()
    }
}
impl Into<Flavor> for LinkerFlavor {
    fn into(self) -> Flavor {
        self as usize
    }
}

type Flavor = usize;

extern "C" {
    // Invokes the native linker
    pub fn lumen_lld(flavor: Flavor, argv: *const *const libc::c_char, argc: libc::c_int) -> bool;
}
