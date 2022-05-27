use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(windows)] {
        mod windows;
        use self::windows as arch;
    } else if #[cfg(unix)] {
        mod unix;
        use self::unix as arch;
    } else if #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))] {
        mod wasm32;
        use self::wasm32 as arch;
    } else {
        compile_error!("unsupported platform!");
    }
}

pub use self::arch::*;

pub type DynamicCallee = extern "C-unwind" fn() -> crate::function::ErlangResult;
