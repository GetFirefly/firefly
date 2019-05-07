// Emit custom cfg types:
//     cargo:rustc-cfg=has_foo
// Can then be used as `#[cfg(has_foo)]` when emitted

// Emit custom env data:
//     cargo:rustc-env=foo=bar
// Can then be fetched with `env!("foo")`

#[cfg(windows)]
fn main() {
    println!("cargo:rustc-cfg=has_mmap");
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn main() {
    println!("cargo:rustc-cfg=has_mmap");
}

#[cfg(target_arch = "wasm32")]
fn main() {}
